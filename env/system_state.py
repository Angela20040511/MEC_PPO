"""系统动态状态。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from config import MECConfig, BsTaskKey, LinkTaskKey, LocalTaskKey, TaskKey


@dataclass
class SystemState:
    """保存 MEC 系统在当前时隙的全部状态量。"""

    config: MECConfig
    slot: int = 0
    rng: np.random.Generator = field(init=False)
    mmpp_states: dict[TaskKey, int] = field(default_factory=dict)
    lambda_imp: dict[TaskKey, float] = field(default_factory=dict)
    data_rate_bits: dict[TaskKey, float] = field(default_factory=dict)
    workload_rate_cycles: dict[TaskKey, float] = field(default_factory=dict)
    access_queue_bits: dict[str, float] = field(default_factory=dict)
    virtual_queue_bits: dict[LinkTaskKey, float] = field(default_factory=dict)
    bs_queue_cycles: dict[BsTaskKey, float] = field(default_factory=dict)
    local_queue_cycles: dict[LocalTaskKey, float] = field(default_factory=dict)
    bs_energy_reference: dict[str, float] = field(default_factory=dict)
    bs_energy_budget: dict[str, float] = field(default_factory=dict)
    bs_energy_pressure: dict[str, float] = field(default_factory=dict)
    arrival_history: dict[str, list[np.ndarray]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """初始化完成后立即执行重置。"""
        self.reset(self.config.training.seed)

    def reset(self, seed: int | None = None) -> None:
        """重置 MMPP 状态、队列和能耗状态。"""
        self.rng = np.random.default_rng(seed or self.config.training.seed)
        self.slot = 0
        self.mmpp_states = {
            (sensor_id, task_name): 2
            for sensor_id in self.config.sensor_ids
            for task_name in self.config.task_names
        }
        self.access_queue_bits = {link_id: 0.0 for link_id in self.config.access_links}
        self.virtual_queue_bits = {
            (link_id, task_name): 0.0
            for link_id in self.config.access_links
            for task_name in self.config.task_names
        }
        self.bs_queue_cycles = {
            (bs_id, task_name): 0.0
            for bs_id in self.config.base_station_ids
            for task_name in self.config.task_names
        }
        self.local_queue_cycles = {
            (sensor_id, task_name): 0.0
            for sensor_id in self.config.sensor_ids
            for task_name in self.config.task_names
        }
        self.bs_energy_reference = self._build_bs_energy_reference()
        self.bs_energy_budget = {
            bs_id: self.config.base_station_by_id(bs_id).energy_pressure_baseline * self.bs_energy_reference[bs_id]
            for bs_id in self.config.base_station_ids
        }
        self.bs_energy_pressure = {
            bs_id: float(
                np.clip(
                    self.bs_energy_budget[bs_id] / max(self.bs_energy_reference[bs_id], self.config.system.epsilon),
                    0.0,
                    1.0,
                )
            )
            for bs_id in self.config.base_station_ids
        }
        self.arrival_history = {sensor_id: [] for sensor_id in self.config.sensor_ids}

        for _ in range(self.config.dt.history_window):
            self._sample_current_arrivals(advance_markov=True)
            self._append_history()

        self.slot = 0

    def _build_bs_energy_reference(self) -> dict[str, float]:
        """构造基站能耗归一化参考值。"""
        references: dict[str, float] = {}
        for bs in self.config.base_stations:
            max_energy = 0.0
            for unit in bs.compute_units:
                max_unit_rate = max(unit.acceleration[task_name] for task_name in self.config.task_names)
                max_energy += unit.energy_coeff * unit.base_rate * max_unit_rate * self.config.system.slot_length
            references[bs.bs_id] = max(max_energy, self.config.system.epsilon)
        return references

    def _sample_current_arrivals(self, advance_markov: bool) -> None:
        """按 MMPP 模型采样当前时隙到达。"""
        slot_length = self.config.system.slot_length
        new_lambda: dict[TaskKey, float] = {}
        new_data_rate: dict[TaskKey, float] = {}
        new_workload_rate: dict[TaskKey, float] = {}

        for sensor_id in self.config.sensor_ids:
            for task_name in self.config.task_names:
                key = (sensor_id, task_name)
                task_spec = self.config.task_by_name(task_name)
                state_id = self.mmpp_states[key]
                lambda_imp = task_spec.mmpp_rates[state_id]
                task_count = self.rng.poisson(lambda_imp * slot_length)

                if task_count > 0:
                    sampled_sizes = self.rng.normal(
                        loc=task_spec.mean_data_bits,
                        scale=task_spec.std_data_bits,
                        size=task_count,
                    )
                    sampled_sizes = np.clip(sampled_sizes, 0.25 * task_spec.mean_data_bits, None)
                    data_rate = float(sampled_sizes.sum() / slot_length)
                else:
                    data_rate = 0.0

                new_lambda[key] = float(lambda_imp)
                new_data_rate[key] = data_rate
                new_workload_rate[key] = float(task_spec.compute_intensity * data_rate)

                if advance_markov:
                    transition = self.config.system.mmpp_transition[state_id - 1]
                    self.mmpp_states[key] = int(self.rng.choice([1, 2, 3], p=transition))

        self.lambda_imp = new_lambda
        self.data_rate_bits = new_data_rate
        self.workload_rate_cycles = new_workload_rate

    def _append_history(self) -> None:
        """把当前到达率写入历史窗口。"""
        for sensor_id in self.config.sensor_ids:
            history_point = np.array(
                [self.lambda_imp[(sensor_id, task_name)] for task_name in self.config.task_names],
                dtype=np.float32,
            )
            self.arrival_history[sensor_id].append(history_point)

    def advance_slot(self) -> None:
        """推进到下一个时隙。"""
        self.slot += 1
        self._sample_current_arrivals(advance_markov=True)
        self._append_history()

    def update_energy_state(self, current_bs_energy: dict[str, float]) -> None:
        """更新基站能耗预算和能耗压力。"""
        alpha = self.config.system.energy_budget_alpha
        for bs_id in self.config.base_station_ids:
            previous_budget = self.bs_energy_budget.get(bs_id, 0.0)
            current_energy = current_bs_energy.get(bs_id, 0.0)
            self.bs_energy_budget[bs_id] = alpha * previous_budget + (1.0 - alpha) * current_energy
            reference = max(self.bs_energy_reference[bs_id], self.config.system.epsilon)
            # 显式定义 bs_energy_pressure[m] = clip(energy_budget[m] / energy_ref[m], 0, 1)
            self.bs_energy_pressure[bs_id] = float(
                np.clip(self.bs_energy_budget[bs_id] / reference, 0.0, 1.0)
            )

    def get_recent_history(self, sensor_id: str) -> np.ndarray:
        """返回最近 W 个时隙的到达率历史。"""
        history = self.arrival_history[sensor_id]
        window = self.config.dt.history_window
        if len(history) >= window:
            return np.stack(history[-window:], axis=0)

        padding = [history[0]] * (window - len(history)) if history else [np.zeros(len(self.config.tasks))] * window
        return np.stack([*padding, *history], axis=0)

    def queue_backlog_sum(self) -> float:
        """返回所有队列 backlog 总和。"""
        return float(
            sum(self.access_queue_bits.values())
            + sum(self.virtual_queue_bits.values())
            + sum(self.bs_queue_cycles.values())
            + sum(self.local_queue_cycles.values())
        )

    def normalized_queue_backlog_sum(self) -> float:
        """返回按队列类型归一化后的 backlog 总和。"""
        norm_bits = max(self.config.system.queue_norm_bits, self.config.system.epsilon)
        norm_cycles = max(self.config.system.queue_norm_cycles, self.config.system.epsilon)

        access_part = sum(value / norm_bits for value in self.access_queue_bits.values())
        virtual_part = sum(value / norm_bits for value in self.virtual_queue_bits.values())
        bs_part = sum(value / norm_cycles for value in self.bs_queue_cycles.values())
        local_part = sum(value / norm_cycles for value in self.local_queue_cycles.values())

        return float(access_part + virtual_part + bs_part + local_part)

    def normalized_average_queue_len(self) -> float:
        """返回归一化后的平均队列长度。"""
        queue_count = (
                len(self.access_queue_bits)
                + len(self.virtual_queue_bits)
                + len(self.bs_queue_cycles)
                + len(self.local_queue_cycles)
        )
        return self.normalized_queue_backlog_sum() / max(queue_count, 1)

    def lyapunov_value(self) -> float:
        """计算当前状态的 Lyapunov 函数值。"""
        return 0.5 * float(
            sum(value * value for value in self.access_queue_bits.values())
            + sum(value * value for value in self.virtual_queue_bits.values())
            + sum(value * value for value in self.bs_queue_cycles.values())
            + sum(value * value for value in self.local_queue_cycles.values())
        )

    def build_state_vector(
        self,
        predicted_loads: dict[TaskKey, float],
        access_gains: dict[str, float],
        effective_compute_rates: dict[tuple[str, str, str], float],
    ) -> np.ndarray:
        """构造提供给 PPO 的归一化状态向量。"""
        values: list[float] = []
        max_access_gain = max(max(access_gains.values(), default=1.0), self.config.system.epsilon)
        max_compute_rate = max(
            max(effective_compute_rates.values(), default=1.0),
            self.config.system.epsilon,
        )

        for sensor_id in self.config.sensor_ids:
            for task_name in self.config.task_names:
                task_key = (sensor_id, task_name)
                values.append(
                    self.lambda_imp[task_key]
                    / max(self.config.max_lambda_for_task(task_name), self.config.system.epsilon)
                )
                values.append(
                    self.data_rate_bits[task_key]
                    / max(self.config.max_data_rate_for_task(task_name), self.config.system.epsilon)
                )
                values.append(
                    self.workload_rate_cycles[task_key]
                    / max(self.config.max_workload_rate_for_task(task_name), self.config.system.epsilon)
                )
                values.append(
                    predicted_loads[task_key]
                    / max(
                        self.config.dt.prediction_horizon * self.config.max_lambda_for_task(task_name),
                        self.config.system.epsilon,
                    )
                )

        for link_id in self.config.access_links:
            values.append(self.access_queue_bits[link_id] / self.config.system.queue_norm_bits)

        for link_id in self.config.access_links:
            for task_name in self.config.task_names:
                values.append(self.virtual_queue_bits[(link_id, task_name)] / self.config.system.queue_norm_bits)

        for bs_id in self.config.base_station_ids:
            for task_name in self.config.task_names:
                values.append(self.bs_queue_cycles[(bs_id, task_name)] / self.config.system.queue_norm_cycles)

        for sensor_id in self.config.sensor_ids:
            for task_name in self.config.task_names:
                values.append(self.local_queue_cycles[(sensor_id, task_name)] / self.config.system.queue_norm_cycles)

        for sensor_id in self.config.sensor_ids:
            reachable = self.config.reachable_base_stations(sensor_id)
            reachable_mask = [1.0 if bs_id in reachable else 0.0 for bs_id in self.config.base_station_ids]
            values.extend(reachable_mask)

        for sensor_id in self.config.sensor_ids:
            link_id = self.config.access_link_id(sensor_id)
            values.append(access_gains[link_id] / max_access_gain)

        for bs_id in self.config.base_station_ids:
            for task_name in self.config.task_names:
                base_station = self.config.base_station_by_id(bs_id)
                for unit in base_station.compute_units:
                    values.append(
                        effective_compute_rates[(bs_id, unit.unit_name, task_name)] / max_compute_rate
                    )

        return np.asarray(values, dtype=np.float32)

    def get_state(self) -> dict[str, Any]:
        """以字典形式导出当前状态。"""
        return {
            "slot": self.slot,
            "lambda": dict(self.lambda_imp),
            "data_rate_bits": dict(self.data_rate_bits),
            "workload_rate_cycles": dict(self.workload_rate_cycles),
            "Q_e": dict(self.access_queue_bits),
            "Z_e_p": dict(self.virtual_queue_bits),
            "Y_m_p": dict(self.bs_queue_cycles),
            "Y_i_p_loc": dict(self.local_queue_cycles),
            "bs_energy_budget": dict(self.bs_energy_budget),
            "bs_energy_pressure": dict(self.bs_energy_pressure),
        }
