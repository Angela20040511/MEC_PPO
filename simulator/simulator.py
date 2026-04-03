"""MEC 仿真器主体。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from config import MECConfig, BsTaskKey, LinkTaskKey, RoutingKey, TaskKey
from dt.digital_twin import DigitalTwin
from env.channel_model import ChannelModel
from env.compute_model import ComputeExecutionResult, ComputeModel
from env.queue_update import QueueUpdater
from env.system_state import SystemState
from scheduler.fast_scheduler import FastScheduler, FastSchedulingResult
from scheduler.lyapunov_weight import LyapunovDecision, LyapunovScheduler


@dataclass
class OffloadingDecision:
    """PPO 动作解码后的卸载决策。"""

    theta: dict[TaskKey, float]
    routing: dict[RoutingKey, float]


@dataclass
class DelayBreakdown:
    """时延分项统计。"""

    total_delay_by_task: dict[TaskKey, float]
    local_delay_cost: float
    uplink_delay_cost: float
    backhaul_delay_cost: float
    bs_compute_delay_cost: float
    total_delay: float


@dataclass
class EnergyBreakdown:
    """能耗分项统计。"""

    uplink_energy: float
    uplink_energy_by_task: dict[str, float]
    local_compute_energy: float
    local_compute_energy_by_task: dict[str, float]
    bs_compute_energy: float
    bs_compute_energy_by_task: dict[str, float]
    total_energy: float
    total_energy_by_task: dict[str, float]


@dataclass
class Simulator:
    """把 PPO、Lyapunov 和 MEC 模型串联起来的仿真器。"""

    config: MECConfig
    channel_model: ChannelModel = field(init=False)
    compute_model: ComputeModel = field(init=False)
    queue_updater: QueueUpdater = field(init=False)
    state: SystemState = field(init=False)
    digital_twin: DigitalTwin = field(init=False)
    lyapunov_scheduler: LyapunovScheduler = field(init=False)
    fast_scheduler: FastScheduler = field(init=False)
    enable_digital_twin_fit: bool = True

    def __post_init__(self) -> None:
        """初始化仿真器依赖的所有子模块。"""
        self.channel_model = ChannelModel(self.config)
        self.compute_model = ComputeModel(self.config)
        self.queue_updater = QueueUpdater(self.config)
        self.state = SystemState(self.config)
        self.digital_twin = DigitalTwin(self.config)
        self.lyapunov_scheduler = LyapunovScheduler(self.config)
        self.fast_scheduler = FastScheduler(self.config, self.channel_model)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """重置环境并返回初始观测。"""
        self.state.reset(seed)
        return self.get_observation()

    def get_observation(self) -> np.ndarray:
        """组装当前时隙的 PPO 观测向量。"""
        predicted_loads = self.digital_twin.predict(self.state)
        return self.state.build_state_vector(
            predicted_loads=predicted_loads,
            access_gains=self.channel_model.access_gains,
            effective_compute_rates=self.compute_model.effective_compute_rates(),
        )

    def _sigmoid(self, value: float) -> float:
        """把实数映射成 0 到 1 的卸载比例。"""
        return float(1.0 / (1.0 + np.exp(-value)))

    def _softmax(self, values: np.ndarray) -> np.ndarray:
        """把路由 logit 映射成概率分布。"""
        shifted = values - np.max(values)
        exp_values = np.exp(shifted)
        return exp_values / np.maximum(exp_values.sum(), self.config.system.epsilon)

    def decode_action(self, raw_action: np.ndarray) -> OffloadingDecision:
        """把 PPO 动作向量解码为 theta 和 routing。"""
        action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
        if action.size != self.config.action_dim:
            raise ValueError(f"Expected action dim {self.config.action_dim}, got {action.size}")

        theta: dict[TaskKey, float] = {}
        routing: dict[RoutingKey, float] = {
            (sensor_id, task_name, bs_id): 0.0
            for sensor_id in self.config.sensor_ids
            for task_name in self.config.task_names
            for bs_id in self.config.base_station_ids
        }

        cursor = 0
        for sensor_id in self.config.sensor_ids:
            reachable = self.config.reachable_base_stations(sensor_id)
            reachable_indices = [self.config.base_station_ids.index(bs_id) for bs_id in reachable]
            for task_name in self.config.task_names:
                raw_theta = float(action[cursor])
                cursor += 1
                raw_logits = action[cursor : cursor + len(self.config.base_station_ids)]
                cursor += len(self.config.base_station_ids)

                theta_value = self._sigmoid(raw_theta)
                theta[(sensor_id, task_name)] = theta_value

                reachable_logits = raw_logits[reachable_indices]
                reachable_probs = self._softmax(reachable_logits)
                for offset, bs_id in enumerate(reachable):
                    routing[(sensor_id, task_name, bs_id)] = theta_value * float(reachable_probs[offset])

        return OffloadingDecision(theta=theta, routing=routing)

    def _compute_access_arrivals(
        self,
        decision: OffloadingDecision,
    ) -> tuple[dict[LinkTaskKey, float], dict[str, float]]:
        """统计各接入链路的到达比特量。"""
        type_arrivals: dict[LinkTaskKey, float] = {}
        total_arrivals: dict[str, float] = {link_id: 0.0 for link_id in self.config.access_links}
        slot_length = self.config.system.slot_length

        for sensor_id in self.config.sensor_ids:
            link_id = self.config.access_link_id(sensor_id)
            for task_name in self.config.task_names:
                rate = decision.theta[(sensor_id, task_name)] * self.state.data_rate_bits[(sensor_id, task_name)]
                arrival_bits = rate * slot_length
                type_arrivals[(link_id, task_name)] = arrival_bits
                total_arrivals[link_id] += arrival_bits
        return type_arrivals, total_arrivals

    def _compute_backhaul_arrivals(
        self,
        decision: OffloadingDecision,
    ) -> dict[tuple[str, str, str], float]:
        """统计回传链路的到达比特量。"""
        arrivals = {
            (source_bs, target_bs, task_name): 0.0
            for source_bs, neighbors in self.config.neighbors.items()
            for target_bs in neighbors
            for task_name in self.config.task_names
        }
        slot_length = self.config.system.slot_length

        for sensor_id in self.config.sensor_ids:
            source_bs = self.config.sensor_by_id(sensor_id).home_bs
            for task_name in self.config.task_names:
                for target_bs in self.config.reachable_base_stations(sensor_id):
                    if target_bs == source_bs:
                        continue
                    arrivals[(source_bs, target_bs, task_name)] += (
                        decision.routing[(sensor_id, task_name, target_bs)]
                        * self.state.data_rate_bits[(sensor_id, task_name)]
                        * slot_length
                    )
        return arrivals

    def _compute_access_delays(
            self,
            link_rates: dict[str, float],
            type_access_arrivals: dict[LinkTaskKey, float],
    ) -> dict[LinkTaskKey, float]:
        """计算上行接入的分任务时延。"""
        system = self.config.system
        epsilon = system.epsilon
        delay_cap = system.delay_cap
        min_access_rate_bps = system.min_access_rate_bps

        delays: dict[LinkTaskKey, float] = {}

        for link_id in self.config.access_links:
            raw_rate = link_rates[link_id]
            denom = max(raw_rate, min_access_rate_bps, epsilon)

            queue_delay = self.state.access_queue_bits[link_id] / denom

            for task_name in self.config.task_names:
                tx_delay = type_access_arrivals[(link_id, task_name)] / denom
                total_delay = queue_delay + tx_delay
                delays[(link_id, task_name)] = min(total_delay, delay_cap)

        return delays

    def _compute_backhaul_delays(
            self,
            backhaul_arrivals: dict[tuple[str, str, str], float],
            backhaul_rates: dict[str, float],
    ) -> dict[tuple[str, str, str], float]:
        """计算回传链路的分任务时延。"""
        system = self.config.system
        epsilon = system.epsilon
        delay_cap = system.delay_cap
        min_backhaul_rate_bps = system.min_backhaul_rate_bps

        delays: dict[tuple[str, str, str], float] = {}

        for (source_bs, target_bs, task_name), arrival_bits in backhaul_arrivals.items():
            link_id = self.config.backhaul_link_id(source_bs, target_bs)
            raw_rate = backhaul_rates[link_id]
            denom = max(raw_rate, min_backhaul_rate_bps, epsilon)

            total_delay = arrival_bits / denom
            delays[(source_bs, target_bs, task_name)] = min(total_delay, delay_cap)

        return delays

    def _compute_delay_breakdown(
        self,
        decision: OffloadingDecision,
        access_delay: dict[LinkTaskKey, float],
        backhaul_delay: dict[tuple[str, str, str], float],
        compute_result: ComputeExecutionResult,
    ) -> DelayBreakdown:
        """统计总时延及其组成部分。"""
        total_delay_by_task: dict[TaskKey, float] = {}
        local_delay_cost = 0.0
        uplink_delay_cost = 0.0
        backhaul_delay_cost = 0.0
        bs_compute_delay_cost = 0.0
        slot_length = self.config.system.slot_length

        for sensor_id in self.config.sensor_ids:
            home_bs = self.config.sensor_by_id(sensor_id).home_bs
            link_id = self.config.access_link_id(sensor_id)
            for task_name in self.config.task_names:
                task_key = (sensor_id, task_name)
                arrival_weight = self.state.lambda_imp[task_key] * slot_length

                local_component = (1.0 - decision.theta[task_key]) * compute_result.local_delay[task_key]
                uplink_component = decision.theta[task_key] * access_delay[(link_id, task_name)]
                backhaul_component = 0.0
                bs_component = decision.routing[(sensor_id, task_name, home_bs)] * compute_result.bs_delay[(home_bs, task_name)]

                for target_bs in self.config.reachable_base_stations(sensor_id):
                    if target_bs == home_bs:
                        continue
                    route_ratio = decision.routing[(sensor_id, task_name, target_bs)]
                    backhaul_component += route_ratio * backhaul_delay[(home_bs, target_bs, task_name)]
                    bs_component += route_ratio * compute_result.bs_delay[(target_bs, task_name)]

                total_path_delay = local_component + uplink_component + backhaul_component + bs_component
                total_delay_by_task[task_key] = total_path_delay
                local_delay_cost += arrival_weight * local_component
                uplink_delay_cost += arrival_weight * uplink_component
                backhaul_delay_cost += arrival_weight * backhaul_component
                bs_compute_delay_cost += arrival_weight * bs_component

        total_delay = local_delay_cost + uplink_delay_cost + backhaul_delay_cost + bs_compute_delay_cost
        return DelayBreakdown(
            total_delay_by_task=total_delay_by_task,
            local_delay_cost=float(local_delay_cost),
            uplink_delay_cost=float(uplink_delay_cost),
            backhaul_delay_cost=float(backhaul_delay_cost),
            bs_compute_delay_cost=float(bs_compute_delay_cost),
            total_delay=float(total_delay),
        )

    def _aggregate_local_energy_by_task(
        self,
        local_energy_by_sensor_task: dict[tuple[str, str], float],
    ) -> dict[str, float]:
        """把本地能耗按任务类型汇总。"""
        aggregated = {task_name: 0.0 for task_name in self.config.task_names}
        for (_, task_name), energy in local_energy_by_sensor_task.items():
            aggregated[task_name] += energy
        return aggregated

    def _compute_uplink_energy(
        self,
        fast_result: FastSchedulingResult,
        communication_shares: dict[LinkTaskKey, float],
    ) -> tuple[float, dict[str, float], dict[LinkTaskKey, float]]:
        """计算上行传输能耗并按任务分摊。"""
        uplink_energy_by_task = {task_name: 0.0 for task_name in self.config.task_names}
        uplink_energy_by_link_task: dict[LinkTaskKey, float] = {}

        for sensor_id in self.config.sensor_ids:
            link_id = self.config.access_link_id(sensor_id)
            link_rate = fast_result.link_rates[link_id]
            tx_power = fast_result.tx_powers.get(sensor_id, 0.0)
            serviced_bits = fast_result.link_service_bits[link_id]
            total_link_energy = self.channel_model.compute_transmission_energy(
                tx_power_w=tx_power,
                serviced_bits=serviced_bits,
                rate_bps=link_rate,
            )

            alpha_sum = sum(communication_shares[(link_id, task_name)] for task_name in self.config.task_names)
            for task_name in self.config.task_names:
                alpha_share = (
                    communication_shares[(link_id, task_name)] / alpha_sum
                    if alpha_sum > self.config.system.epsilon
                    else 1.0 / len(self.config.task_names)
                )
                task_energy = total_link_energy * alpha_share
                uplink_energy_by_link_task[(link_id, task_name)] = task_energy
                uplink_energy_by_task[task_name] += task_energy

        total_uplink_energy = float(sum(uplink_energy_by_task.values()))
        return total_uplink_energy, uplink_energy_by_task, uplink_energy_by_link_task

    def compute_energy_per_task_type(
        self,
        task_name: str,
        uplink_energy_by_task: dict[str, float],
        local_energy_by_task: dict[str, float],
        bs_energy_by_task: dict[str, float],
    ) -> float:
        """统计某类任务的总能耗。"""
        return (
            uplink_energy_by_task.get(task_name, 0.0)
            + local_energy_by_task.get(task_name, 0.0)
            + bs_energy_by_task.get(task_name, 0.0)
        )

    def _compute_energy_breakdown(
        self,
        fast_result: FastSchedulingResult,
        communication_shares: dict[LinkTaskKey, float],
        compute_result: ComputeExecutionResult,
    ) -> EnergyBreakdown:
        """统计总能耗及其组成部分。"""
        uplink_energy, uplink_energy_by_task, _ = self._compute_uplink_energy(
            fast_result,
            communication_shares,
        )
        local_energy_by_task = self._aggregate_local_energy_by_task(compute_result.local_energy_by_task)
        total_energy_by_task = {
            task_name: self.compute_energy_per_task_type(
                task_name,
                uplink_energy_by_task,
                local_energy_by_task,
                compute_result.bs_energy_by_task,
            )
            for task_name in self.config.task_names
        }
        total_energy = uplink_energy + compute_result.total_local_energy + compute_result.total_bs_energy
        return EnergyBreakdown(
            uplink_energy=float(uplink_energy),
            uplink_energy_by_task=uplink_energy_by_task,
            local_compute_energy=float(compute_result.total_local_energy),
            local_compute_energy_by_task=local_energy_by_task,
            bs_compute_energy=float(compute_result.total_bs_energy),
            bs_compute_energy_by_task=compute_result.bs_energy_by_task,
            total_energy=float(total_energy),
            total_energy_by_task=total_energy_by_task,
        )

    def _reward_scale_denominator(self) -> float:
        reward_mode = self.config.system.reward_mode
        if reward_mode == "raw_total":
            return 1.0
        if reward_mode == "per_sensor":
            return float(max(len(self.config.sensors), 1))
        raise ValueError(f"Unsupported reward_mode: {reward_mode}")

    def _compute_cost(self, total_energy: float, total_delay: float) -> tuple[float, float, float]:
        """计算能耗和时延组成的代价，并返回分项。"""
        scale_denominator = self._reward_scale_denominator()
        delay_term = self.config.system.omega_t * total_delay / scale_denominator
        energy_term = self.config.system.omega_e * total_energy / scale_denominator
        cost = delay_term + energy_term
        return cost, delay_term, energy_term

    def _compute_queue_penalty(self, normalized_avg_backlog: float) -> float:
        """计算归一化 backlog 额外惩罚。"""
        return self.config.system.omega_q * normalized_avg_backlog

    def _compute_reward(self, cost: float, queue_penalty: float) -> float:
        """把代价转换成 PPO 奖励。"""
        return -cost - queue_penalty

    def _summarize_alpha(self, alpha: dict[LinkTaskKey, float]) -> dict[str, float]:
        """按任务类型汇总平均 alpha。"""
        link_count = max(len(self.config.access_links), 1)
        return {
            task_name: float(
                sum(alpha[(link_id, task_name)] for link_id in self.config.access_links) / link_count
            )
            for task_name in self.config.task_names
        }

    def _summarize_beta(self, beta: dict[tuple[str, str, str], float]) -> dict[str, float]:
        """按任务类型汇总平均 beta。"""
        slot_count = max(
            sum(len(bs.compute_units) for bs in self.config.base_stations),
            1,
        )
        return {
            task_name: float(
                sum(
                    beta[(bs.bs_id, unit.unit_name, task_name)]
                    for bs in self.config.base_stations
                    for unit in bs.compute_units
                ) / slot_count
            )
            for task_name in self.config.task_names
        }

    def _average_theta(self, theta: dict[TaskKey, float]) -> float:
        """计算平均卸载率。"""
        return float(np.mean(list(theta.values()))) if theta else 0.0

    def _routing_distribution(self, routing: dict[RoutingKey, float]) -> dict[str, float]:
        """统计平均路由分布。"""
        pair_count = max(len(self.config.sensor_ids) * len(self.config.task_names), 1)
        return {
            bs_id: float(
                sum(
                    routing[(sensor_id, task_name, bs_id)]
                    for sensor_id in self.config.sensor_ids
                    for task_name in self.config.task_names
                ) / pair_count
            )
            for bs_id in self.config.base_station_ids
        }

    def _summarize_clusters(self, fast_result: FastSchedulingResult) -> dict[str, Any]:
        """Summarize cluster composition for scheduler diagnostics."""
        plans = [
            plan
            for cluster_plans in fast_result.clusters.values()
            for plan in cluster_plans
        ]
        cluster_sizes = [len(plan.members) for plan in plans]
        oma_cluster_count = sum(1 for plan in plans if plan.mode == "OMA")
        noma_cluster_count = sum(1 for plan in plans if plan.mode == "NOMA")
        total_cluster_count = len(plans)
        users_in_noma_clusters = sum(len(plan.members) for plan in plans if plan.mode == "NOMA")
        total_users_in_clusters = sum(cluster_sizes)
        max_cluster_size_observed = max(cluster_sizes, default=0)
        avg_cluster_size = (
            total_users_in_clusters / max(total_cluster_count, 1)
            if total_cluster_count > 0
            else 0.0
        )

        cluster_size_histogram: dict[str, int] = {}
        for size in cluster_sizes:
            key = str(size)
            cluster_size_histogram[key] = cluster_size_histogram.get(key, 0) + 1

        return {
            "oma_cluster_count": oma_cluster_count,
            "noma_cluster_count": noma_cluster_count,
            "multi_user_cluster_count": noma_cluster_count,
            "total_cluster_count": total_cluster_count,
            "users_in_noma_clusters": users_in_noma_clusters,
            "total_users_in_clusters": total_users_in_clusters,
            "avg_cluster_size": float(avg_cluster_size),
            "max_cluster_size_observed": max_cluster_size_observed,
            "cluster_size_histogram": cluster_size_histogram,
        }

    def log_metrics(self, info: dict[str, Any]) -> dict[str, Any]:
        """从 step 返回信息中提取更紧凑的日志指标。"""
        return {
            "total_delay": float(info["total_delay"]),
            "total_energy": float(info["total_energy"]),
            "raw_total_delay": float(info["raw_total_delay"]),
            "raw_total_energy": float(info["raw_total_energy"]),
            "cost": float(info["cost"]),
            "queue_penalty": float(info["queue_penalty"]),
            "ppo_reward": float(info["reward"]),
            "uplink_energy": float(info["uplink_energy"]),
            "local_compute_energy": float(info["local_compute_energy"]),
            "bs_compute_energy": float(info["bs_compute_energy"]),
            "uplink_delay_cost": float(info["uplink_delay_cost"]),
            "backhaul_delay_cost": float(info["backhaul_delay_cost"]),
            "local_delay_cost": float(info["local_delay_cost"]),
            "bs_compute_delay_cost": float(info["bs_compute_delay_cost"]),
            "normalized_avg_queue_len": float(info["avg_queue_len"]),
            "avg_theta": float(info["avg_theta"]),
            "alpha_summary": info["alpha_summary"],
            "beta_summary": info["beta_summary"],
            "rho_m_noma": info["rho_m_noma"],
            "routing_distribution": info["routing_distribution"],
            "uplink_energy_by_task": info["uplink_energy_by_task"],
            "local_compute_energy_by_task": info["local_compute_energy_by_task"],
            "bs_compute_energy_by_task": info["bs_compute_energy_by_task"],
            "total_energy_by_task": info["total_energy_by_task"],
            "bs_energy_budget": info["bs_energy_budget"],
            "bs_energy_pressure": info["bs_energy_pressure"],
            "raw_total_backlog": float(info["raw_total_backlog"]),
            "normalized_total_backlog": float(info["normalized_total_backlog"]),
            "num_sensors": int(info["num_sensors"]),
            "reward_mode": info["reward_mode"],
            "reward_delay_term": float(info["reward_delay_term"]),
            "reward_energy_term": float(info["reward_energy_term"]),
            "reward_backlog_term": float(info["reward_backlog_term"]),
            "raw_reward_delay_term": float(info["raw_reward_delay_term"]),
            "raw_reward_energy_term": float(info["raw_reward_energy_term"]),
            "raw_reward_backlog_term": float(info["raw_reward_backlog_term"]),
            "oma_cluster_count": int(info["oma_cluster_count"]),
            "noma_cluster_count": int(info["noma_cluster_count"]),
            "multi_user_cluster_count": int(info["multi_user_cluster_count"]),
            "total_cluster_count": int(info["total_cluster_count"]),
            "users_in_noma_clusters": int(info["users_in_noma_clusters"]),
            "avg_cluster_size": float(info["avg_cluster_size"]),
            "max_cluster_size_observed": int(info["max_cluster_size_observed"]),
            "cluster_size_histogram": info["cluster_size_histogram"],
        }

    def step(self, raw_action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """推进环境一个时隙。"""
        decision = self.decode_action(raw_action)
        local_arrivals = self.compute_model.compute_local_arrivals(decision.theta, self.state.workload_rate_cycles)
        remote_arrivals = self.compute_model.compute_remote_arrivals(decision.routing, self.state.workload_rate_cycles)
        type_access_arrivals, total_access_arrivals = self._compute_access_arrivals(decision)
        backhaul_arrivals = self._compute_backhaul_arrivals(decision)

        effective_rates = self.compute_model.effective_compute_rates()
        lyapunov_decision: LyapunovDecision = self.lyapunov_scheduler.calculate_lyapunov_weights(
            self.state,
            effective_rates,
        )
        fast_result = self.fast_scheduler.schedule_access(
            state=self.state,
            alpha=lyapunov_decision.communication_shares,
        )
        compute_result = self.compute_model.execute(
            self.state,
            lyapunov_decision.compute_shares,
            local_arrivals,
            remote_arrivals,
        )

        access_delay = self._compute_access_delays(fast_result.link_rates, type_access_arrivals)
        backhaul_delay = self._compute_backhaul_delays(backhaul_arrivals, fast_result.backhaul_rates)
        delay_breakdown = self._compute_delay_breakdown(
            decision,
            access_delay,
            backhaul_delay,
            compute_result,
        )
        energy_breakdown = self._compute_energy_breakdown(
            fast_result,
            lyapunov_decision.communication_shares,
            compute_result,
        )

        cost, delay_term, energy_term = self._compute_cost(
            energy_breakdown.total_energy,
            delay_breakdown.total_delay,
        )

        alpha_summary = self._summarize_alpha(lyapunov_decision.communication_shares)
        beta_summary = self._summarize_beta(lyapunov_decision.compute_shares)
        avg_theta = self._average_theta(decision.theta)
        routing_distribution = self._routing_distribution(decision.routing)
        cluster_summary = self._summarize_clusters(fast_result)

        self.queue_updater.update_queues(
            state=self.state,
            total_access_arrivals=total_access_arrivals,
            type_access_arrivals=type_access_arrivals,
            total_access_service=fast_result.link_service_bits,
            communication_shares=lyapunov_decision.communication_shares,
            remote_arrivals=remote_arrivals,
            bs_service=compute_result.bs_service,
            local_arrivals=local_arrivals,
            local_service=compute_result.local_service,
        )
        self.state.update_energy_state(compute_result.bs_energy_by_station)

        raw_total_backlog = self.state.queue_backlog_sum()
        normalized_total_backlog = self.state.normalized_queue_backlog_sum()
        normalized_avg_backlog = self.state.normalized_average_queue_len()

        queue_penalty = self._compute_queue_penalty(normalized_avg_backlog)
        reward = self._compute_reward(cost, queue_penalty)

        done = self.state.slot + 1 >= self.config.training.time_steps
        self.state.advance_slot()
        if self.enable_digital_twin_fit:
            self.digital_twin.maybe_fit(self.state)
        next_state = self.get_observation()

        info: dict[str, Any] = {
            "slot": self.state.slot,
            "theta": decision.theta,
            "routing": decision.routing,
            "link_rates": fast_result.link_rates,
            "communication_shares": lyapunov_decision.communication_shares,
            "compute_shares": lyapunov_decision.compute_shares,
            "total_delay_by_task": delay_breakdown.total_delay_by_task,
            "raw_total_delay": delay_breakdown.total_delay,
            "total_delay": delay_breakdown.total_delay,
            "cost": cost,
            "queue_penalty": queue_penalty,
            "reward": reward,
            "uplink_energy": energy_breakdown.uplink_energy,
            "local_compute_energy": energy_breakdown.local_compute_energy,
            "bs_compute_energy": energy_breakdown.bs_compute_energy,
            "raw_total_energy": energy_breakdown.total_energy,
            "total_energy": energy_breakdown.total_energy,
            "uplink_energy_by_task": energy_breakdown.uplink_energy_by_task,
            "local_compute_energy_by_task": energy_breakdown.local_compute_energy_by_task,
            "bs_compute_energy_by_task": energy_breakdown.bs_compute_energy_by_task,
            "total_energy_by_task": energy_breakdown.total_energy_by_task,
            "local_delay_cost": delay_breakdown.local_delay_cost,
            "uplink_delay_cost": delay_breakdown.uplink_delay_cost,
            "backhaul_delay_cost": delay_breakdown.backhaul_delay_cost,
            "bs_compute_delay_cost": delay_breakdown.bs_compute_delay_cost,
            "raw_total_backlog": raw_total_backlog,
            "normalized_total_backlog": normalized_total_backlog,
            "total_backlog": normalized_avg_backlog,
            "avg_queue_len": normalized_avg_backlog,
            "num_sensors": len(self.config.sensors),
            "reward_mode": self.config.system.reward_mode,
            "reward_delay_term": delay_term,
            "reward_energy_term": energy_term,
            "reward_backlog_term": queue_penalty,
            "raw_reward_delay_term": self.config.system.omega_t * delay_breakdown.total_delay,
            "raw_reward_energy_term": self.config.system.omega_e * energy_breakdown.total_energy,
            "raw_reward_backlog_term": queue_penalty,
            "alpha_summary": alpha_summary,
            "beta_summary": beta_summary,
            "rho_m_noma": fast_result.rho_m_noma,
            "avg_theta": avg_theta,
            "routing_distribution": routing_distribution,
            "bs_energy_budget": dict(self.state.bs_energy_budget),
            "bs_energy_pressure": dict(self.state.bs_energy_pressure),
            "oma_cluster_count": cluster_summary["oma_cluster_count"],
            "noma_cluster_count": cluster_summary["noma_cluster_count"],
            "multi_user_cluster_count": cluster_summary["multi_user_cluster_count"],
            "total_cluster_count": cluster_summary["total_cluster_count"],
            "users_in_noma_clusters": cluster_summary["users_in_noma_clusters"],
            "total_users_in_clusters": cluster_summary["total_users_in_clusters"],
            "avg_cluster_size": cluster_summary["avg_cluster_size"],
            "max_cluster_size_observed": cluster_summary["max_cluster_size_observed"],
            "cluster_size_histogram": cluster_summary["cluster_size_histogram"],
        }
        info["metrics"] = self.log_metrics(info)
        return next_state, float(reward), done, info
