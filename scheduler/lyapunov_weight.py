"""Lyapunov 权重计算器。"""

from __future__ import annotations

from dataclasses import dataclass

from config import MECConfig, BsUnitTaskKey, LinkTaskKey
from env.system_state import SystemState


@dataclass
class LyapunovDecision:
    """保存一次 Lyapunov 决策的 alpha 和 beta 。"""

    communication_shares: dict[LinkTaskKey, float]
    compute_shares: dict[BsUnitTaskKey, float]


@dataclass
class LyapunovScheduler:
    """根据队列压力计算通信与计算份额。"""

    config: MECConfig

    def _task_weight(self, task_name: str) -> float:
        """计算任务权重 w_p。"""
        task = self.config.task_by_name(task_name)
        return 1.0 + self.config.system.alpha_s * task.sensitivity

    def calculate_communication_shares(self, state: SystemState) -> dict[LinkTaskKey, float]:
        """根据虚拟队列计算 alpha_(e,p)。"""
        shares: dict[LinkTaskKey, float] = {}
        task_count = len(self.config.task_names)

        for link_id in self.config.access_links:
            weighted_pressure = {
                task_name: self._task_weight(task_name) * state.virtual_queue_bits[(link_id, task_name)]
                for task_name in self.config.task_names
            }
            total_pressure = sum(weighted_pressure.values())

            for task_name in self.config.task_names:
                # 对应 Gamma_e,p(k) = w_p Z_e,p(k), alpha_e,p(k) = Gamma_e,p / sum Gamma
                shares[(link_id, task_name)] = (
                    weighted_pressure[task_name] / total_pressure
                    if total_pressure > self.config.system.epsilon
                    else 1.0 / task_count
                )
        return shares

    def calculate_compute_shares(
        self,
        state: SystemState,
        effective_rates: dict[BsUnitTaskKey, float],
    ) -> dict[BsUnitTaskKey, float]:
        """根据计算队列计算 beta_(m,u,p)。"""
        shares: dict[BsUnitTaskKey, float] = {}
        task_count = len(self.config.task_names)

        for bs in self.config.base_stations:
            for unit in bs.compute_units:
                weighted_priority = {
                    task_name: (
                        self._task_weight(task_name)
                        * state.bs_queue_cycles[(bs.bs_id, task_name)]
                        * effective_rates[(bs.bs_id, unit.unit_name, task_name)]
                    )
                    for task_name in self.config.task_names
                }
                total_priority = sum(weighted_priority.values())

                for task_name in self.config.task_names:
                    # 对应 Phi_m,u,p(k) = Omega_m,p(k) R_m,u,p^comp
                    shares[(bs.bs_id, unit.unit_name, task_name)] = (
                        weighted_priority[task_name] / total_priority
                        if total_priority > self.config.system.epsilon
                        else 1.0 / task_count
                    )
        return shares

    def calculate_lyapunov_weights(
        self,
        state: SystemState,
        effective_rates: dict[BsUnitTaskKey, float],
    ) -> LyapunovDecision:
        """同时返回 alpha 和 beta。"""
        return LyapunovDecision(
            communication_shares=self.calculate_communication_shares(state),
            compute_shares=self.calculate_compute_shares(state, effective_rates),
        )
