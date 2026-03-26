"""队列更新模块。"""

from __future__ import annotations

from dataclasses import dataclass

from config import MECConfig, BsTaskKey, LinkTaskKey, LocalTaskKey
from env.system_state import SystemState


@dataclass
class QueueUpdater:
    """按建模公式更新各类队列。"""

    config: MECConfig

    def update_queues(
        self,
        state: SystemState,
        total_access_arrivals: dict[str, float],
        type_access_arrivals: dict[LinkTaskKey, float],
        total_access_service: dict[str, float],
        communication_shares: dict[LinkTaskKey, float],
        remote_arrivals: dict[BsTaskKey, float],
        bs_service: dict[BsTaskKey, float],
        local_arrivals: dict[LocalTaskKey, float],
        local_service: dict[LocalTaskKey, float],
    ) -> None:
        """用一个时隙的到达和服务结果更新队列。"""
        for link_id in self.config.access_links:
            # 对应 Q_e(k+1) = max(Q_e(k) - S_e(k), 0) + A_e(k)
            state.access_queue_bits[link_id] = max(
                state.access_queue_bits[link_id] - total_access_service[link_id],
                0.0,
            ) + total_access_arrivals[link_id]

        for link_id in self.config.access_links:
            for task_name in self.config.task_names:
                key = (link_id, task_name)
                # 对应 Z_e,p(k+1) = max(Z_e,p(k) - alpha_e,p(k) S_e(k), 0) + A_e,p(k)
                state.virtual_queue_bits[key] = max(
                    state.virtual_queue_bits[key]
                    - communication_shares[key] * total_access_service[link_id],
                    0.0,
                ) + type_access_arrivals[key]

        for bs_id in self.config.base_station_ids:
            for task_name in self.config.task_names:
                key = (bs_id, task_name)
                # 对应 Y_m,p(k+1) = max(Y_m,p(k) - S_m,p(k), 0) + A_m,p^rem(k)
                state.bs_queue_cycles[key] = max(
                    state.bs_queue_cycles[key] - bs_service[key],
                    0.0,
                ) + remote_arrivals[key]

        for sensor_id in self.config.sensor_ids:
            for task_name in self.config.task_names:
                key = (sensor_id, task_name)
                # 对应 Y_i,p^loc(k+1) = max(Y_i,p^loc(k) - S_i,p^loc(k), 0) + A_i,p^loc(k)
                state.local_queue_cycles[key] = max(
                    state.local_queue_cycles[key] - local_service[key],
                    0.0,
                ) + local_arrivals[key]
