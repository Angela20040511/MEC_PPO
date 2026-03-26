"""计算执行模型。"""

from __future__ import annotations

from dataclasses import dataclass

from config import MECConfig, BsTaskKey, BsUnitTaskKey, LocalTaskKey, RoutingKey, TaskKey
from env.system_state import SystemState


@dataclass
class ComputeExecutionResult:
    """保存一个时隙的计算服务、时延和能耗结果。"""

    local_service: dict[LocalTaskKey, float]
    bs_service_by_unit: dict[BsUnitTaskKey, float]
    bs_service: dict[BsTaskKey, float]
    local_delay: dict[LocalTaskKey, float]
    bs_delay: dict[BsTaskKey, float]
    local_energy_by_task: dict[LocalTaskKey, float]
    bs_energy_by_unit: dict[BsUnitTaskKey, float]
    bs_energy_by_station: dict[str, float]
    bs_energy_by_task: dict[str, float]
    total_local_energy: float
    total_bs_energy: float


@dataclass
class ComputeModel:
    """处理本地计算和基站计算的服务分配。"""

    config: MECConfig

    def effective_compute_rates(self) -> dict[BsUnitTaskKey, float]:
        """返回各计算单元对各任务的有效算力。"""
        effective_rates: dict[BsUnitTaskKey, float] = {}
        for bs in self.config.base_stations:
            for unit in bs.compute_units:
                for task_name in self.config.task_names:
                    effective_rates[(bs.bs_id, unit.unit_name, task_name)] = (
                        unit.base_rate * unit.acceleration[task_name]
                    )
        return effective_rates

    def compute_local_arrivals(
        self,
        theta: dict[TaskKey, float],
        workload_rate_cycles: dict[TaskKey, float],
    ) -> dict[LocalTaskKey, float]:
        """计算本地计算队列到达量。"""
        slot_length = self.config.system.slot_length
        return {
            key: (1.0 - theta[key]) * workload_rate_cycles[key] * slot_length
            for key in workload_rate_cycles
        }

    def compute_remote_arrivals(
        self,
        routing: dict[RoutingKey, float],
        workload_rate_cycles: dict[TaskKey, float],
    ) -> dict[BsTaskKey, float]:
        """统计进入各基站的远程计算到达量。"""
        slot_length = self.config.system.slot_length
        remote_arrivals = {
            (bs_id, task_name): 0.0
            for bs_id in self.config.base_station_ids
            for task_name in self.config.task_names
        }
        for (sensor_id, task_name), workload_rate in workload_rate_cycles.items():
            for bs_id in self.config.reachable_base_stations(sensor_id):
                remote_arrivals[(bs_id, task_name)] += (
                    routing[(sensor_id, task_name, bs_id)] * workload_rate * slot_length
                )
        return remote_arrivals

    def allocate_local_service(self, state: SystemState) -> dict[LocalTaskKey, float]:
        """按本地优先级分配 CPU 服务量。"""
        services: dict[LocalTaskKey, float] = {
            (sensor_id, task_name): 0.0
            for sensor_id in self.config.sensor_ids
            for task_name in self.config.task_names
        }

        for sensor in self.config.sensors:
            remaining_capacity = sensor.local_cpu_rate * self.config.system.slot_length
            for task_name in self.config.local_priority:
                queue_value = state.local_queue_cycles[(sensor.sensor_id, task_name)]
                service = min(queue_value, remaining_capacity)
                services[(sensor.sensor_id, task_name)] = service
                remaining_capacity -= service
        return services

    def allocate_bs_service(
        self,
        state: SystemState,
        compute_shares: dict[BsUnitTaskKey, float],
    ) -> tuple[dict[BsUnitTaskKey, float], dict[BsTaskKey, float]]:
        """根据 beta 为基站计算单元分配服务量。"""
        effective_rates = self.effective_compute_rates()
        raw_service: dict[BsUnitTaskKey, float] = {}
        scaled_service: dict[BsUnitTaskKey, float] = {}
        aggregated: dict[BsTaskKey, float] = {
            (bs_id, task_name): 0.0
            for bs_id in self.config.base_station_ids
            for task_name in self.config.task_names
        }

        for bs in self.config.base_stations:
            for unit in bs.compute_units:
                for task_name in self.config.task_names:
                    key = (bs.bs_id, unit.unit_name, task_name)
                    # 对应 S_m,u,p^comp(k) = beta_m,u,p(k) * R_m,u,p^comp * Delta t
                    raw_service[key] = (
                        compute_shares[key]
                        * effective_rates[key]
                        * self.config.system.slot_length
                    )

            for task_name in self.config.task_names:
                potential = sum(
                    raw_service[(bs.bs_id, unit.unit_name, task_name)]
                    for unit in bs.compute_units
                )
                queue_value = state.bs_queue_cycles[(bs.bs_id, task_name)]
                scale = min(1.0, queue_value / potential) if potential > 0.0 else 0.0
                aggregated[(bs.bs_id, task_name)] = potential * scale
                for unit in bs.compute_units:
                    key = (bs.bs_id, unit.unit_name, task_name)
                    scaled_service[key] = raw_service[key] * scale

        return scaled_service, aggregated

    def _compute_local_delay(
        self,
        state: SystemState,
        local_arrivals: dict[LocalTaskKey, float],
        sensor_id: str,
        task_name: str,
    ) -> float:
        """计算本地执行时延近似值。"""
        sensor = self.config.sensor_by_id(sensor_id)
        denom = max(sensor.local_cpu_rate, self.config.system.epsilon)
        key = (sensor_id, task_name)
        # 对应 T_i,p^loc(k) = Y_i,p^loc(k) / R_i^comp + A_i,p^loc(k) / R_i^comp
        return state.local_queue_cycles[key] / denom + local_arrivals[key] / denom

    def _compute_bs_delay(
            self,
            state: SystemState,
            remote_arrivals: dict[BsTaskKey, float],
            compute_shares: dict[BsUnitTaskKey, float],
            effective_rates: dict[BsUnitTaskKey, float],
            bs_id: str,
            task_name: str,
    ) -> float:
        """计算基站执行时延近似值。"""
        system = self.config.system
        epsilon = system.epsilon
        min_service_ratio = system.min_service_ratio
        delay_cap = system.delay_cap

        bs = self.config.base_station_by_id(bs_id)

        weighted_effective_rate = sum(
            compute_shares[(bs_id, unit.unit_name, task_name)]
            * effective_rates[(bs_id, unit.unit_name, task_name)]
            for unit in bs.compute_units
        )

        theoretical_total_rate = sum(
            effective_rates[(bs_id, unit.unit_name, task_name)]
            for unit in bs.compute_units
        )

        min_rate_floor = max(
            epsilon,
            min_service_ratio * theoretical_total_rate,
        )

        denom = max(weighted_effective_rate, min_rate_floor)

        queue_delay = state.bs_queue_cycles[(bs_id, task_name)] / denom
        service_delay = remote_arrivals[(bs_id, task_name)] / denom
        total_delay = queue_delay + service_delay

        return min(total_delay, delay_cap)

    def execute(
        self,
        state: SystemState,
        compute_shares: dict[BsUnitTaskKey, float],
        local_arrivals: dict[LocalTaskKey, float],
        remote_arrivals: dict[BsTaskKey, float],
    ) -> ComputeExecutionResult:
        """执行一个时隙的计算服务。"""
        effective_rates = self.effective_compute_rates()
        local_service = self.allocate_local_service(state)
        bs_service_by_unit, bs_service = self.allocate_bs_service(state, compute_shares)

        local_delay: dict[LocalTaskKey, float] = {}
        bs_delay: dict[BsTaskKey, float] = {}
        local_energy_by_task: dict[LocalTaskKey, float] = {}
        bs_energy_by_unit: dict[BsUnitTaskKey, float] = {}
        bs_energy_by_station = {bs_id: 0.0 for bs_id in self.config.base_station_ids}
        bs_energy_by_task = {task_name: 0.0 for task_name in self.config.task_names}

        for sensor in self.config.sensors:
            for task_name in self.config.task_names:
                key = (sensor.sensor_id, task_name)
                local_delay[key] = self._compute_local_delay(state, local_arrivals, sensor.sensor_id, task_name)
                # 对应 E_i,p^loc(k) = kappa_i^loc * S_i,p^loc(k)
                local_energy_by_task[key] = sensor.local_energy_coeff * local_service[key]

        for bs in self.config.base_stations:
            for unit in bs.compute_units:
                for task_name in self.config.task_names:
                    unit_key = (bs.bs_id, unit.unit_name, task_name)
                    # 对应 E_m,u,p^comp(k) = kappa_m,u^comp * S_m,u,p^comp(k)
                    energy = unit.energy_coeff * bs_service_by_unit[unit_key]
                    bs_energy_by_unit[unit_key] = energy
                    bs_energy_by_station[bs.bs_id] += energy
                    bs_energy_by_task[task_name] += energy

            for task_name in self.config.task_names:
                bs_delay[(bs.bs_id, task_name)] = self._compute_bs_delay(
                    state,
                    remote_arrivals,
                    compute_shares,
                    effective_rates,
                    bs.bs_id,
                    task_name,
                )

        return ComputeExecutionResult(
            local_service=local_service,
            bs_service_by_unit=bs_service_by_unit,
            bs_service=bs_service,
            local_delay=local_delay,
            bs_delay=bs_delay,
            local_energy_by_task=local_energy_by_task,
            bs_energy_by_unit=bs_energy_by_unit,
            bs_energy_by_station=bs_energy_by_station,
            bs_energy_by_task=bs_energy_by_task,
            total_local_energy=float(sum(local_energy_by_task.values())),
            total_bs_energy=float(sum(bs_energy_by_station.values())),
        )
