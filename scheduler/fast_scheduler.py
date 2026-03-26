"""Fast scheduler for access-side user clustering and resource allocation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from config import MECConfig, LinkTaskKey
from env.channel_model import ChannelModel
from env.system_state import SystemState


@dataclass
class ClusterPlan:
    """Resource allocation for one OMA or NOMA cluster."""

    base_station_id: str
    members: tuple[str, ...]
    mode: Literal["OMA", "NOMA"]
    bandwidth_hz: float
    tx_powers: dict[str, float]
    decoding_order: tuple[str, ...]


@dataclass
class FastSchedulingResult:
    """Collected outputs of one fast-scheduling step."""

    clusters: dict[str, list[ClusterPlan]]
    user_pressure: dict[str, float]
    link_rates: dict[str, float]
    link_service_bits: dict[str, float]
    tx_powers: dict[str, float]
    backhaul_rates: dict[str, float]
    rho_m_noma: dict[str, float]
    cluster_debug: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class FastScheduler:
    """Cluster users and allocate access resources from Lyapunov pressure."""

    config: MECConfig
    channel_model: ChannelModel

    def _covered_users(self, bs_id: str) -> list[str]:
        return [sensor.sensor_id for sensor in self.config.sensors if sensor.home_bs == bs_id]

    def _task_weight(self, task_name: str) -> float:
        task = self.config.task_by_name(task_name)
        return 1.0 + self.config.system.alpha_s * task.sensitivity

    def build_user_pressure(
        self,
        state: SystemState,
        alpha: dict[LinkTaskKey, float],
    ) -> dict[str, float]:
        pressure: dict[str, float] = {}
        for sensor_id in self.config.sensor_ids:
            link_id = self.config.access_link_id(sensor_id)
            pressure[sensor_id] = sum(
                self._task_weight(task_name)
                * alpha[(link_id, task_name)]
                * state.virtual_queue_bits[(link_id, task_name)]
                for task_name in self.config.task_names
            )
        return pressure

    def _cluster_users_with_debug(
        self,
        pressure: dict[str, float],
    ) -> tuple[dict[str, list[tuple[str, ...]]], dict[str, dict[str, Any]]]:
        clusters: dict[str, list[tuple[str, ...]]] = {}
        cluster_debug: dict[str, dict[str, Any]] = {}

        for bs_id in self.config.base_station_ids:
            covered = self._covered_users(bs_id)
            if not covered:
                clusters[bs_id] = []
                cluster_debug[bs_id] = {
                    "bs_id": bs_id,
                    "candidate_users": [],
                    "candidate_user_count": 0,
                    "user_pressures": {},
                    "sorted_candidates": [],
                    "quantile_threshold": 0.0,
                    "oma_users": [],
                    "noma_users": [],
                    "noma_clusters": [],
                    "cluster_sizes": [],
                    "oma_cluster_count": 0,
                    "noma_cluster_count": 0,
                    "users_in_noma_clusters": 0,
                    "max_cluster_size_seen": 0,
                }
                continue

            pressure_values = np.asarray([pressure[sensor_id] for sensor_id in covered], dtype=np.float64)
            threshold = float(np.quantile(pressure_values, self.config.system.noma_quantile))
            oma_users = [sensor_id for sensor_id in covered if pressure[sensor_id] >= threshold]
            ordered_noma_users = [sensor_id for sensor_id in covered if sensor_id not in oma_users]
            ordered_noma_users.sort(key=lambda sensor_id: pressure[sensor_id], reverse=True)

            bs_clusters = [(sensor_id,) for sensor_id in oma_users]
            noma_clusters: list[list[str]] = []
            remaining_noma_users = ordered_noma_users.copy()
            while remaining_noma_users:
                members = tuple(remaining_noma_users[: self.config.system.max_cluster_size])
                bs_clusters.append(members)
                noma_clusters.append(list(members))
                remaining_noma_users = remaining_noma_users[self.config.system.max_cluster_size :]

            sorted_candidates = sorted(covered, key=lambda sensor_id: pressure[sensor_id], reverse=True)
            cluster_sizes = [len(members) for members in bs_clusters]
            clusters[bs_id] = bs_clusters
            cluster_debug[bs_id] = {
                "bs_id": bs_id,
                "candidate_users": list(covered),
                "candidate_user_count": len(covered),
                "user_pressures": {
                    sensor_id: float(pressure[sensor_id])
                    for sensor_id in covered
                },
                "sorted_candidates": [
                    {
                        "user_id": sensor_id,
                        "pressure": float(pressure[sensor_id]),
                    }
                    for sensor_id in sorted_candidates
                ],
                "quantile_threshold": threshold,
                "oma_users": list(oma_users),
                "noma_users": ordered_noma_users,
                "noma_clusters": noma_clusters,
                "cluster_sizes": cluster_sizes,
                "oma_cluster_count": len(oma_users),
                "noma_cluster_count": len(noma_clusters),
                "users_in_noma_clusters": sum(len(members) for members in noma_clusters),
                "max_cluster_size_seen": max(cluster_sizes, default=0),
            }

        return clusters, cluster_debug

    def cluster_users(self, pressure: dict[str, float]) -> dict[str, list[tuple[str, ...]]]:
        clusters, _ = self._cluster_users_with_debug(pressure)
        return clusters

    def cluster_users_with_debug(
        self,
        pressure: dict[str, float],
    ) -> tuple[dict[str, list[tuple[str, ...]]], dict[str, dict[str, Any]]]:
        return self._cluster_users_with_debug(pressure)

    def allocate_bandwidth(
        self,
        pressure: dict[str, float],
        clusters: dict[str, list[tuple[str, ...]]],
    ) -> dict[tuple[str, int], float]:
        bandwidths: dict[tuple[str, int], float] = {}
        for bs_id, cluster_list in clusters.items():
            if not cluster_list:
                continue
            base_station = self.config.base_station_by_id(bs_id)
            weights = [sum(pressure[user_id] for user_id in members) for members in cluster_list]
            total_weight = sum(weights)
            for index, members in enumerate(cluster_list):
                weight = weights[index]
                share = (
                    weight / total_weight
                    if total_weight > self.config.system.epsilon
                    else 1.0 / len(cluster_list)
                )
                bandwidths[(bs_id, index)] = share * base_station.uplink_bandwidth_hz
        return bandwidths

    def _normalized_pressure(self, bs_id: str, pressure: dict[str, float]) -> dict[str, float]:
        covered = self._covered_users(bs_id)
        max_pressure = max((pressure[sensor_id] for sensor_id in covered), default=0.0)
        return {
            sensor_id: pressure[sensor_id] / max(max_pressure, self.config.system.epsilon)
            for sensor_id in covered
        }

    def _power_budget_coefficient(self, bs_id: str, state: SystemState) -> float:
        covered_links = [
            self.config.access_link_id(sensor.sensor_id)
            for sensor in self.config.sensors
            if sensor.home_bs == bs_id
        ]
        queue_sum = sum(state.access_queue_bits[link_id] for link_id in covered_links)
        max_access_rate = self.channel_model.max_access_rate(bs_id)
        q_hat = queue_sum / max(
            max_access_rate * self.config.system.q_target_delay,
            self.config.system.epsilon,
        )
        q_tilde = min(q_hat, self.config.system.q_max)
        energy_pressure = state.bs_energy_pressure.get(
            bs_id,
            self.config.base_station_by_id(bs_id).energy_pressure_baseline,
        )
        rho = (
            self.config.system.rho0
            - self.config.system.gamma_e * energy_pressure
            - self.config.system.gamma_q * q_tilde
        )
        return float(np.clip(rho, self.config.system.rho_min, self.config.system.rho_max))

    def _max_tx_powers(self, bs_id: str, pressure: dict[str, float]) -> dict[str, float]:
        normalized_pressure = self._normalized_pressure(bs_id, pressure)
        return {
            sensor.sensor_id: sensor.tx_power_base
            * (1.0 + self.config.system.alpha_p * normalized_pressure[sensor.sensor_id])
            for sensor in self.config.sensors
            if sensor.home_bs == bs_id
        }

    def allocate_tx_powers(
        self,
        bs_id: str,
        cluster_members: tuple[str, ...],
        pressure: dict[str, float],
        rho_m_noma: float,
    ) -> dict[str, float]:
        max_powers = self._max_tx_powers(bs_id, pressure)

        if len(cluster_members) == 1:
            user_id = cluster_members[0]
            return {user_id: max_powers[user_id]}

        cluster_budget = rho_m_noma * sum(max_powers[user_id] for user_id in cluster_members)
        sqrt_pressure = {
            user_id: np.sqrt(max(pressure[user_id], 0.0))
            for user_id in cluster_members
        }
        sqrt_sum = sum(sqrt_pressure.values())

        tx_powers: dict[str, float] = {}
        for user_id in cluster_members:
            proposed_power = (
                cluster_budget * sqrt_pressure[user_id] / sqrt_sum
                if sqrt_sum > self.config.system.epsilon
                else cluster_budget / len(cluster_members)
            )
            tx_powers[user_id] = min(max_powers[user_id], proposed_power)
        return tx_powers

    def determine_sic_order(
        self,
        bs_id: str,
        cluster_members: tuple[str, ...],
        pressure: dict[str, float],
    ) -> tuple[str, ...]:
        normalized_pressure = self._normalized_pressure(bs_id, pressure)
        scored_members: list[tuple[str, float]] = []
        for user_id in cluster_members:
            link_id = self.config.access_link_id(user_id)
            score = (
                self.config.system.eta_g * self.channel_model.access_gains[link_id]
                + self.config.system.eta_u * normalized_pressure[user_id]
            )
            scored_members.append((user_id, score))
        scored_members.sort(key=lambda item: item[1], reverse=True)
        return tuple(user_id for user_id, _ in scored_members)

    def schedule_access(
        self,
        state: SystemState,
        alpha: dict[LinkTaskKey, float],
    ) -> FastSchedulingResult:
        pressure = self.build_user_pressure(state, alpha)
        clusters, cluster_debug = self.cluster_users_with_debug(pressure)
        bandwidths = self.allocate_bandwidth(pressure, clusters)
        cluster_plans: dict[str, list[ClusterPlan]] = {
            bs_id: []
            for bs_id in self.config.base_station_ids
        }
        tx_power_map: dict[str, float] = {}
        link_rates: dict[str, float] = {link_id: 0.0 for link_id in self.config.access_links}
        rho_m_noma = {
            bs_id: self._power_budget_coefficient(bs_id, state)
            for bs_id in self.config.base_station_ids
        }

        for bs_id, cluster_list in clusters.items():
            for index, members in enumerate(cluster_list):
                tx_powers = self.allocate_tx_powers(bs_id, members, pressure, rho_m_noma[bs_id])
                tx_power_map.update(tx_powers)
                mode: Literal["OMA", "NOMA"] = "OMA" if len(members) == 1 else "NOMA"
                decoding_order = self.determine_sic_order(bs_id, members, pressure)
                bandwidth = bandwidths[(bs_id, index)]
                plan = ClusterPlan(bs_id, members, mode, bandwidth, tx_powers, decoding_order)
                cluster_plans[bs_id].append(plan)

                for user_id in members:
                    link_id = self.config.access_link_id(user_id)
                    if mode == "OMA":
                        link_rates[link_id] = self.channel_model.compute_oma_rate(
                            tx_power_w=tx_powers[user_id],
                            link_gain=self.channel_model.access_gains[link_id],
                            bandwidth_hz=bandwidth,
                        )
                    else:
                        link_rates[link_id] = self.channel_model.compute_noma_rate(
                            user_id=user_id,
                            members=members,
                            tx_powers=tx_powers,
                            bandwidth_hz=bandwidth,
                            decoding_order=decoding_order,
                        )

        link_service_bits = {
            link_id: min(
                state.access_queue_bits[link_id],
                link_rates[link_id] * self.config.system.slot_length,
            )
            for link_id in self.config.access_links
        }
        backhaul_rates = {
            self.config.backhaul_link_id(source_bs, target_bs): self.channel_model.compute_backhaul_rate(
                source_bs,
                target_bs,
            )
            for source_bs, neighbors in self.config.neighbors.items()
            for target_bs in neighbors
        }

        return FastSchedulingResult(
            clusters=cluster_plans,
            user_pressure=pressure,
            link_rates=link_rates,
            link_service_bits=link_service_bits,
            tx_powers=tx_power_map,
            backhaul_rates=backhaul_rates,
            rho_m_noma=rho_m_noma,
            cluster_debug=cluster_debug,
        )

    def schedule(
        self,
        state: SystemState,
        alpha: dict[LinkTaskKey, float],
    ) -> FastSchedulingResult:
        return self.schedule_access(state=state, alpha=alpha)
