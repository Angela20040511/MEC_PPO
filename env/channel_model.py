"""信道模型。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from config import MECConfig


@dataclass
class ChannelModel:
    """负责接入链路和回传链路的速率与能耗计算。"""

    config: MECConfig
    access_gains: dict[str, float] = field(init=False)
    backhaul_gains: dict[str, float] = field(init=False)

    def __post_init__(self) -> None:
        """根据几何拓扑预计算链路增益。"""
        self.access_gains = self._build_access_gains()
        self.backhaul_gains = self._build_backhaul_gains()

    def _distance_3d(
        self,
        point_a: tuple[float, float, float],
        point_b: tuple[float, float, float],
    ) -> float:
        """计算两个三维点之间的欧氏距离。"""
        return math.sqrt(sum((coord_a - coord_b) ** 2 for coord_a, coord_b in zip(point_a, point_b)))

    def _path_loss_db(self, distance_3d: float, carrier_freq_ghz: float) -> float:
        """按 3GPP 风格公式估计路径损耗。"""
        distance = max(distance_3d, 1.0)
        return 19.0 * math.log10(carrier_freq_ghz) + 21.5 * math.log10(distance) + 31.84

    def _build_access_gains(self) -> dict[str, float]:
        """预计算所有接入链路增益。"""
        gains: dict[str, float] = {}
        for sensor in self.config.sensors:
            base_station = self.config.base_station_by_id(sensor.home_bs)
            distance = self._distance_3d(sensor.position, base_station.position)
            loss_db = self._path_loss_db(distance, self.config.system.carrier_freq_ghz)
            gains[self.config.access_link_id(sensor.sensor_id)] = 10.0 ** (-loss_db / 10.0)
        return gains

    def _build_backhaul_gains(self) -> dict[str, float]:
        """预计算所有回传链路增益。"""
        gains: dict[str, float] = {}
        for source_bs, neighbor_ids in self.config.neighbors.items():
            source = self.config.base_station_by_id(source_bs)
            for target_bs in neighbor_ids:
                target = self.config.base_station_by_id(target_bs)
                distance = self._distance_3d(source.position, target.position)
                loss_db = self._path_loss_db(distance, self.config.system.backhaul_freq_ghz)
                gains[self.config.backhaul_link_id(source_bs, target_bs)] = 10.0 ** (-loss_db / 10.0)
        return gains

    def compute_snr(self, tx_power_w: float, link_gain: float, bandwidth_hz: float) -> float:
        """按 OMA 模型计算信噪比。"""
        noise = self.config.system.eta2 * self.config.system.access_noise_density * bandwidth_hz
        return self.config.system.eta1 * tx_power_w * link_gain / max(noise, self.config.system.epsilon)

    def compute_oma_rate(self, tx_power_w: float, link_gain: float, bandwidth_hz: float) -> float:
        """用 Shannon 公式计算 OMA 速率。"""
        snr = self.compute_snr(tx_power_w, link_gain, bandwidth_hz)
        return bandwidth_hz * math.log2(1.0 + snr)

    def compute_noma_rate(
        self,
        user_id: str,
        members: tuple[str, ...],
        tx_powers: dict[str, float],
        bandwidth_hz: float,
        decoding_order: tuple[str, ...],
    ) -> float:
        """根据 SIC 解码顺序计算 NOMA 用户速率。"""
        link_id = self.config.access_link_id(user_id)
        link_gain = self.access_gains[link_id]
        user_position = decoding_order.index(user_id)
        residual_users = decoding_order[user_position + 1 :]
        interference = sum(
            self.config.system.eta1
            * tx_powers[other_user]
            * self.access_gains[self.config.access_link_id(other_user)]
            for other_user in residual_users
            if other_user in members
        )
        noise = self.config.system.eta2 * self.config.system.access_noise_density * bandwidth_hz
        sinr = (
            self.config.system.eta1 * tx_powers[user_id] * link_gain
            / max(noise + interference, self.config.system.epsilon)
        )
        return bandwidth_hz * math.log2(1.0 + sinr)

    def compute_backhaul_rate(self, source_bs: str, target_bs: str) -> float:
        """计算基站间回传链路速率。"""
        link_id = self.config.backhaul_link_id(source_bs, target_bs)
        bandwidth = self.config.system.backhaul_bandwidth_hz
        gain = self.backhaul_gains[link_id]
        numerator = self.config.system.eta1_bh * self.config.system.backhaul_power_w * gain
        denominator = self.config.system.eta2_bh * self.config.system.backhaul_noise_density * bandwidth
        return bandwidth * math.log2(1.0 + numerator / max(denominator, self.config.system.epsilon))

    def compute_transmission_energy(
        self,
        tx_power_w: float,
        serviced_bits: float,
        rate_bps: float,
    ) -> float:
        """根据发送功率和发送时长计算能耗。"""
        if rate_bps <= self.config.system.epsilon:
            return 0.0
        return tx_power_w * serviced_bits / rate_bps

    def max_access_rate(self, bs_id: str) -> float:
        """估计某个基站接入速率上界。"""
        base_station = self.config.base_station_by_id(bs_id)
        covering_sensors = [sensor for sensor in self.config.sensors if sensor.home_bs == bs_id]
        if not covering_sensors:
            return base_station.uplink_bandwidth_hz
        max_power = max(sensor.tx_power_base * (1.0 + self.config.system.alpha_p) for sensor in covering_sensors)
        max_gain = max(self.access_gains[self.config.access_link_id(sensor.sensor_id)] for sensor in covering_sensors)
        return self.compute_oma_rate(max_power, max_gain, base_station.uplink_bandwidth_hz)
