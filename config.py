"""项目配置定义。"""
# 放系统常量、拓扑、任务类型、基站计算单元、训练超参数

from __future__ import annotations

from dataclasses import dataclass, replace

TaskKey = tuple[str, str]
LinkTaskKey = tuple[str, str]
BsTaskKey = tuple[str, str]
LocalTaskKey = tuple[str, str]
RoutingKey = tuple[str, str, str]
BsUnitTaskKey = tuple[str, str, str]


@dataclass(frozen=True)
class TaskTypeSpec:
    """单类任务的静态参数。"""

    name: str
    sensitivity: int
    compute_intensity: float
    mean_data_bits: float
    std_data_bits: float
    mmpp_rates: dict[int, float]


@dataclass(frozen=True)
class SensorSpec:
    """传感器节点的静态信息。"""

    sensor_id: str
    home_bs: str
    position: tuple[float, float, float]
    local_cpu_rate: float
    local_energy_coeff: float
    tx_power_base: float


@dataclass(frozen=True)
class ComputeUnitSpec:
    """基站计算单元参数。"""

    unit_name: str
    base_rate: float
    energy_coeff: float
    acceleration: dict[str, float]


@dataclass(frozen=True)
class BaseStationSpec:
    """边缘基站的拓扑和资源配置。"""

    bs_id: str
    position: tuple[float, float, float]
    uplink_bandwidth_hz: float
    compute_units: tuple[ComputeUnitSpec, ...]
    energy_pressure_baseline: float = 0.2


@dataclass(frozen=True)
class DTConfig:
    """数字孪生模型的超参数。"""

    history_window: int = 8
    prediction_horizon: int = 4
    hidden_size: int = 64
    num_layers: int = 1
    learning_rate: float = 1e-3
    batch_size: int = 32
    train_epochs: int = 5
    min_history_to_train: int = 20
    retrain_interval: int = 5


@dataclass(frozen=True)
class PPOConfig:
    """PPO 算法的超参数。"""

    hidden_size: int = 128
    actor_log_std: float = -1.0
    actor_structure_mode: str = "flat_joint_actor"
    critic_arch_mode: str = "baseline_critic"
    critic_input_mode: str = "current_stronger_input"
    actor_input_mode: str = "current_actor_input"
    actor_raw_input_mode: str = "baseline_actor_raw"
    policy_ratio_mode: str = "current_joint_sum_ratio"
    actor_learning_rate: float = 5e-4
    critic_learning_rate: float = 5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coeff: float = 5e-4
    value_coeff: float = 0.5
    value_target_mode: str = "raw_return"
    value_loss_mode: str = "mse"
    update_epochs: int = 8
    mini_batch_size: int = 32
    max_grad_norm: float = 0.5
    policy_update_diagnosis_enabled: bool = False
    policy_update_bucket_quantile: float = 0.75
    policy_update_near_zero_adv_std_scale: float = 0.1
    policy_update_probe_noise_seed: int = 2025
    theta_kl_target: float = 0.03
    route_kl_target: float = 0.015
    coupled_stop_severity_factor: float = 1.0
    coupled_stop_warmup_epochs: int = 0
    coupled_stop_min_theta_updates_per_epoch: int = 0
    route_confident_mask_margin: float = 0.0
    route_update_cap_per_epoch: int = 0
    route_alignment_gate_threshold: float = 0.0


@dataclass(frozen=True)
class TrainingConfig:
    """训练过程配置。"""

    num_epochs: int = 20
    time_steps: int = 50
    seed: int = 7


@dataclass(frozen=True)
class SystemConfig:
    """系统级常量。"""

    slot_length: float = 1.0
    carrier_freq_ghz: float = 3.5
    backhaul_freq_ghz: float = 5.8
    access_noise_density: float = 1e-17
    backhaul_noise_density: float = 1e-17
    eta1: float = 1.0
    eta2: float = 1.0
    eta1_bh: float = 1.0
    eta2_bh: float = 1.0
    epsilon: float = 1e-9

    min_service_ratio: float = 0.05
    min_access_rate_bps: float = 1e6
    min_backhaul_rate_bps: float = 1e6
    delay_cap: float = 1000.0

    alpha_s: float = 0.35
    alpha_p: float = 0.45
    eta_g: float = 0.6
    eta_u: float = 0.4
    noma_quantile: float = 0.85
    max_cluster_size: int = 3
    rho0: float = 0.85
    gamma_e: float = 0.15
    gamma_q: float = 0.2
    rho_min: float = 0.35
    rho_max: float = 1.0
    q_target_delay: float = 0.4
    q_max: float = 2.0
    backhaul_bandwidth_hz: float = 80e6
    backhaul_power_w: float = 5.0
    lyapunov_v: float = 10.0
    omega_e: float = 0.05
    omega_t: float = 1.0
    omega_q: float = 2.0
    reward_mode: str = "raw_total"
    queue_norm_bits: float = 5e6
    queue_norm_cycles: float = 5e9
    energy_budget_alpha: float = 0.7
    mmpp_transition: tuple[tuple[float, float, float], ...] = (
        (0.72, 0.22, 0.06),
        (0.18, 0.64, 0.18),
        (0.08, 0.27, 0.65),
    )


@dataclass(frozen=True)
class MECConfig:
    """MEC 系统总配置。"""

    tasks: tuple[TaskTypeSpec, ...]
    sensors: tuple[SensorSpec, ...]
    base_stations: tuple[BaseStationSpec, ...]
    neighbors: dict[str, tuple[str, ...]]
    local_priority: tuple[str, ...]
    dt: DTConfig
    ppo: PPOConfig
    training: TrainingConfig
    system: SystemConfig

    @property
    def task_names(self) -> tuple[str, ...]:
        """返回任务类型列表。"""
        return tuple(task.name for task in self.tasks)

    @property
    def sensor_ids(self) -> tuple[str, ...]:
        """返回传感器编号列表。"""
        return tuple(sensor.sensor_id for sensor in self.sensors)

    @property
    def base_station_ids(self) -> tuple[str, ...]:
        """返回基站编号列表。"""
        return tuple(bs.bs_id for bs in self.base_stations)

    @property
    def access_links(self) -> tuple[str, ...]:
        """返回所有接入链路标识。"""
        return tuple(self.access_link_id(sensor.sensor_id) for sensor in self.sensors)

    @property
    def static_state_dim(self) -> int:
        """状态向量中静态特征的维度。"""
        effective_rate_features = sum(
            len(bs.compute_units) * len(self.tasks) for bs in self.base_stations
        )
        reachable_mask_features = len(self.sensors) * len(self.base_stations)
        return reachable_mask_features + len(self.sensors) + effective_rate_features

    @property
    def state_dim(self) -> int:
        """提供给 PPO 的状态维度。"""
        sensor_count = len(self.sensors)
        task_count = len(self.tasks)
        bs_count = len(self.base_stations)
        dynamic_dim = (
            4 * sensor_count * task_count
            + len(self.access_links)
            + len(self.access_links) * task_count
            + bs_count * task_count
            + sensor_count * task_count
        )
        return dynamic_dim + self.static_state_dim

    @property
    def action_dim(self) -> int:
        """PPO 动作向量的维度。"""
        return len(self.sensors) * len(self.tasks) * (1 + len(self.base_stations))

    def task_by_name(self, task_name: str) -> TaskTypeSpec:
        """根据名称查找任务配置。"""
        for task in self.tasks:
            if task.name == task_name:
                return task
        raise KeyError(f"Unknown task type: {task_name}")

    def sensor_by_id(self, sensor_id: str) -> SensorSpec:
        """根据编号查找传感器。"""
        for sensor in self.sensors:
            if sensor.sensor_id == sensor_id:
                return sensor
        raise KeyError(f"Unknown sensor: {sensor_id}")

    def base_station_by_id(self, bs_id: str) -> BaseStationSpec:
        """根据编号查找基站。"""
        for bs in self.base_stations:
            if bs.bs_id == bs_id:
                return bs
        raise KeyError(f"Unknown base station: {bs_id}")

    def reachable_base_stations(self, sensor_id: str) -> tuple[str, ...]:
        """返回某个传感器可到达的基站集合。"""
        home_bs = self.sensor_by_id(sensor_id).home_bs
        return (home_bs, *self.neighbors.get(home_bs, ()))

    def access_link_id(self, sensor_id: str) -> str:
        """构造接入链路标识。"""
        sensor = self.sensor_by_id(sensor_id)
        return f"{sensor.sensor_id}->{sensor.home_bs}"

    def backhaul_link_id(self, source_bs: str, target_bs: str) -> str:
        """构造回传链路标识。"""
        return f"{source_bs}->{target_bs}"

    def max_lambda_for_task(self, task_name: str) -> float:
        """返回任务到达率上界。"""
        task = self.task_by_name(task_name)
        return max(task.mmpp_rates.values())

    def max_data_rate_for_task(self, task_name: str) -> float:
        """返回任务数据速率估计上界。"""
        task = self.task_by_name(task_name)
        return self.max_lambda_for_task(task_name) * (
            task.mean_data_bits + 3.0 * task.std_data_bits
        )

    def max_workload_rate_for_task(self, task_name: str) -> float:
        """返回任务计算负载率估计上界。"""
        task = self.task_by_name(task_name)
        return task.compute_intensity * self.max_data_rate_for_task(task_name)


def build_default_config() -> MECConfig:
    """构造一套可直接运行的默认配置。"""
    tasks = (
        TaskTypeSpec(
            name="LatCrit",
            sensitivity=2,
            compute_intensity=1200.0,
            mean_data_bits=2.0e5,
            std_data_bits=4.0e4,
            mmpp_rates={1: 0.8, 2: 1.6, 3: 3.0},
        ),
        TaskTypeSpec(
            name="ComHvy",
            sensitivity=1,
            compute_intensity=2500.0,
            mean_data_bits=1.6e5,
            std_data_bits=3.0e4,
            mmpp_rates={1: 0.6, 2: 1.1, 3: 2.1},
        ),
        TaskTypeSpec(
            name="BWHvy",
            sensitivity=0,
            compute_intensity=500.0,
            mean_data_bits=3.8e5,
            std_data_bits=5.0e4,
            mmpp_rates={1: 0.7, 2: 1.3, 3: 2.8},
        ),
    )

    sensors = (
        SensorSpec("S1", "BS1", (8.0, 6.0, 1.5), 1.6e9, 1.4e-10, 0.32),
        SensorSpec("S2", "BS1", (18.0, -4.0, 1.5), 1.4e9, 1.6e-10, 0.30),
        SensorSpec("S3", "BS2", (92.0, 8.0, 1.5), 1.5e9, 1.5e-10, 0.31),
        SensorSpec("S4", "BS2", (108.0, -6.0, 1.5), 1.45e9, 1.55e-10, 0.29),
    )

    compute_units = (
        ComputeUnitSpec(
            unit_name="CPU",
            base_rate=3.5e9,
            energy_coeff=2.2e-10,
            acceleration={"LatCrit": 1.0, "ComHvy": 1.0, "BWHvy": 1.0},
        ),
        ComputeUnitSpec(
            unit_name="GPU",
            base_rate=5.0e9,
            energy_coeff=2.8e-10,
            acceleration={"LatCrit": 0.9, "ComHvy": 2.6, "BWHvy": 1.4},
        ),
        ComputeUnitSpec(
            unit_name="ACC",
            base_rate=4.2e9,
            energy_coeff=1.9e-10,
            acceleration={"LatCrit": 1.8, "ComHvy": 1.3, "BWHvy": 2.1},
        ),
    )

    base_stations = (
        BaseStationSpec("BS1", (0.0, 0.0, 10.0), 20e6, compute_units, 0.18),
        BaseStationSpec("BS2", (100.0, 0.0, 10.0), 20e6, compute_units, 0.22),
    )

    neighbors = {"BS1": ("BS2",), "BS2": ("BS1",)}

    return MECConfig(
        tasks=tasks,
        sensors=sensors,
        base_stations=base_stations,
        neighbors=neighbors,
        local_priority=("LatCrit", "ComHvy", "BWHvy"),
        dt=DTConfig(),
        ppo=PPOConfig(),
        training=TrainingConfig(),
        system=SystemConfig(),
    )


def build_dense_topology_config() -> MECConfig:
    """Build a denser 2-BS / 8-sensor topology for scheduler studies."""
    base_config = build_default_config()
    sensors = (
        SensorSpec("S1", "BS1", (6.0, 8.0, 1.5), 1.6e9, 1.4e-10, 0.32),
        SensorSpec("S2", "BS1", (12.0, -6.0, 1.5), 1.5e9, 1.5e-10, 0.31),
        SensorSpec("S3", "BS1", (18.0, 5.0, 1.5), 1.45e9, 1.55e-10, 0.30),
        SensorSpec("S4", "BS1", (22.0, -8.0, 1.5), 1.4e9, 1.6e-10, 0.29),
        SensorSpec("S5", "BS2", (82.0, 7.0, 1.5), 1.6e9, 1.4e-10, 0.32),
        SensorSpec("S6", "BS2", (88.0, -5.0, 1.5), 1.5e9, 1.5e-10, 0.31),
        SensorSpec("S7", "BS2", (96.0, 6.0, 1.5), 1.45e9, 1.55e-10, 0.30),
        SensorSpec("S8", "BS2", (108.0, -7.0, 1.5), 1.4e9, 1.6e-10, 0.29),
    )
    return replace(base_config, sensors=sensors)


def build_config(topology_mode: str = "default") -> MECConfig:
    """Build a config for the requested topology mode."""
    if topology_mode == "default":
        return build_default_config()
    if topology_mode == "dense":
        return build_dense_topology_config()
    raise ValueError(f"Unsupported topology_mode: {topology_mode}")
