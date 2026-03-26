"""数字孪生负载预测模块。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import MECConfig, TaskKey
from env.system_state import SystemState


class ArrivalLSTM(nn.Module):
    """用 LSTM 预测任务到达序列。"""

    def __init__(self, task_count: int, hidden_size: int, num_layers: int, horizon: int) -> None:
        """初始化 LSTM 主干和预测头。"""
        super().__init__()
        self.horizon = horizon
        self.task_count = task_count
        self.lstm = nn.LSTM(
            input_size=task_count,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, horizon * task_count)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输出未来多个时隙的到达预测。"""
        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]
        prediction = self.head(last_hidden)
        return prediction.view(-1, self.horizon, self.task_count)


@dataclass
class DigitalTwin:
    """数字孪生预测器。"""

    config: MECConfig
    device: torch.device = field(init=False)
    model: ArrivalLSTM = field(init=False)
    optimizer: torch.optim.Optimizer = field(init=False)
    loss_fn: nn.Module = field(init=False)
    feature_mean: torch.Tensor = field(init=False)
    feature_std: torch.Tensor = field(init=False)
    trained: bool = False
    last_fit_history_length: int = 0

    def __post_init__(self) -> None:
        """创建模型、优化器和归一化统计量。"""
        self.device = torch.device("cpu")
        self.model = ArrivalLSTM(
            task_count=len(self.config.task_names),
            hidden_size=self.config.dt.hidden_size,
            num_layers=self.config.dt.num_layers,
            horizon=self.config.dt.prediction_horizon,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.dt.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.feature_mean = torch.zeros(len(self.config.task_names), dtype=torch.float32, device=self.device)
        self.feature_std = torch.ones(len(self.config.task_names), dtype=torch.float32, device=self.device)

    def _build_dataset(self, state: SystemState) -> tuple[torch.Tensor, torch.Tensor] | None:
        """把历史到达率窗口切成训练样本。"""
        window = self.config.dt.history_window
        horizon = self.config.dt.prediction_horizon
        features: list[np.ndarray] = []
        labels: list[np.ndarray] = []

        for sensor_id in self.config.sensor_ids:
            history = state.arrival_history[sensor_id]
            if len(history) < window + horizon:
                continue
            sequence = np.stack(history, axis=0)
            for end_index in range(window - 1, len(sequence) - horizon):
                features.append(sequence[end_index - window + 1 : end_index + 1])
                labels.append(sequence[end_index + 1 : end_index + 1 + horizon])

        if not features:
            return None

        feature_array = np.asarray(features, dtype=np.float32)
        label_array = np.asarray(labels, dtype=np.float32)
        return torch.from_numpy(feature_array), torch.from_numpy(label_array)

    def maybe_fit(self, state: SystemState) -> None:
        """在历史数据足够时训练或重训模型。"""
        dataset = self._build_dataset(state)
        if dataset is None:
            return

        features, labels = dataset
        history_length = max(len(state.arrival_history[sensor_id]) for sensor_id in self.config.sensor_ids)
        if history_length < self.config.dt.min_history_to_train:
            return
        if self.trained and history_length - self.last_fit_history_length < self.config.dt.retrain_interval:
            return

        flattened_features = features.reshape(-1, features.shape[-1])
        self.feature_mean = flattened_features.mean(dim=0, keepdim=False).to(self.device)
        self.feature_std = flattened_features.std(dim=0, keepdim=False).clamp_min(1e-6).to(self.device)

        normalized_features = (features.to(self.device) - self.feature_mean) / self.feature_std
        normalized_labels = (labels.to(self.device) - self.feature_mean) / self.feature_std
        loader = DataLoader(
            TensorDataset(normalized_features, normalized_labels),
            batch_size=self.config.dt.batch_size,
            shuffle=True,
        )

        self.model.train()
        for _ in range(self.config.dt.train_epochs):
            for batch_x, batch_y in loader:
                prediction = self.model(batch_x)
                # 这里最小化 DT 预测序列与真实序列的均方误差。
                loss = self.loss_fn(prediction, batch_y)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

        self.trained = True
        self.last_fit_history_length = history_length

    def predict(self, state: SystemState) -> dict[TaskKey, float]:
        """预测未来几个时隙的累计到达负载。"""
        horizon = self.config.dt.prediction_horizon
        predicted_loads: dict[TaskKey, float] = {}

        input_windows = []
        for sensor_id in self.config.sensor_ids:
            history_window = state.get_recent_history(sensor_id)
            input_windows.append(history_window)

        input_tensor = torch.tensor(np.stack(input_windows, axis=0), dtype=torch.float32, device=self.device)

        if self.trained:
            self.model.eval()
            with torch.no_grad():
                normalized_input = (input_tensor - self.feature_mean) / self.feature_std
                prediction = self.model(normalized_input)
                prediction = prediction * self.feature_std + self.feature_mean
                prediction = prediction.clamp_min(0.0)
        else:
            # 模型尚未训练时，用最后一个观测值填充未来时域。
            last_observation = input_tensor[:, -1:, :]
            prediction = last_observation.repeat(1, horizon, 1)

        aggregated = prediction.sum(dim=1).cpu().numpy()
        for sensor_index, sensor_id in enumerate(self.config.sensor_ids):
            for task_index, task_name in enumerate(self.config.task_names):
                predicted_loads[(sensor_id, task_name)] = float(aggregated[sensor_index, task_index])
        return predicted_loads
