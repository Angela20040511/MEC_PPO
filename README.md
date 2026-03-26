# MEC_PPO_Project

本项目按照 `项目结构.docx` 中的模块拆分实现，核心逻辑对应：

- `env/`：承载《建模.pdf》里的任务生成、信道、计算和队列更新模型。
- `dt/`：实现《算法.pdf》中的 LSTM 数字孪生预测器。
- `scheduler/`：实现 Lyapunov 权重与 OMA/NOMA 快调度器。
- `rl/`：实现 PPO actor-critic、buffer 与训练代理。
- `simulator/`：按《算法.pdf》Algorithm 1 串联单时隙在线执行流程。
- `train.py` / `main.py`：训练与主程序入口。

## 运行方式

```bash
python main.py --mode full --steps 5
```

## 说明

文档中未给出具体工业拓扑和全部数值超参数，因此默认传感器数量、基站位置、算力参数和奖励权重统一集中在 `config.py` 中，便于后续按你的正式实验参数替换。
