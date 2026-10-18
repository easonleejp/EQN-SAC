# 调度与训练诊断

`tools/analyze_dispatch.py` 只读取逐小时结果，不载入模型、不训练、不改动运行产物。可用于纯 SAC 或任何使用相同环境记录格式的策略。

## 命令

当前SAC只有一个已保存的 `test_1ep.json`，使用显式评价文件和本轮有效环境快照：

```bash
.venv/bin/python tools/analyze_dispatch.py \
  --evaluation experiments/sac_causal13_500_20261018/run/test_1ep.json \
  --environment experiments/sac_causal13_500_20261018/run/effective_environment.json \
  --training experiments/sac_causal13_500_20261018/run/training.csv \
  --output checks/current_dispatch --window 20 --plots
```

输出目录必须不存在。此操作仅读取已保存结果，不增加策略测试回合。环境快照包含本次响应、储能与新能源覆盖参数；不能用未覆盖的 `source/data/environment.json` 替代。

`--run-dir` 模式只搜索 `evaluation_*.json`，适用于具有定期评价文件的其他运行；当前固定日结果应使用上面的 `--evaluation`。训练期间反复查看的评价属于开发验证，同一固定日上的单次测试也不代表未知日期泛化。

## 文件

| 输出 | 内容 |
| --- | --- |
| `analysis.json` | 指标定义、环境来源、各次评价的逐案例结果与均值/总体标准差、训练滚动统计 |
| `evaluation_metrics.csv` | 每次评价一行；先算每个案例的指标，再跨案例平均 |
| `case_metrics.csv` | 每个评价案例一行，包括配对参考策略、物理一致性检查 |
| `hourly.csv` | 原始逐小时记录；`role` 区分策略和参考策略 |
| `learning_curves.csv` | 原训练数据及尾随窗口均值；未更新时的空损失保留为空 |
| `learning_curves.png` | 随机训练回合的利润改善、峰值、critic loss 和温度曲线 |
| `validation_curves.png` | 固定开发验证集的利润改善、峰值削减、PAR 和电池吞吐 |
| `hourly_dispatch.png` | 最后一次评价的平均逐小时负荷、电价、充放电功率及储能量 |

`--plots` 需要 Matplotlib。缺少时仍写出 CSV/JSON，并在输出中明确记录未绘图。图上采用数据源的单位，未将未经核准的价格或利润标注为某一货币。

## 负荷与参考定义

- `baseline_load`：场景给定的无需求响应基线负荷。
- `user_load`：实际需求响应后的用户负荷。
- `baseline_grid`：`max(baseline_load - wind, 0)`，无响应、无储能的算术对照。
- `response_grid_without_storage`：`max(user_load - wind, 0)`，固定本次响应后去掉储能的算术对照。
- `grid`：`max(user_load + charge - discharge - wind, 0)`，实际电网购电功率。
- `reference.curves.grid`：保存的配对参考策略购电功率。当前 SAC 参考是名义分时零售价、无储能，仍会产生需求响应。

峰值、谷值、峰谷差、PAR、总体标准差、变异系数、相邻时段爬坡和总能量分别计算。PAR 为峰值除以均值；零均值曲线返回空值。爬坡以功率差除以时间间隔计算，不连接最后和第一个时段。

配对峰值/峰谷差/波动/PAR 削减为“参考减策略”，谷值抬升为“策略减参考”。正值分别表示对应改善。均值曲线的峰值不等于逐日峰值的平均值；CSV/JSON 使用后者。

## 储能与边界检查

SOC 在此环境中表示储存能量。每一步核对：

```text
soc_next = soc_before * eta_self + (eta_ch * charge - discharge / eta_dis) * dt
grid = max(user_load + charge - discharge - wind, 0)
```

还检查前后 SOC 连续性、SOC/功率界限、同时充放电、负荷与响应平衡、价格越界。浮点容差默认为绝对 `1e-3`、相对 `1e-7`，适配环境 float32 负荷舍入。格式错误和非有限值会报错；数值不一致会保留记录，并将 `checks.all_ok` 设为 false。

吞吐同时报告电网侧 `charge + discharge` 和电芯侧 `eta_ch * charge + discharge / eta_dis` 的能量积分。等效完整循环数采用电芯吞吐除以两倍可用储能容量。报告终值 SOC、转换损耗、自放电损耗及丢弃的多余能量。多余能量可能包含电池放电，不能全部解释为弃风。

价格上下限饱和比例默认统计距边界不超过允许价格跨度的 1%；共享价格动作下对应原始动作 `<= -0.98` 或 `>= 0.98`。

## Python 接口

```python
from tools.analyze_dispatch import (
    analyze_case, analyze_evaluation, analyze_learning, rollout_environment,
)

analysis = analyze_case(evaluation_json["cases"][0], environment_spec)
report = analyze_evaluation(evaluation_json, environment_spec)
learning = analyze_learning(list_of_training_csv_rows, window=20)

# 无模型的固定策略探针：动作使用 Environment 的物理归一化坐标。
trace = rollout_environment(env, scenario, lambda state: [0.0, 0.0], response_seed=42)
probe_analysis = analyze_case(trace, env.spec)
```

`hourly` 行必须包含 `hour, baseline_load, wind, user_load, charge, discharge, soc, price`。`hour` 从 0 连续递增。可选 `grid_purchase, initial_soc, total_response, profit` 等字段必须每行都有或全无。`reference_hourly` 可省略外生曲线，脚本使用同一案例的外生曲线；若显式提供，必须与策略一致。不会把原 EQN 仅有日汇总的评价文件推测成逐小时轨迹。

固定探针中的储能动作 `0` 表示零充放电。这与 SAC 在可行区间上仿射映射后的原始动作坐标不同；调用者负责传入实际环境动作。固定中点价格仍会产生需求响应，不代表“无控制”。

## 验证

```bash
.venv/bin/python -m unittest discover -s tests -p test_dispatch_analysis.py -v
```

测试覆盖手算的峰谷/PAR/爬坡、充放电损耗与能量守恒、半小时时步、自放电、无出口余能、错误轨迹、配对外生输入、缺失训练损失及无模型固定策略探针。
