> 以下五组 SAC 对照属于已撤回推荐的奖励修改试验。当前原利润参数试验请使用 [新指南](sac/README.md)中的分析工具与实际参数快照。

# 调度与训练诊断

`tools/analyze_dispatch.py` 只读取逐小时结果，不载入模型、不训练、不改动运行产物。可用于纯 SAC 或任何使用相同环境记录格式的策略。

## 命令

```bash
.venv/bin/python tools/analyze_dispatch.py \
  --run-dir runs/sac_baseline_500 \
  --output checks/sac_audit/dispatch/baseline \
  --window 20 --plots
```

脚本读取运行目录中的全部 `evaluation_*.json` 和可选 `training.csv`。环境参数默认取该运行保存的 `source/data/environment.json`，保证效率、SOC 和电价边界与历史运行一致。输出目录必须不存在，避免覆盖旧报告。

也可指定一个或多个独立评价文件：

```bash
.venv/bin/python tools/analyze_dispatch.py \
  --evaluation runs/sac_baseline_500/evaluation_00500.json \
  --environment runs/sac_baseline_500/source/data/environment.json \
  --training runs/sac_baseline_500/training.csv \
  --output checks/single_evaluation --plots
```

训练期间反复查看的定期评价是开发验证集，不是未经使用的独立测试集。脚本不推断一个外部文件是否属于独立测试集。学习曲线来自随机训练场景和随机动作，不能代替固定场景下的配对评价。

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

## 复现五组 500 回合比较

`tools/compare_sac_runs.py` 专门复现 2026-10-18 的五组探索实验：`baseline`、`centered`、`compact`、`balanced`、`peak_balanced`。它不是自动挑选任意实验或最优策略的工具。输入必须包含五个完整运行目录、每组的 11 次开发验证、`holdout_108.json`、最终检查点及冻结 `source/`。

```bash
.venv/bin/python -m pip install -r requirements-analysis.txt
.venv/bin/python tools/compare_sac_runs.py

# 也可明确指定路径。
.venv/bin/python tools/compare_sac_runs.py \
  --runs-root runs \
  --output experiments/sac_500ep_20261018
```

默认输入和输出相对于脚本所在项目根目录解析，不依赖当前工作目录。显式传入的相对路径仍按调用者工作目录解析；可以从任意目录使用解释器和脚本的绝对路径执行。支持 `python -m tools.compare_sac_runs`。

工具只更新输出目录中的 `comparison.csv`、`comparison.json`、`development_comparison.png`、`heldout_dispatch.png`；不会重训、重新评价、复制或改写已有冻结源码和逐案例原始结果。输入的训练回合、训练步数、共同设置、最终检查点存在性、冻结文件哈希、案例数量/顺序、参考策略及外生曲线一致性均会核验。日汇总与逐小时重算的主要比较指标也必须相符。

旧评价文件没有显式储存扰动档位标签。工具读取经过哈希检查的冻结场景生成器和 profile，按原协议 `case_id % 27` 对应的 `low/medium/high` 三源组合重建曲线，与记录逐点核对。108 案例必须每组恰好 4 个实现；开发集必须每组恰好 1 个实现，且两者场景种子集合不相交。

候选 `peak_balanced` 分别与 `baseline`、`centered` 比较。按 27 组分别有放回抽取 4 个配对案例，共重采样 2,000 次；不同指标和参考策略复用相同重采样索引。95% 区间是配对均值差的 2.5%/97.5% 百分位数。bootstrap 种子 `20261018` 属于固定研究协议；训练和评价种子从实际文件读取。区间只描述给定这一次训练所得模型的场景变化，不包含重新训练种子的方差。

`delta` 始终为“候选均值减参考均值”；参考均值大于 0 时才报告百分比。参考均值为负或零时 `percent` 为 `null`，尤其不能把负的储能净利润作为常规百分比改善的分母。实际电池功率使用 `charge - discharge`，不会假定最大充电和放电功率相等。图中价格界限来自运行保存的环境参数。

所需分析依赖为 NumPy、SciPy、Matplotlib；SciPy 用于冻结场景重建，Matplotlib 用于两张静态图。`requirements-analysis.txt` 提供安装入口，历史精确版本见各运行 `manifest.json`。源代码哈希验证和逐点重建是严格检查；跨数值库版本导致的场景差异会明确报错，不会自动放宽匹配条件。
