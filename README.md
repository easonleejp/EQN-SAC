# EQN-SAC / SAC

当前 `sac` 分支运行**原利润目标的纯 SAC**。允许调整的范围是响应方程参数、训练超参、状态空间、新能源与储能参数；不添加价格跟随规则、峰值惩罚、负荷平滑奖励或额外调度约束。价格和充放电趋势作为评价指标，不进入训练奖励。

前一轮 `centered/balanced/peak_balanced` 属于已撤回的目标修改试验，不能作为当前推荐。记录与冻结代码保存在 `experiments/sac_500ep_20261018/`，其收益和削峰结论仅属于原试验。

最新完成**仅当前与历史信息的13维状态扩展：固定日500 ep＋最终1 ep测试**。在前一6维状态上加入时钟、前三小时批发价、当前储能可行边界，其他参数、奖励与动作映射不变。18时可用储能48.89→1976.14 kWh，17–20时放电39.95→2547.99 kWh；购电RMS343.69→254.80 kW、峰值3553.36→3314.20 kW，利润增加0.331%。但凌晨2→3时充电突增，最大购电跳变710.76→779.30 kW，尚非全面平滑。见[前后六联图与完整结果](experiments/sac_causal13_500_20261018/README.md)和[13维信息边界](docs/sac/causal-observation.md)。

此前完成**6维状态下的训练超参调整：固定日500 ep＋最终1 ep测试**。仅改预热1200步、Actor学习率0.00002、目标熵−2。中午均价32.082→31.892、削减率14.74%→13.19%，日利润基本持平；储能仍有96.72%的放电集中于8–15时，购电RMS329.69→343.69 kW，提前放空问题未解决。本轮保留完整结果，不视为调度优化成功。见[前后六联图](experiments/sac_current6_tuned_500_20261018/README.md)和[训练设置](docs/sac/current6-training.md)。

此前完成**仅当前6维观测：固定同一天训练500 ep，最终模型测试1 ep**。网络只接收当前批发价、原负荷、风电、SOC、剩余时间、当前TOU；去掉未来价格和响应模型特征，归一化也不用整日统计。其余参数、奖励与约束不变。相对17维，最大购电跳变908.16→711.08 kW，但购电RMS278.56→329.69 kW、峰值3365.46→3573.18 kW；中午价格及削减回升，储能提前放空，日利润下降0.242%。这是满足信息边界的状态消融结果，尚非调度改善方案。见[前后六联图与完整结果](experiments/sac_current6_500_20261018/README.md)和[6维定义](docs/sac/current-observation.md)。

前一次**中午降价候选**仅继续调整逐时a、b，10–14时零售均价32.470→31.667，下降2.47%；11–15时整体削减12.82%→11.63%，但12时单独削减有所加重。日利润下降0.27%，购电峰值增加13.29%、最大跳变增加102.09%，因此保留为降价候选，不能视为全面改善的最终方案。见[前后六联图与完整结果](experiments/sac_midday_price_500_20261018/README.md)和[参数依据](docs/sac/midday-price.md)。该轮原利润奖励、响应公式、状态结构、训练超参、储能及动作规则不变。

上一轮逐小时二次效用系数 **a、b** 调整，使11–15时加权削减率由18.65%降至12.82%，11、12时由21.71%、20.12%降至11.84%、10.77%。该轮[六联图与取舍](experiments/sac_hourly_utility_500_20261018/README.md)和[系数映射说明](docs/sac/hourly-utility.md)保留作为本轮对照。两轮均属于人工响应参数校准，利润变化不能直接归因于同一市场下的算法提升。

上一轮17维状态及参数调整的[固定日结果](experiments/sac_fixed_day_refined_500_20261018/README.md)与[参数说明](docs/sac/fixed-day-refinement.md)仍保留，作为本轮对照。

更早的固定日500+1记录保存在[原基线结果](experiments/sac_fixed_day_500_20261018/README.md)，没有覆盖。各次固定日运行均无环境随机扰动；SAC训练探索仍有随机性。

之前的16组随机场景参数试验属于历史探索，超出了本次500+1的验证范围，不能与当前单日结果混用。记录保留在[历史参数试验](experiments/sac_parameter_revision_20261018/README.md)。

当前入口为 `train_sac.py`、`sac/`、`configs/sac/`。本轮参数核对、20% 新能源渗透率和储能规模依据见 [参数审查](docs/sac/parameter-review-20261018.md)，运行方法见 [SAC 指南](docs/sac/README.md)。

下方保留原 EQN 项目的说明；其 `train.py --arm original` 不是纯 SAC。两个创新扩展仍暂不研究。

面向24小时售电定价与储能联合控制的强化学习研究项目。两个 Actor 分别控制零售价与储能动作，两套集中式 IQN 集成学习各自奖励。当前根目录是唯一开发入口；三组已有5000轮实验使用各自冻结代码。

当前工作集中于代码正确性、可读性与工程结构。自适应 CVaR 和 Tail 惩罚两个扩展模块保留现状，暂不继续设计或评估。

## 快速开始

推荐 Python 3.12。以下命令在仓库根目录运行：

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python train.py --help
```

Windows 激活命令为 `.venv\Scripts\activate`。本机已有 `.venv`，可直接使用 `.venv/bin/python`。

进行环境检查与3轮小模型训练验证：

```sh
python tools/audit.py --output checks/audit_new
```

开始正式基础组训练（默认5000轮），或评价已有模型：

```sh
python train.py train --config configs/baseline.json --arm original --output runs/original_new
python train.py evaluate --checkpoint runs/original_new/checkpoint.pt --output runs/evaluation_new.json --cases 270
```

`original` 指本项目 EQN-SAC 基础组。单因素配置中的 `device` 为 `cuda`；CPU 使用者应复制配置并改为 `cpu`。请为每次实验选择新的输出目录。完整参数、恢复与结果解释见[运行指南](docs/usage.md)。

## 目录

| 路径 | 用途 |
|---|---|
| `train.py` | 训练、恢复与评价的统一入口 |
| `eqnsac/` | 环境、场景、网络、回放与训练算法 |
| `configs/` | 基础组、候选及三组单因素配置 |
| `data/` | 运行必需的名义日、物理参数和来源 JSON |
| `tests/` | 数值、环境、快照、配置与训练回归检查 |
| `tools/` | 环境审查与性能测量工具 |
| `docs/` | 架构、数据来源、运行说明和已完成审查 |
| `experiments/` | 三组历史实验的冻结代码、日志、评价和摘要 |
| `archive/` | 本地完整归档说明与文件哈希清单 |
| `runs/`、`checks/` | 新训练与检查产物，自动排除 Git |

阅读建议：[项目架构](docs/architecture.md) → [数据说明](docs/data.md) → [运行指南](docs/usage.md)。原论文公式核对见[已完成算法审查](docs/review/algorithm-review-20261018.md)，历史结果见[实验索引](experiments/README.md)。

## 复现范围

本项目是 EQN-SAC 扩展，保留 EQN 的集成分位网络核心。基础组也包含连续随机 Actor、软 Bellman 目标和双角色奖励，不能直接称为原 EQN 论文的逐项复现。

仓库包含当前运行所需的全部代码与训练 JSON，以及三组历史实验的轻量复现记录。原始工作簿、CSV、论文附件、完整审查备份和模型权重保存在本机 `archive/local/`，不随 Git 上传；获取范围与 SHA256 见[归档说明](archive/README.md)。没有权重的远程副本可以重新训练、审查历史日志，但不能直接评价已有最终模型。

默认环境逐日重新初始化 SOC，日末无残值；风电参考日由场景模型生成。历史实验仅为单 seed 基础组的单因素诊断。当前测试通过不等于完成新的5000轮训练，也不构成两个扩展模块的有效性证据。
