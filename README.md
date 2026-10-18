# EQN-SAC / SAC

当前 `sac` 分支运行**原利润目标的纯 SAC**。允许调整的范围是响应方程参数、训练超参、状态空间、新能源与储能参数；不添加价格跟随规则、峰值惩罚、负荷平滑奖励或额外调度约束。价格和充放电趋势作为评价指标，不进入训练奖励。

前一轮 `centered/balanced/peak_balanced` 属于已撤回的目标修改试验，不能作为当前推荐。记录与冻结代码保存在 `experiments/sac_500ep_20261018/`，其收益和削峰结论仅属于原试验。

最新完成第二次**固定同一天训练500 ep，最终模型测试1 ep**。17维状态及参数调整后，22→23时价格由上涨0.887变为下降0.039；最大购电跳变减少34.10%，购电小时变化RMS减少32.32%。购电峰值略升0.47%，单独的用户响应负荷平滑度也有所退步。原利润奖励、响应公式和动作规则不变。见[前后六联图及完整取舍](experiments/sac_fixed_day_refined_500_20261018/README.md)与[参数说明](docs/sac/fixed-day-refinement.md)。

前一次固定日500+1记录保存在[原基线结果](experiments/sac_fixed_day_500_20261018/README.md)，没有覆盖。两次均无环境随机扰动；SAC训练探索仍有随机性。

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
