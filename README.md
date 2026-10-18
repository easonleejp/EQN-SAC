# EQN-SAC / 纯SAC causal13

当前 `sac` 分支只保留一个SAC实验版本：**causal13，固定同一天训练500回合，最终测试1回合**。入口为[当前配置](configs/sac/causal13_500.json)、[SAC运行指南](docs/sac/README.md)及[实验结果与检查点](experiments/sac_causal13_500_20261018/README.md)。

旧SAC配置、独立实验记录和专用说明已从当前工作树删除，仍可通过Git历史追溯。本次整理不改变当前模型、训练结果或物理公式，不重新训练或评价。原EQN代码、三组历史EQN档案及 `main` 分支不在本次删除范围内。

## 当前SAC

一个联合Gaussian Actor输出零售价和储能两维动作，两个标量Critic学习原利润的软价值。环境奖励为 `0.001×原利润`，没有价格跟随规则、削峰/平滑奖励或额外调度约束。13维状态只使用当前信息、过去三小时批发价及当前物理可行边界，详见[逐维定义](docs/sac/causal-observation.md)。

当前保存的单次测试：日利润原值1,169,753.64，购电峰值3314.20 kW，购电相邻变化RMS为254.80 kW；h18开始可用储能量1976.14 kWh，h17–20放电2547.99 kWh。最大购电跳变仍有779.30 kW，发生在h2→3，不能称为全面平滑。利润单位采用澳分的建模假设，原始来源单位尚未核实；单seed固定日结果不证明未知场景泛化。

查看[当前图表、原始单次数据和来源清单](experiments/sac_causal13_500_20261018/README.md)无需运行模型。当前检查点位于[本版实验目录](experiments/sac_causal13_500_20261018/run/checkpoint_00500.pt)，配套冻结源码在同目录的 `source/`。

## 安装与独立复现

推荐Python 3.12，以下命令从仓库根目录执行。查看已有结果无需重训；下方是需要独立复现时使用的操作说明，本次文件整理没有执行这些命令。

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt

python experiments/sac_causal13_500_20261018/run/source/train_sac.py train \
  --config configs/sac/causal13_500.json \
  --output runs/sac_causal13_500_reproduce

python runs/sac_causal13_500_reproduce/source/train_sac.py evaluate \
  --checkpoint runs/sac_causal13_500_reproduce/checkpoint_00500.pt \
  --output runs/sac_causal13_500_reproduce/test_1ep.json \
  --cases 1 --seed 20260914
```

Windows激活命令为 `.venv\Scripts\activate`。复现输出目录必须尚不存在。配置关闭内部评价；每次完整复现训练500回合后，只执行一次最终测试。训练与评价均使用对应冻结源码，避免活动代码变化影响检查点来源核验。更完整的状态、奖励和离线绘图说明见[SAC指南](docs/sac/README.md)。

## 原EQN项目与目录

原EQN-SAC采用两个Actor和两套集中式分位数集成，与当前纯SAC架构不同。`train.py` 和 `eqnsac/` 继续保留；其基础组及两个扩展不因本次SAC文件整理发生改变。使用方法见[原EQN运行指南](docs/usage.md)，历史来源见[实验索引](experiments/README.md)。

| 路径 | 用途 |
|---|---|
| `train_sac.py`、`sac/` | 当前纯SAC开发入口与实现 |
| `configs/sac/causal13_500.json` | 唯一活动SAC实验配置 |
| `experiments/sac_causal13_500_20261018/` | 当前SAC结果、冻结源码、检查点和来源清单 |
| `train.py`、`eqnsac/` | 保留的原EQN训练入口、环境和算法 |
| `data/`、`configs/` | 输入数据、物理参数及原EQN配置 |
| `experiments/historical/`、`experiments/summary/` | 三组历史EQN冻结记录和复查摘要 |
| `docs/`、`tests/`、`tools/` | 说明、测试与分析工具 |
| `archive/` | 原EQN本地归档说明与文件哈希清单 |
| `runs/`、`checks/` | 本地运行及检查产物 |

原项目背景可从[架构](docs/architecture.md)、[数据说明](docs/data.md)和[算法审查](docs/review/algorithm-review-20261018.md)阅读。原始工作簿及历史EQN权重的本机保存范围见[归档说明](archive/README.md)；它们与本版SAC实验目录内的检查点应分别理解。
