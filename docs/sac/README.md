# 纯 SAC 运行指南

本分支用普通 SAC 分别验证利润目标、观测压缩和负荷曲线目标。当前建议从 `peak_balanced_500.json` 继续做多种子验证；它是 500 episode 的候选配置，不是已经证明收敛的最终参数。结果与期刊依据见[审核报告](review-20261018.md)。

## 代码入口

| 路径 | 职责 |
|---|---|
| `train_sac.py` | 训练 / 评价命令行 |
| `sac/config.py` | 明确配置及输入校验 |
| `sac/task.py` | 观测、可行动作映射、利润和负荷目标、日指标 |
| `sac/networks.py`、`sac/agent.py` | 联合高斯 Actor、双标量 Q、温度及更新 |
| `sac/replay.py` | 均匀回放、输入校验和独立 RNG |
| `sac/runner.py` | 训练循环、冻结源码、配对评价、检查点 |
| `configs/sac/` | 五组可直接运行的配置 |
| `tools/analyze_dispatch.py` | 独立复算收益与曲线指标、检查物理约束 |
| `tools/compare_sac_runs.py` | 汇总五组试验并生成比较图 |
| `experiments/sac_500ep_20261018/` | 已完成实验的配置、源码、日志和评价 |

`eqnsac/environment.py`、`scenarios.py` 和 `single_source.py` 提供共享业务环境；不导入 EQN Agent、分位网络或风险模块。保留这种共用关系是为了避免两套物理模型在维护中发生偏差。

## 训练与评价

在仓库根目录，用 Python 3.12 安装 `requirements.txt`；绘图另安装 `requirements-analysis.txt`。本轮运行版本为 Python 3.12.14、PyTorch 2.7.1、NumPy 2.5.4、SciPy 1.18.1，CPU 单线程。具体哈希和版本见每组 `manifest.json`。

```sh
python -m pip install -r requirements-analysis.txt
python -m unittest discover -s tests -v
python train_sac.py train --config configs/sac/baseline_500.json --output runs/sac_baseline_new
python train_sac.py train --config configs/sac/peak_balanced_500.json --output runs/sac_peak_balanced_new
python runs/sac_peak_balanced_new/source/train_sac.py evaluate --checkpoint runs/sac_peak_balanced_new/checkpoint_00500.pt --output runs/sac_peak_balanced_new/heldout.json --cases 108 --seed 930000000
python tools/analyze_dispatch.py --evaluation runs/sac_peak_balanced_new/heldout.json --environment data/environment.json --output checks/sac_peak_balanced_new --plots
```

输出目录/评价文件必须尚不存在，以避免覆盖实验。加载模型时核验关键源码和数据哈希；开发代码变动后应使用该次训练的 `source/train_sac.py`。检查点用于评价，当前 CLI **不提供训练恢复**：其中没有完整回放、场景与探索 RNG，因此不能把加载模型称为无缝续训。只加载可信的本地检查点。

每次训练自动保存源码、输入 JSON、配置、依赖版本、每回合 CSV、固定 27 案例阶段评价以及每 100 回合检查点。训练抽样 RNG、探索 RNG、策略采样和评价相互隔离；确定性评价不消耗训练策略 RNG。训练为 24 步一日，`done` 屏蔽终态 bootstrap。

## 本轮配置

| 配置 | 观测维数 | 经济奖励 | 负荷偏差权重 | 日峰值权重 |
|---|---:|---|---:|---:|
| `baseline_500.json` | 19 | 原利润 | 0 | 0 |
| `centered_500.json` | 19 | 利润减固定策略参考利润 | 0 | 0 |
| `compact_500.json` | 11 | 同上 | 0 | 0 |
| `balanced_500.json` | 19 | 同上 | 0.5 | 0 |
| `peak_balanced_500.json` | 20 | 同上 | 0.5 | 1.0 |

全部采用 seed 42、两层 128 隐藏单元、批量 128、回放 50000、随机探索前 1000 步、Actor/Q/温度学习率 `3e-4`、初始温度 `0.2`、目标熵 `-2`、`tau=0.005`、奖励缩放 `0.001`、`gamma=1`。峰值目标配置要求 `gamma=1`，保证逐步峰值增量之和正好等于日峰值。

推荐配置保留原 19 维，再加入“本日截至当前的最大购电功率”。11 维组保留为对照，本轮没有表现出更好的综合结果。所有维度都基于当前或历史可见信息；名义曲线是已知参考，不是未来实际场景。

Actor 的两维原始动作都在 `[-1,1]`。储能动作只在执行时映射到 SOC 可行区间，回放、Q 和熵始终使用原始动作坐标。原始动作零不一定代表电池停机；绘图应使用实际充放电量，停机参考策略直接使用物理零动作。详细推导见[审核报告](review-20261018.md)。

## 继续研究时的比较纪律

本轮 27 场景用于开发调参，108 场景是在五组配置完成后一次性评价的均衡压力套件。27 种档位组合等权，不等于训练时 `.2/.6/.2` 档位概率下的期望收益。若继续使用这 108 场景调参，它们就应视为开发集，并另留最终测试集。

下一轮应先验证多训练种子及更长训练曲线，再调整权重或模型。连续多日经营、真实价格单位、跨时需求回补和电池退化参数均属于业务建模决定，不能靠提高训练轮数解决。
