# 运行指南

更新于2026年10月18日。仓库根目录是后续开发入口，检查点版本为 `clean-2`。本次修复数值计算、输入校验、回放快照和断点恢复，保留两 Actor、两套各5成员 IQN Critic、19维全局状态、奖励与场景定义。随后已完成可读性重构与原EQN论文对照，见[项目整体分析与原论文对照](review/algorithm-review-20261018.md)。第一轮数值修复报告保存在本机 `archive/local/代码审查_20261018/项目审查报告.md`。

三组 Original 单因素实验已各完成5000轮，其模型和冻结代码位于 本机 `archive/local/EQN-SAC_项目交接_20261018/05_近三次实验完整记录`；轻量副本见[历史实验](../experiments/README.md)。新代码的测试与短训练不代表重新完成这些实验。10月8日的审查报告保留在本地交接归档内。

## 模块职责

| 模块 | 职责 |
|---|---|
| `config.py` | 配置类型、范围、概率与未知字段校验 |
| `scenarios.py` | 三源负荷、风电和购电价场景生成 |
| `single_source.py` | 负荷、新能源或电价的单日缩放实验 |
| `environment.py` | 19维状态、响应、储能可行域、结算和输入快照 |
| `networks.py` | Actor、储能动作映射、IQN及分位损失 |
| `risk.py` | 原EQN两类不确定性诊断、自适应风险水平和Tail惩罚 |
| `replay.py` | 环形回放、固定 bootstrap 成员归属和独立快照 |
| `agent.py` | 两套目标更新、温度优化、非有限值定位 |
| `runner.py` | 训练、固定评价、来源核验、故障记录和恢复 |
| `tests` | 环境、数值、快照、输入与恢复回归检查 |
| `tools` | 显式执行的统计审查和短时性能测量 |

## 环境和检查

依赖见 `requirements.txt`。本机已在项目根目录 `.venv` 安装 Python3.12、PyTorch2.7.1、NumPy 和 SciPy；精确版本见[验证环境](validation-environment.txt)。从仓库根目录运行：

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python train.py --help
```

其他电脑先创建独立虚拟环境并安装 `requirements.txt`，再用该环境的 `python` 执行相同命令。

```sh
python tools/audit.py --output checks/audit_new
python tools/benchmark.py --output checks/benchmark_new.json --threads 1
```

`audit.py` 生成900天环境统计、90天固定策略统计，并进行3轮小模型训练及断点恢复比较。`benchmark.py` 仅执行7次完整规模更新，其中5次计时。输出目录或文件已存在时会拒绝覆盖。导入这两个模块不会启动计算。基准线程数会写入结果；不能直接当作默认线程配置下的正式训练速度。

## 新训练与评价

以下命令由使用者显式执行，会按配置开展训练；baseline 默认5000轮：

```sh
python train.py train --config configs/baseline.json --arm original --output runs/original_new
python train.py train --config configs/baseline.json --arm original --output runs/original_new --resume runs/original_new/checkpoint.pt --episodes 6000
python train.py evaluate --checkpoint runs/original_new/checkpoint.pt --output evaluation_new.json --cases 270
```

这里的 `original` 是本项目的EQN-SAC基础组，既不是普通SAC，也不是原EQN论文的直接复现。其他 arm 为 `adaptive_only`、`tail_only`、`full`；当前保留兼容，两个创新模块暂不继续调整。单因素实验应显式选择 `original_load_scale_5000.json`、`original_wind_scale_5000.json` 或 `original_price_scale_5000.json`；这些原配置指定 CUDA，本机 CPU 运行需另存配置并将 device 改为 `cpu`。`storage_candidate.json` 同时改变观测、学习率、预热和响应随机流，只是候选配置，不能用于单因素归因。

## 检查点和故障处理

新检查点内嵌源码与数据 SHA256、实际设备、Python/PyTorch/NumPy/SciPy版本。恢复会核验这些信息、原始 manifest、配置以及截至检查点的完整日志；配置仅允许修改 episodes/device 字段，实际 CPU/CUDA 设备仍须一致。恢复不会重写原 manifest。历史 `clean-1` 检查点可评价，结果会标记 `legacy_unverified`，新入口拒绝把它们作为同版本续训；历史结果复查优先使用同组冻结代码。

训练中的非有限损失、梯度、参数、Adam状态、动作或环境输出会中止运行，`failure.json` 记录轮次、小时、场景种子与错误位置。不会用零替换 NaN，也不会覆盖最后有效检查点。恢复时，超过检查点的日志和评价以及原完成/失败记录先保存在 `recovery_archive`。

日志的 `price_risk_level`、`storage_risk_level` 是风险水平，不是CVaR数值。`price_aleatoric_std/storage_aleatoric_std` 是先平均成员再跨分位计算的标准差；新增 `price_epistemic_std/storage_epistemic_std` 是先平均分位再跨成员计算的原EQN认知标准差。原来的 `price_ue/storage_ue` 实际是尾部分歧，现在明确命名为 `price_tail_disagreement/storage_tail_disagreement`。这项惩罚仍沿用既有Actor目标；日志与Agent共享字段顺序。评价 `cvar10` 按精确10%经验下尾质量计算，边界样本使用部分权重；100案例时仍取最差10个。`seconds`/`wall_seconds` 累计已提交检查点对应的训练片段，不包含中断后丢失的工作和全部初始化开销，不应解释为总算力耗时。

## 模型和结论边界

- 当前没有经济后备、示范预训练或行为克隆；SOC与功率约束属于环境物理限制。
- 训练数据来自名义24小时曲线；近期风电是合成参考日，尚未据实测误差校准。价格和成本的货币单位仍需核准。
- 每日终端SOC自由、无残值；次日重新初始化。当前实现不代表连续跨日运营。
- Adaptive 风险水平停止梯度并使用离散分位选择；Tail 是尾部分位分歧代理。软Q含奖励尺度和熵项，不是现金利润分布。
- 固定评价集已反复用于监测，不能当作最终未见测试集。三组实验均为单seed Original，不能单独证明创新模块优越。
- 新能源第722轮中断的历史根因尚未确定。本次数值修复和定位检查不等于完成5000轮稳定性验证；CPU短运行恢复一致也不保证跨硬件逐位一致。
