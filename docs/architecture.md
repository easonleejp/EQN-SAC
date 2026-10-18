# 项目结构与代码阅读

本项目研究 24 小时售电定价与储能控制：零售价影响三类用户的用电响应，储能改变购电量，环境计算利润、储能边际收益和满意度。当前实现使用两个 Actor 和两套集成分位 Critic。本文说明现有代码；本次目录整理不调整算法，也不继续推进创新点或新增正式实验。

运行命令见[根目录说明](../README.md)，输入与来源见[数据说明](data.md)。根目录是唯一开发入口；`archive/local/` 中的交接包、冻结代码和审查备份用于本地追溯，不参与导入。

## 模块和执行流程

| 路径 | 职责 |
|---|---|
| [train.py](../train.py) | 命令行入口，提供训练和评价命令 |
| [eqnsac/config.py](../eqnsac/config.py) | 配置定义、字段类型与取值校验 |
| [eqnsac/scenarios.py](../eqnsac/scenarios.py) | 生成负荷、风电和购电价三源场景 |
| [eqnsac/single_source.py](../eqnsac/single_source.py) | 生成只缩放一类外生输入的受控场景 |
| [eqnsac/environment.py](../eqnsac/environment.py) | 状态、用户响应、储能约束、结算和奖励 |
| [eqnsac/networks.py](../eqnsac/networks.py) | Actor、储能动作映射、分位 Critic 和分位损失 |
| [eqnsac/risk.py](../eqnsac/risk.py) | 不确定性诊断与既有 Actor 风险目标 |
| [eqnsac/replay.py](../eqnsac/replay.py) | 环形回放和固定 bootstrap 成员子集 |
| [eqnsac/agent.py](../eqnsac/agent.py) | Critic、Actor、熵温度和目标网络更新 |
| [eqnsac/runner.py](../eqnsac/runner.py) | 训练循环、固定评价、日志、检查点和恢复 |
| [configs](../configs/) / [data](../data/) | 实验配置 / 运行所需 JSON 数据 |
| [tests](../tests/) / [tools](../tools/) | 回归检查 / 显式执行的环境审查与性能测量 |

一次训练交互的顺序是：生成一天的场景 → 重置环境 → Actor 生成联合动作 → 环境执行并结算 → 将转移存入回放 → 满足预热和样本条件后更新网络。每轮包含 24 次小时交互；评价和检查点按配置的轮次间隔执行。

建议先读 `environment.FEATURES` 和 `Environment.step`，理解状态与奖励；再读 `QuantileCritic`、`risk_statistics` 和 `Agent.update`，最后看 `runner.train` 的文件与恢复管理。

## 状态、动作与网络

环境状态是长度为 19 的归一化向量，顺序由 `environment.FEATURES` 定义。下表索引从 0 开始。

| 索引 | 含义 |
|---|---|
| 0–4 | 当前购电价、响应前净负荷、温度、小时、风电 |
| 5 | 储能量在上下限之间的归一化比例 |
| 6–8 | 上一小时三类用户的零售价，**不是三个时间滞后** |
| 9 | 上一小时总响应量 |
| 10–12 | 购电价的前 1、2、3 小时值 |
| 13–15 | 用当前可用信息与名义曲线构造的负荷预测 |
| 16–18 | 风电的前 1、2、3 小时值 |

Price Actor 使用购电价、净负荷、温度和小时共 4 维观测，输出一个共享零售价动作。该动作映射到三类用户各自的价格区间；当前三类价格区间相同。

Storage Actor 默认使用 `compact10`：SOC、小时、购电价、净负荷，以及三阶购电价和风电历史。已有的 `causal15` 配置再加入当前风电、上一响应量和三个负荷预测。其原始 `tanh` 输出映射到当时 SOC 允许的区间，正值表示充电，负值表示放电；原始输出 0 不一定代表停机。

两套 Critic 都读取完整 19 维状态和 2 维联合执行动作，各有 5 个成员，分别学习 Price 与 Storage 的奖励。它们是不同目标的集成，不是标准 SAC 中用于取最小值的两个同目标 Q 网络。每个成员由可训练分支加固定随机 prior 分支组成；每条经验在插入回放时确定其 bootstrap 成员归属。

## 结算与奖励

每小时总利润为：

```text
总利润 = 零售收入 − 购电成本 − 储能吞吐成本
储能边际收益 = 无储能时的购电成本 − 实际购电成本 − 储能吞吐成本
Price 奖励 = 总利润
Storage 奖励 = mix × 总利润 + (1 − mix) × 储能边际收益
```

默认 `mix=0.1`。Storage 奖励也可写为 `0.1 × 无储能零售利润 + 储能边际收益`。网络更新时再统一乘 `reward_scale`；默认值为 0.001。日志中的利润与储能边际收益仍是环境结算量。

训练 Critic 的目标还包含对应 Actor 的熵项，因此分位值描述的是 soft return。`aleatoric_std` 和 `epistemic_std` 不是现金利润标准差，也不能直接解释成纯风电或负荷噪声。实际经济表现应查看评价轨迹的利润及经验下尾收益。

## 方法来源与已有配置

集成分位网络来自 Hoel、Wolff、Laine 的 [EQN 原论文](https://doi.org/10.1109/TITS.2023.3251376)及[作者实现](https://github.com/carljohanhoel/EnsembleQuantileNetworks)。项目保留了分位回归、bootstrap 和固定随机 prior，采用连续随机 Actor、双角色奖励及软 Bellman 目标进行任务迁移；它不是原 EQN 的直接复现。包内[电压控制应用论文](https://doi.org/10.1109/NAPS66256.2025.11272369)采用 EQN-DDPG，也应与当前实现区分。

既有 `arm` 键继续保留：`original` 是本项目基础组，`adaptive_only`、`tail_only`、`full` 对应已有风险目标开关。`original` 不表示普通 SAC，也不表示原论文逐项复现。本次只整理现有代码与资料，不新增或调整这些方法。

原 EQN 的两类标准差与项目尾部分歧已分别命名：`aleatoric_std`、`epistemic_std`、`tail_disagreement`。精确公式、差异及既有验证结果见[算法审查记录](review/algorithm-review-20261018.md)；详细研究讨论保留在该历史记录中，不作为新的实验结论。

## 当前任务边界

时步固定为 1 小时，每天从初始储能量重新开始；日末储能量自由、没有残值奖励。储能不向电网售电，多余风电不会产生销售收入。环境只执行 SOC 和功率物理限制，当前没有经济后备、示范预训练或行为克隆。价格与成本的实际货币单位仍需核准。

现有三组 5000 轮单因素实验使用各自冻结代码；当前源码通过的回归与短运行检查，不等于已经用当前版本重新完成这些训练。
