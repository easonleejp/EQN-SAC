# 当前纯SAC：causal13

当前工作树的SAC只保留 [`configs/sac/causal13_500.json`](../../configs/sac/causal13_500.json) 和[当前实验档案](../../experiments/sac_causal13_500_20261018/README.md)。旧SAC版本的配置、独立实验和说明已移除，Git历史保留来源。本次整理不改变当前训练代码或结果，也不重新训练、评价；原EQN与main分支不在删除范围内。

## 模型、状态和奖励

当前为一个联合Gaussian Actor和两个标量Q Critic，价格、储能共两维原始动作；不是两个Actor或EQN集成。Actor、回放和Critic均使用13维状态，每个Critic另接2维联合动作。储能原始动作按当前SOC的可行区间映射，原始动作0并不总代表实际功率为0。

13维输入包括当前批发价、原始总负荷、风电、可用SOC比例、剩余时间、当前TOU、日内正余弦、过去1/2/3小时批发价和当前充放电可行边界。价格按 `(price-20)/2`，负荷与风电按 `/5000` 缩放；常数固定，不计算全日统计。日初历史不足时重复第0小时已知价格。没有未来报价、预测曲线、响应参考价、曲率或温度因子输入，详见[状态定义](causal-observation.md)。

环境奖励只有 `profit * 0.001`；SAC熵项属于学习算法，不是新增环境利润。没有利润中心化、削峰、平滑奖励、价格跟随规则或额外调度约束。当前TOU是输入和账单参照，不是动作目标。

逐时二次效用 `U_h(L)=a_h*L²+b_h*L` 的24对系数保存在配置中，环境使用 `kappa_h=-a_h` 和 `reference_h=b_h+2*a_h*B_h`，三类用户按原份额处理。原TOU、批发价、零售价边界及响应、结算公式保持原实现。a、b属于人工建模参数，不是实测客户拟合。纯文献证据保留在[文献资料](literature/)，不作为其他SAC版本入口。

## 当前协议与文件

| 项目 | 设置 |
|---|---|
| 数据与种子 | 同一个固定24小时日，seed 42，CPU单线程 |
| 环境随机性 | 曲率、需求、参与三项噪声标准差均为0 |
| 训练 | 500回合，12000步，预热1200步，每步更新8次，共86408次更新 |
| 网络与批次 | 隐藏层128，batch 256 |
| 学习率 | Actor `2e-5`，Critic `5e-4`，alpha `3e-4` |
| 其他超参 | 初始alpha `0.05`，目标熵−2，gamma 1，target tau `0.01` |
| 评价 | `evaluation_every=0`；最终检查点唯一1回合、24步测试 |

目标熵−2是二维联合动作的启发式，与13维观测无关。环境无噪声不代表训练无随机性，探索、初始化与回放抽样仍使用随机过程。已保存训练耗时179.238秒，实际记录见[完成文件](../../experiments/sac_causal13_500_20261018/run/complete.json)。

| 文件 | 用途 |
|---|---|
| `train_sac.py`、`sac/runner.py` | 活动训练入口、源码冻结与评价 |
| `sac/agent.py`、`networks.py`、`replay.py` | 联合Actor、双标量Q、自动温度及回放 |
| `sac/task.py` | 当前观测、原动作映射与利润奖励 |
| `sac/parameters.py`、`scenarios.py` | 参数校验与固定日输入 |
| `experiments/sac_causal13_500_20261018/run/` | 配置、训练CSV、单次测试、检查点、有效参数及来源记录 |
| `experiments/sac_causal13_500_20261018/run/source/` | 本版实验实际使用的冻结源码和数据 |

## 独立复现

Python 3.12环境可安装 `requirements-analysis.txt`。查看已有图表与数据无需训练或评价；下列命令仅供需要独立复现时使用，不是本次目录整理执行的操作。请从仓库根目录运行，并选用尚不存在的输出目录。

```sh
python experiments/sac_causal13_500_20261018/run/source/train_sac.py train \
  --config configs/sac/causal13_500.json \
  --output runs/sac_causal13_500_reproduce

python runs/sac_causal13_500_reproduce/source/train_sac.py evaluate \
  --checkpoint runs/sac_causal13_500_reproduce/checkpoint_00500.pt \
  --output runs/sac_causal13_500_reproduce/test_1ep.json \
  --cases 1 --seed 20260914
```

[本版检查点](../../experiments/sac_causal13_500_20261018/run/checkpoint_00500.pt)与冻结源码配套保存。检查点加载器会核验源码和数据；不要修改manifest绕过检查。CLI支持重新训练和加载评价，不提供完整训练恢复功能。

只重画当前已保存的单次结果，不加载策略：

```sh
python tools/plot_sac_fixed_day.py \
  --run-dir experiments/sac_causal13_500_20261018/run \
  --output checks/sac_causal13_figures
```

## 当前结果与适用范围

当前[单版本结果图](../../experiments/sac_causal13_500_20261018/figures/fixed_day_results.png)及[完整报告](../../experiments/sac_causal13_500_20261018/README.md)对应一个真实24小时测试。日利润原值1,169,753.64，购电峰值3314.20 kW、相邻变化RMS 254.80 kW；h18开始有1976.14 kWh可用储能，h17–20放电2547.99 kWh。h10–14零售均价31.5793澳分/kWh，h11–15加权削减率11.1459%。h2→3仍有779.30 kW的最大购电跳变，不能称为全面平滑。

测试动作和负荷不平滑、不跨案例平均；仅训练利润叠加20回合尾随均值。当前响应没有跨时能量补回机制，需求削减不等于等量负荷转移。初始SOC在下界，日末无残值。

本次训练版本的[103项测试记录](../../experiments/sac_causal13_500_20261018/checks/unit_tests.log)随档案保存；文件清理后另通过[103项回归检查](../../experiments/sac_causal13_500_20261018/checks/cleanup_unit_tests.log)。当前训练结果和唯一测试数据未改，也没有因整理文件而重新进行500回合训练或最终策略评价。

单seed固定日反复训练可以学习日程，不能证明未见日期泛化或严格收敛。数值采用澳分/kWh、kW、kWh的建模假设，利润原值按澳分理解；原始数据单位尚未核实，不应表述为已经验证的真实市场收益。
