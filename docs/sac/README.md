# 原利润目标的纯 SAC

用户已限定本轮只调整响应方程参数、训练超参、状态空间、新能源与储能参数。活动代码只使用 `profit * reward_scale` 作为奖励；乘法缩放是训练数值尺度。没有利润中心化、削峰/平滑奖励、价格跟随规则或额外调度约束。前一轮目标修改试验已退出推荐，原记录仍在 `experiments/sac_500ep_20261018/`。

最新完成只用当前与历史信息的13维状态扩展：固定同一天训练500回合，最终模型只测试1回合，无内部评价。相对[调参后6维对照](../../experiments/sac_current6_tuned_500_20261018/README.md)，仅改变观测，保留预热1200步、Actor学习率2e-5、目标熵−2及全部其他参数。18时可用电量48.89→1976.14 kWh，17–20时放电39.95→2547.99 kWh；购电RMS343.69→254.80 kW，峰值3553.36→3314.20 kW，利润增加0.331%。但最大购电跳变710.76→779.30 kW，仍有凌晨充电突增。见[最新实验记录](../../experiments/sac_causal13_500_20261018/README.md)和[13维定义](causal-observation.md)。

前一轮[训练超参调整](current6-training.md)及[最初6维观测实验](../../experiments/sac_current6_500_20261018/README.md)保留；更早的[午间降价说明](midday-price.md)与[实验记录](../../experiments/sac_midday_price_500_20261018/README.md)保留。

上一轮[逐时效用说明](hourly-utility.md)与[实验记录](../../experiments/sac_hourly_utility_500_20261018/README.md)继续保留。更早的17维状态、储能参数与训练超参的[固定日改进记录](fixed-day-refinement.md)及[固定日记录](fixed-day-500.md)也保留。文献与参数核对见[参数审查](parameter-review-20261018.md)。

## 代码与配置

| 文件 | 职责 |
|---|---|
| `train_sac.py`、`sac/runner.py` | 训练、冻结源码、评价和实际输入快照 |
| `sac/agent.py`、`networks.py`、`replay.py` | 原有联合高斯 Actor、双标量 Q、自动温度和回放 |
| `sac/task.py` | 观测与原有 SOC 可行动作映射，原利润奖励 |
| `sac/parameters.py` | 允许范围内的参数覆盖和校验 |
| `sac/scenarios.py` | 固定名义日输入；其余模式沿用原随机场景采样 |
| `configs/sac/causal13_500.json` | 最新13维状态扩展，固定日500+1；仅当前与历史信息 |
| `configs/sac/current6_tuned_500.json` | 本轮6维对照，预热1200、Actor学习率2e-5、目标熵−2 |
| `configs/sac/current6_500.json` | 更早6维对照，固定日500+1；不输入未来价格及响应模型参数 |
| `configs/sac/midday_price_500.json` | 更早17维对照：午间降价候选，固定日500+1 |
| `configs/sac/hourly_utility_500.json` | 更早对照：前一逐时效用配置，17维状态和18%峰值功率，固定日500+1 |
| `configs/sac/fixed_day_refined_500.json` | 更早对照：原响应参数、17维状态，18%峰值功率，固定日500+1 |
| `configs/sac/fixed_day_500.json` | 前一次对照：11维状态，20%峰值功率，固定日500+1 |
| `configs/sac/baseline_500.json` | 原响应、设备与19维状态的500回合基线 |
| `configs/sac/experimental_market18_2000.json` | 较小储能、18维状态的实验候选；尚不能视为稳定达标方案 |
| `configs/sac/studies/parameter_revision_20261018/` | 本轮各项敏感性配置 |
| `tools/summarize_profit_only_study.py` | 完成记录、源码、输入、奖励与物理核验，开发/最终评价分别汇总 |
| `tools/analyze_price_response.py` | 价格趋势、真实充放电、需求响应和购电曲线分解 |

`response_parameters` 当前开放曲率均值/标准差/下限、需求噪声和参与噪声；`hourly_utility` 可提供二次效用 `U_h(L)=a_h*L²+b_h*L` 的24对系数，并优先派生逐时曲率与响应参考价，默认空字典保留原路径。`storage_parameters` 开放 SOC、功率、两向效率和吞吐成本；`renewable_penetration` 在随机模式中是期望日风电能量/名义日负荷能量比例，在固定日模式中直接决定恒定风电功率。未给出的值保留原参数。逐时 `a、b` 可以改变数学派生的响应参考价；它不同于TOU账单参照。原TOU、批发价、零售价边界、响应及结算方程均未改变。

当前 `causal13` 保留 `current6` 的当前批发价、原始负荷、风电、SOC、剩余时间、当前TOU，新增日内正余弦、前三小时实测批发价及当前SOC的充放电可行上下界。批发价按 `(price-20)/2`、负荷和风电按 `/5000` 归一化；SOC和TOU使用固定物理/价格边界。历史不足时重复第0小时已知报价，不跨日回绕。固定常数在训练前冻结，不计算整日统计。响应参考价、曲率、温度和所有未来报价特征均不进入网络。上下界只暴露已有物理，不改变约束或动作映射；新增特征主要帮助函数拟合，不能直接提供晚峰计划。

旧 `full19`、`causal11`、`market14`、`market18`、`nominal11`、`dispatch17` 模式只保留供历史复现。其中部分模式使用未来名义报价；按本轮要求，它们不满足当前信息边界，不能因名义模板已存于文件就视作实时可知。TOU只是当前输入，不是动作目标或约束。

## 运行

在仓库根目录使用 Python 3.12；绘图需要 `requirements-analysis.txt`。

```sh
python -m pip install -r requirements-analysis.txt
python -m unittest discover -s tests -v
python train_sac.py train --config configs/sac/causal13_500.json --output runs/sac_causal13_new
python runs/sac_causal13_new/source/train_sac.py evaluate --checkpoint runs/sac_causal13_new/checkpoint_00500.pt --output runs/sac_causal13_new/test_1ep.json --cases 1 --seed 20260914
python tools/plot_sac_fixed_day.py --run-dir runs/sac_causal13_new --output checks/sac_causal13_new/figures
```

每次使用新的输出目录。当前配置 `evaluation_every=0` 不执行训练前、中途或结束时的内部策略评价；随后单独执行一次 `evaluate --cases 1`。500回合共12000环境步，预热1200步，每步8次更新，前后均实际86408次更新。负荷、批发价和恒定风电逐日相同，三项环境响应噪声均为0；SAC训练探索与回放仍有随机性。复现本轮对照时换用 `current6_tuned_500.json` 和新输出目录。

本轮103项单元测试通过，含未来数据扰动、响应特征隔离、历史索引边界及相同动作下奖励与物理一致性。配置严格差异、500+1预算、冻结源码、配对数据、奖励恒等式和轨迹能量平衡核验也通过。

训练保存配置、依赖版本、完整源码哈希、训练 CSV、检查点以及 `effective_environment/profile/generator.json`。评价包含相同实际输入快照；分析必须使用该次运行的实际参数，不能套用默认数据。

当前版本是 `sac-3-fixed-day`，旧模型用它自己的冻结入口评价。检查点用于评价，CLI没有完整训练恢复功能；重训与加载评价是不同操作。不要修改 manifest 来绕过版本检查。

## 结果图

[最新13维状态前后六联图](../../experiments/sac_causal13_500_20261018/README.md)附有PNG、PDF、逐小时CSV与来源记录；[前一轮训练超参比较](../../experiments/sac_current6_tuned_500_20261018/README.md)及其他历史结果保留。`tools/plot_sac_fixed_day_comparison.py`比较两个保存的单次测试，无需权重或重新执行策略。单组图仍用`tools/plot_sac_fixed_day.py`。只有训练利润叠加20回合尾随均值，测试动作和负荷均不平滑、不跨案例平均。

本轮h11–15削减率13.19%→11.15%，h10–14均价31.8920→31.5793。全天放电中h8–15占比96.72%→11.73%，h17–20占比1.85%→69.92%；储能由午间提前放空转为更多支撑晚峰。仍有h2→3的充电启动跳变，且低负荷六小时的用户需求整体仍略减，填谷主要来自储能。新增7维同时变化、输入尺寸影响初始化，单seed结果不能分离各特征贡献或证明泛化。

## 指标解释

价格相关系数逐场景计算，并另报原TOU谷/平/峰均价顺序。低/高负荷小时由原始负荷确定，不能用控制后的曲线重新分类。实际充放量和能量加权买卖价用于判断储能时机；正买卖价差还需扣除效率损失、吞吐成本和无终值的期末库存。

需求变化比较 `user_load` 与原始 `baseline_load`；不是与已经发生响应的TOU参考策略比较。购电曲线依次拆为原负荷扣风电、同价同响应但储能停机、实际储能三个口径。当前响应没有跨时补回机制，所以不把需求削减全部称为负荷转移。

统计上的总体趋势不保证每个随机场景都符合；模拟数据和人为参数也不等于真实客户校准。原始数据单位未恢复，本轮统一采用澳分/kWh、kW、kWh作为明确建模假设，数值不变；不能将其说成已核实的澳大利亚实测收益。
