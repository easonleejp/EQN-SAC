# 原利润目标的纯 SAC

用户已限定本轮只调整响应方程参数、训练超参、状态空间、新能源与储能参数。活动代码只使用 `profit * reward_scale` 作为奖励；乘法缩放是训练数值尺度。没有利润中心化、削峰/平滑奖励、价格跟随规则或额外调度约束。前一轮目标修改试验已退出推荐，原记录仍在 `experiments/sac_500ep_20261018/`。

最新完成午间降价候选的固定日实验：训练500回合，只测试最终模型1回合，无训练内部评价；相对上一轮仅 `hourly_utility` 的24对a、b改变，没有新算法代码。h10–14均价从32.469894降至31.666847澳分/kWh，下降2.473%；但购电峰值上升13.29%，最大相邻小时变化上升102.09%。这是**最新降价候选，不是全面改善的最终方案**。参数、真实结果及限制见[午间降价说明](midday-price.md)和[最新实验记录](../../experiments/sac_midday_price_500_20261018/README.md)。

上一轮[逐时效用说明](hourly-utility.md)与[实验记录](../../experiments/sac_hourly_utility_500_20261018/README.md)继续保留。更早的17维状态、储能参数与训练超参的[固定日改进记录](fixed-day-refinement.md)及[固定日记录](fixed-day-500.md)也保留。文献与参数核对见[参数审查](parameter-review-20261018.md)。

## 代码与配置

| 文件 | 职责 |
|---|---|
| `train_sac.py`、`sac/runner.py` | 训练、冻结源码、评价和实际输入快照 |
| `sac/agent.py`、`networks.py`、`replay.py` | 原有联合高斯 Actor、双标量 Q、自动温度和回放 |
| `sac/task.py` | 观测与原有 SOC 可行动作映射，原利润奖励 |
| `sac/parameters.py` | 允许范围内的参数覆盖和校验 |
| `sac/scenarios.py` | 固定名义日输入；其余模式沿用原随机场景采样 |
| `configs/sac/midday_price_500.json` | 最新降价候选：仅联动24对a、b，固定日500+1；尚非全面改善方案 |
| `configs/sac/hourly_utility_500.json` | 本轮对照：前一逐时效用配置，17维状态和18%峰值功率，固定日500+1 |
| `configs/sac/fixed_day_refined_500.json` | 更早对照：原响应参数、17维状态，18%峰值功率，固定日500+1 |
| `configs/sac/fixed_day_500.json` | 前一次对照：11维状态，20%峰值功率，固定日500+1 |
| `configs/sac/baseline_500.json` | 原响应、设备与19维状态的500回合基线 |
| `configs/sac/experimental_market18_2000.json` | 较小储能、18维状态的实验候选；尚不能视为稳定达标方案 |
| `configs/sac/studies/parameter_revision_20261018/` | 本轮各项敏感性配置 |
| `tools/summarize_profit_only_study.py` | 完成记录、源码、输入、奖励与物理核验，开发/最终评价分别汇总 |
| `tools/analyze_price_response.py` | 价格趋势、真实充放电、需求响应和购电曲线分解 |

`response_parameters` 当前开放曲率均值/标准差/下限、需求噪声和参与噪声；`hourly_utility` 可提供二次效用 `U_h(L)=a_h*L²+b_h*L` 的24对系数，并优先派生逐时曲率与响应参考价，默认空字典保留原路径。`storage_parameters` 开放 SOC、功率、两向效率和吞吐成本；`renewable_penetration` 在随机模式中是期望日风电能量/名义日负荷能量比例，在固定日模式中直接决定恒定风电功率。未给出的值保留原参数。逐时 `a、b` 可以改变数学派生的响应参考价；它不同于TOU账单参照。原TOU、批发价、零售价边界、响应及结算方程均未改变。

观测可选 `full19`、`causal11`、`market14`、`market18`、`nominal11`、`dispatch17`。11维观测包含当前批发价、负荷、风电、SOC、剩余时间、时间正余弦、当前已知TOU、响应参考价、剩余时段名义批发价的最大/最小值。17维在此基础上补充温度、现有可行充放功率界、末小时标志和名义极值到达时间。本轮保持 `dispatch17` 结构，状态与响应共用由实际负荷和效用系数计算的参考价。名义价格来自预先已知的输入模板，不读取未来随机实现值。TOU只是信息，不是动作目标或标签。

## 运行

在仓库根目录使用 Python 3.12；绘图需要 `requirements-analysis.txt`。

```sh
python -m pip install -r requirements-analysis.txt
python -m unittest discover -s tests -v
python train_sac.py train --config configs/sac/midday_price_500.json --output runs/sac_midday_price_new
python runs/sac_midday_price_new/source/train_sac.py evaluate --checkpoint runs/sac_midday_price_new/checkpoint_00500.pt --output runs/sac_midday_price_new/test_1ep.json --cases 1 --seed 20260914
python tools/plot_sac_fixed_day.py --run-dir runs/sac_midday_price_new --output checks/sac_midday_price_new/figures
```

每次使用新的输出目录。当前配置 `evaluation_every=0` 不执行训练前、中途或结束时的内部策略评价；随后单独执行一次 `evaluate --cases 1`。训练500回合共12000环境步，本轮、前一逐时效用及refined配置均为每步8次梯度更新、共92168次更新；最初11维固定日配置为4次/46084次。负荷、批发价和恒定风电逐日相同，三项环境响应噪声均为0；SAC训练探索及回放抽样仍具有随机性。若复现上一轮，请将训练配置换为保留的 `hourly_utility_500.json`，并使用另一个新目录。

上述命令提供从头复现入口。本轮实际没有重跑92项单元测试；核心源码与上轮哈希相同，沿用其92项测试通过及CI绿色的验证记录，本轮另做参数、冻结源码和轨迹物理核验。

训练保存配置、依赖版本、完整源码哈希、训练 CSV、检查点以及 `effective_environment/profile/generator.json`。评价包含相同实际输入快照；分析必须使用该次运行的实际参数，不能套用默认数据。

当前版本是 `sac-3-fixed-day`，旧模型用它自己的冻结入口评价。检查点用于评价，CLI没有完整训练恢复功能；重训与加载评价是不同操作。不要修改 manifest 来绕过版本检查。

## 结果图

[最新午间降价前后六联图](../../experiments/sac_midday_price_500_20261018/README.md)附有PNG、PDF、逐小时CSV与来源记录；[前一逐时效用六联图](../../experiments/sac_hourly_utility_500_20261018/README.md)和[更早固定日六联图](../../experiments/sac_fixed_day_refined_500_20261018/README.md)仍保留。`tools/plot_sac_fixed_day_comparison.py`比较两个保存的单次测试，无需权重或重新执行策略。单组图仍用`tools/plot_sac_fixed_day.py`。只有训练利润叠加20回合尾随均值，测试动作和负荷均不平滑、不跨案例平均。

本轮h11–15负荷加权削减率从12.81995%降至11.63431%，但h12单小时削减率由10.7681%升至14.7512%。购电差分RMS由248.02升至278.56 kW，最大相邻小时变化由449.39升至908.16 kW，利润下降约0.266%；不能概括为全局更平滑。效用系数已改变响应环境，利润差不能解释为同一环境下SAC算法提升。逐时系数属于人工建模调参，未由真实客户数据拟合。

## 指标解释

价格相关系数逐场景计算，并另报原TOU谷/平/峰均价顺序。低/高负荷小时由原始负荷确定，不能用控制后的曲线重新分类。实际充放量和能量加权买卖价用于判断储能时机；正买卖价差还需扣除效率损失、吞吐成本和无终值的期末库存。

需求变化比较 `user_load` 与原始 `baseline_load`；不是与已经发生响应的TOU参考策略比较。购电曲线依次拆为原负荷扣风电、同价同响应但储能停机、实际储能三个口径。当前响应没有跨时补回机制，所以不把需求削减全部称为负荷转移。

统计上的总体趋势不保证每个随机场景都符合；模拟数据和人为参数也不等于真实客户校准。原始数据单位未恢复，本轮统一采用澳分/kWh、kW、kWh作为明确建模假设，数值不变；不能将其说成已核实的澳大利亚实测收益。
