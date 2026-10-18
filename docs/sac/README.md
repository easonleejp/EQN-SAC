# 原利润目标的纯 SAC

用户已限定本轮只调整响应方程参数、训练超参、状态空间、新能源与储能参数。活动代码只使用 `profit * reward_scale` 作为奖励；乘法缩放是训练数值尺度。没有利润中心化、削峰/平滑奖励、价格跟随规则或额外调度约束。前一轮目标修改试验已退出推荐，原记录仍在 `experiments/sac_500ep_20261018/`。

最新完成第二次固定同一天的500回合训练和1回合最终测试，重点改进23时电价及储能跳变，见[前后配置与结果](fixed-day-refinement.md)。前一次[固定日记录](fixed-day-500.md)仍保留。文献与参数核对见[参数审查](parameter-review-20261018.md)。

## 代码与配置

| 文件 | 职责 |
|---|---|
| `train_sac.py`、`sac/runner.py` | 训练、冻结源码、评价和实际输入快照 |
| `sac/agent.py`、`networks.py`、`replay.py` | 原有联合高斯 Actor、双标量 Q、自动温度和回放 |
| `sac/task.py` | 观测与原有 SOC 可行动作映射，原利润奖励 |
| `sac/parameters.py` | 允许范围内的参数覆盖和校验 |
| `sac/scenarios.py` | 固定名义日输入；其余模式沿用原随机场景采样 |
| `configs/sac/fixed_day_refined_500.json` | 最新：17维状态，18%峰值功率，固定日500+1 |
| `configs/sac/fixed_day_500.json` | 前一次对照：11维状态，20%峰值功率，固定日500+1 |
| `configs/sac/baseline_500.json` | 原响应、设备与19维状态的500回合基线 |
| `configs/sac/experimental_market18_2000.json` | 较小储能、18维状态的实验候选；尚不能视为稳定达标方案 |
| `configs/sac/studies/parameter_revision_20261018/` | 本轮各项敏感性配置 |
| `tools/summarize_profit_only_study.py` | 完成记录、源码、输入、奖励与物理核验，开发/最终评价分别汇总 |
| `tools/analyze_price_response.py` | 价格趋势、真实充放电、需求响应和购电曲线分解 |

`response_parameters` 当前开放曲率均值/标准差/下限、需求噪声和参与噪声；`storage_parameters` 开放 SOC、功率、两向效率和吞吐成本；`renewable_penetration` 在随机模式中是期望日风电能量/名义日负荷能量比例，在固定日模式中直接决定恒定风电功率。未给出的值保留原参数。原 TOU、批发价、价格边界、响应参考价格、响应/结算方程不通过该入口改动。

观测可选 `full19`、`causal11`、`market14`、`market18`、`nominal11`、`dispatch17`。11维观测包含当前批发价、负荷、风电、SOC、剩余时间、时间正余弦、当前已知TOU、响应参考价、剩余时段名义批发价的最大/最小值。最新17维在此基础上补充温度、现有可行充放功率界、末小时标志和名义极值到达时间。名义价格来自公开输入模板，不读取未来随机实现值。TOU只是信息，不是动作目标或标签。

## 运行

在仓库根目录使用 Python 3.12；绘图需要 `requirements-analysis.txt`。

```sh
python -m pip install -r requirements-analysis.txt
python -m unittest discover -s tests -v
python train_sac.py train --config configs/sac/fixed_day_refined_500.json --output runs/sac_fixed_day_new
python runs/sac_fixed_day_new/source/train_sac.py evaluate --checkpoint runs/sac_fixed_day_new/checkpoint_00500.pt --output runs/sac_fixed_day_new/test_1ep.json --cases 1 --seed 20260914
python tools/plot_sac_fixed_day.py --run-dir runs/sac_fixed_day_new --output checks/sac_fixed_day_new/figures
```

每次使用新的输出目录。当前配置 `evaluation_every=0` 不执行训练前、中途或结束时的策略评价；第二条命令仅测试最终模型一次。训练500回合共12000环境步，最新配置每步8次梯度更新，共92168次更新；前一次为4次/46084次。训练保存配置、依赖版本、完整源码哈希、训练 CSV、检查点以及 `effective_environment/profile/generator.json`。评价包含相同实际输入快照；分析必须使用该次运行的实际参数，不能套用默认数据。

当前版本是 `sac-3-fixed-day`，旧模型用它自己的冻结入口评价。检查点用于评价，CLI没有完整训练恢复功能；重训与加载评价是不同操作。不要修改 manifest 来绕过版本检查。

## 结果图

[最新前后六联图](../../experiments/sac_fixed_day_refined_500_20261018/README.md)附有PNG、PDF、逐小时CSV与来源清单。`tools/plot_sac_fixed_day_comparison.py`比较两个保存的单次测试，无需权重或重新执行策略。单组图仍用`tools/plot_sac_fixed_day.py`。只有训练利润叠加20回合尾随均值，测试动作和负荷均不平滑。

## 指标解释

价格相关系数逐场景计算，并另报原TOU谷/平/峰均价顺序。低/高负荷小时由原始负荷确定，不能用控制后的曲线重新分类。实际充放量和能量加权买卖价用于判断储能时机；正买卖价差还需扣除效率损失、吞吐成本和无终值的期末库存。

需求变化比较 `user_load` 与原始 `baseline_load`；不是与已经发生响应的TOU参考策略比较。购电曲线依次拆为原负荷扣风电、同价同响应但储能停机、实际储能三个口径。当前响应没有跨时补回机制，所以不把需求削减全部称为负荷转移。

统计上的总体趋势不保证每个随机场景都符合；模拟数据和人为参数也不等于真实客户校准。原始数据单位未恢复，本轮统一采用澳分/kWh、kW、kWh作为明确建模假设，数值不变；不能将其说成已核实的澳大利亚实测收益。
