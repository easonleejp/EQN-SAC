# 原利润目标的纯 SAC

用户已限定本轮只调整响应方程参数、训练超参、状态空间、新能源与储能参数。活动代码只使用 `profit * reward_scale` 作为奖励；乘法缩放是训练数值尺度。没有利润中心化、削峰/平滑奖励、价格跟随规则或额外调度约束。前一轮目标修改试验已退出推荐，原记录仍在 `experiments/sac_500ep_20261018/`。

设计、文献和结果见[参数审查](parameter-review-20261018.md)。

## 代码与配置

| 文件 | 职责 |
|---|---|
| `train_sac.py`、`sac/runner.py` | 训练、冻结源码、评价和实际输入快照 |
| `sac/agent.py`、`networks.py`、`replay.py` | 原有联合高斯 Actor、双标量 Q、自动温度和回放 |
| `sac/task.py` | 观测与原有 SOC 可行动作映射，原利润奖励 |
| `sac/parameters.py` | 允许范围内的参数覆盖和校验 |
| `configs/sac/baseline_500.json` | 原响应、设备与19维状态的500回合基线 |
| `configs/sac/experimental_market18_2000.json` | 较小储能、18维状态的实验候选；尚不能视为稳定达标方案 |
| `configs/sac/studies/parameter_revision_20261018/` | 本轮各项敏感性配置 |
| `tools/summarize_profit_only_study.py` | 完成记录、源码、输入、奖励与物理核验，开发/最终评价分别汇总 |
| `tools/analyze_price_response.py` | 价格趋势、真实充放电、需求响应和购电曲线分解 |

`response_parameters` 当前开放曲率均值/标准差/下限与需求噪声；`storage_parameters` 开放 SOC、功率、两向效率和吞吐成本；`renewable_penetration` 是生成器的期望日风电能量/名义日负荷能量比例。未给出的值保留原参数。原 TOU、批发价、价格边界、响应参考价格、响应/结算方程不通过该入口改动。

观测可选 `full19`、`causal11`、`market14`、`market18`。14维观测包含标准化当前批发价、总负荷、风电、SOC、剩余时间、时间正余弦、当前已知TOU和三阶价格/风电历史。18维观测另加未来1/3/6小时的已知名义批发价及当前响应参考价；不读取未来实际随机价格。TOU只是信息，不是动作目标或标签。所有模式均通过全日未来实际曲线扰动检查。

## 运行

在仓库根目录使用 Python 3.12；绘图需要 `requirements-analysis.txt`。

```sh
python -m pip install -r requirements-analysis.txt
python -m unittest discover -s tests -v
python train_sac.py train --config configs/sac/baseline_500.json --output runs/sac_baseline_new
python runs/sac_baseline_new/source/train_sac.py evaluate --checkpoint runs/sac_baseline_new/checkpoint_00500.pt --output runs/sac_baseline_new/holdout.json --cases 108 --seed 950000000
python tools/analyze_price_response.py --evaluation runs/sac_baseline_new/holdout.json --environment runs/sac_baseline_new/effective_environment.json --profile runs/sac_baseline_new/effective_profile.json --output checks/sac_baseline_new
```

每次使用新的输出目录。训练保存配置、依赖版本、完整源码哈希、训练 CSV、阶段评价、检查点以及 `effective_environment/profile/generator.json`。评价包含相同实际输入快照及场景档位；分析必须使用该次运行的实际参数，不能套用默认数据。

当前版本是 `sac-2-profit-only`，旧模型用它自己的冻结入口评价。检查点用于评价，CLI没有完整训练恢复功能；重训与加载评价是不同操作。不要修改 manifest 来绕过版本检查。

## 结果图

[训练曲线、价格动作、SOC和负荷变化](../../experiments/sac_parameter_revision_20261018/README.md#展示图)附有PNG和矢量PDF，以及输入/输出来源清单。两个`tools/plot_sac_*.py`脚本只读归档结果。

## 指标解释

价格相关系数逐场景计算，并另报原TOU谷/平/峰均价顺序。低/高负荷小时由原始负荷确定，不能用控制后的曲线重新分类。实际充放量和能量加权买卖价用于判断储能时机；正买卖价差还需扣除效率损失、吞吐成本和无终值的期末库存。

需求变化比较 `user_load` 与原始 `baseline_load`；不是与已经发生响应的TOU参考策略比较。购电曲线依次拆为原负荷扣风电、同价同响应但储能停机、实际储能三个口径。当前响应没有跨时补回机制，所以不把需求削减全部称为负荷转移。

统计上的总体趋势不保证每个随机场景都符合；模拟数据和人为参数也不等于真实客户校准。原始数据单位未恢复，本轮统一采用澳分/kWh、kW、kWh作为明确建模假设，数值不变；不能将其说成已核实的澳大利亚实测收益。
