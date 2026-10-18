# EQN-SAC 精简训练入口

这是2026-10-08整理的新训练版本。入口只有 `train.py`，没有经济后备、风险接管、P4、LP动作替换、优化器示范预训练、行为克隆、批量搜索或Word/绘图依赖。SOC与充放电功率限制属于物理环境，继续保留。

## 运行

在本目录使用原有 Python 3.12 / PyTorch 2.7.1 环境。无需重新安装已满足的依赖。

```powershell
python -m unittest discover -s tests -v
python train.py train --config configs/baseline.json --arm original --output runs/original_seed42
python train.py train --config configs/baseline.json --arm original --output runs/original_seed42 --resume runs/original_seed42/checkpoint.pt --episodes 6000
python train.py evaluate --checkpoint runs/original_seed42/checkpoint.pt --output evaluation.json --cases 270
```

以上训练命令默认5000轮，不会自动执行；本次仅运行了短时功能检查，没有启动正式训练。每个命令只训练一个模型。其他arm为 `adaptive_only`、`tail_only`、`full`。改变seed请复制JSON并显式修改；四组比较应使用同一seed与相同配置，只有arm不同。

## 文件职责

| 文件 | 职责 |
|---|---|
| eqnsac/config.py | 参数校验，拒绝拼写错误和旧参数 |
| eqnsac/scenarios.py | 三源场景生成；保留原生成公式 |
| eqnsac/environment.py | 19维状态、用户响应、SOC和经济结算 |
| eqnsac/networks.py | Actor、可行域映射、IQN、风险计算 |
| eqnsac/replay.py | 固定bootstrap归属和回放 |
| eqnsac/agent.py | 两Actor、两套集中式Critic集成的更新 |
| eqnsac/runner.py | 单模型训练、检查点和独立评价 |
| configs/baseline.json | 清理后的基线设置 |
| configs/storage_candidate.json | 待验证的Storage候选设置 |
| data | 导出的名义24小时输入、物理参数与来源SHA |
| tests | 新旧环境轨迹、会计、动作边界、梯度与更新检查 |
| tools | 环境审查和短时性能测量；输出到checks |

## 与旧5000轮版本的关系

新旧环境的固定动作轨迹已核对，但新训练不是历史训练的逐位复现。两Actor与两套5成员Critic、prior300、原三源生成器、19维Critic输入和原经济定义保留。保留原Actor层数与分位损失求和尺度。

明确变化包括：移除所有后备与优化器示范训练；修复冻结prior的动作梯度；四组Actor统一中点分位网格；分开Actor与tau随机流；独立实现回放、状态保存和训练入口；采用数值稳定的tanh对数概率计算；日志在episode边界同步。历史权重不兼容新的键名，加载时应报错，不提供静默迁移。

Adaptive采用明确的停止梯度风险水平和硬分位选择，不宣称完整连续风险梯度。Tail称为尾部分位分歧代理，不宣称未校准的精确Wasserstein距离。SAC软目标包含相应Actor熵，Q不是现金利润。

`baseline.json`保留Storage的compact10观测、1e-3学习率、200步warmup、0.1混合奖励以及固定响应随机流，便于识别原配置。它不是推荐的最终论文配置。

`storage_candidate.json`只作为开发候选：Storage扩充到15个已有因果特征，学习率降至3e-4，warmup增至2000步，响应噪声按场景种子改变。它同时改变四项，不能用一次对比把结果归因于其中某一项，也尚未证明更好。正式消融应从baseline逐项修改，不开展笛卡尔积搜索。

## 数据与评价边界

数据是从冻结环境导出的原名义日输入，来源见 `data/provenance.json`；不是新采集的真实多日数据。加载训练代码不需要原E盘数据或任何旧Python模块。

环境限定为24小时、时步1小时、自放电系数1；不支持通过修改JSON直接扩展到其他时步。终端SOC自由且无残值；储能不向电网卖电；多余风电弃用。当前预计状态中的负荷预测只读取当前观测与已知名义曲线，不读取未来真实实现。

运行评价使用独立种子和27组合轮转；只有案例数是27倍数时才是等组合权重。它是开发监测，不自动代表真实运营分布或最终未见测试。正式测试集、分布和服务要求需在实验前锁定。

检查点保存网络、优化器、回放及随机流，支持同版本episode边界恢复；CPU短运行验证连续运行与断点恢复参数逐位一致。不同硬件/版本的位级可重复性未保证。每次训练输出配置、源码与数据SHA，拒绝覆盖已有新训练目录。

## 本次验证范围

固定动作环境核对72小时，涵盖low/medium/high；9项单元检查；3轮49次更新的短时训练、独立评价及断点恢复；完整256宽度、128批量、32分位和2×5成员的CPU/GPU短测。上述不构成收敛、经济优越性或最终速度提升证明。
