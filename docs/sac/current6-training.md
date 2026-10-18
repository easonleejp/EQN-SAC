# current6的训练超参调整：固定日500+1

本轮保留[当前6维状态及归一化](current-observation.md)，只调整训练超参。配置为 [`current6_tuned_500.json`](../../configs/sac/current6_tuned_500.json)，结果由[本轮实验记录](../../experiments/sac_current6_tuned_500_20261018/README.md)汇总；本文不预判调参效果。

| 参数 | 上一轮current6 | 本轮 | 用意 |
|---|---:|---:|---|
| 预热步数 `warmup_steps` | 480 | 1200 | 增加均匀随机探索，延后首次更新 |
| Actor学习率 | `5e-5` | `2e-5` | 减小策略更新步长 |
| 目标熵 `target_entropy` | −4 | −2 | 尝试二维动作的负维数启发式 |

Actor输出价格、储能两维动作，代码将两个坐标的log概率相加。目标熵−2对应 `−动作维数` 的经验选择，并非强制规则或最优值；观测是6维，不意味着目标熵应为−6，也不要求每个动作维度分别达到−1。

环境奖励仍是 `0.001×原利润`，没有新增削峰、平滑奖励或约束。不过目标熵会影响自适应alpha，进而改变SAC的软价值目标、策略更新和探索，不能把本轮说成“学习目标完全未变”。

逐时a、b、响应公式、动作映射、设备、固定日数据、seed 42及其余超参不变。两轮冻结清单中的17项源码和基础数据哈希完全相同。三项环境噪声均为0；训练探索与回放仍有随机性。

预算仍为500回合、12000环境步，每步更新8次。代码在达到预热门槛的当步开始更新，因此更新数由 `(12000−480+1)×8=92168` 降为 `(12000−1200+1)×8=86408`。不为补齐更新数而增加每步更新或回合数。内部评价关闭，最终检查点只测试1回合、24步。

## 复现与离线比较

从仓库根目录执行，使用本轮保存的配置和冻结源码；复现输出目录须尚不存在。已有本轮结果无需重跑。

```bash
.venv/bin/python runs/sac_current6_tuned_500/source/train_sac.py train \
  --config runs/sac_current6_tuned_500/config.json \
  --output runs/sac_current6_tuned_500_reproduce

.venv/bin/python runs/sac_current6_tuned_500_reproduce/source/train_sac.py evaluate \
  --checkpoint runs/sac_current6_tuned_500_reproduce/checkpoint_00500.pt \
  --output runs/sac_current6_tuned_500_reproduce/test_1ep.json \
  --cases 1 --seed 20260914
```

以下只读取两条已保存轨迹，不加载策略、不补测。a、b未变，比较无需 `--hourly-utility` 参数。

```bash
.venv/bin/python tools/plot_sac_fixed_day_comparison.py \
  --before experiments/sac_current6_500_20261018/run \
  --after runs/sac_current6_tuned_500_reproduce \
  --output checks/current6_tuned_reproduce_figures
```

三项设置同时改变，单seed不能区分各项贡献，也不能证明普遍更优。固定同日重复训练和单次测试仍不构成未见场景泛化证据。
