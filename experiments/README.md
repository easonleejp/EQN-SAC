# 实验记录

当前：[固定同一天训练500回合、测试1回合](sac_fixed_day_500_20261018/README.md)。环境扰动全部关闭，原利润目标不变，附收敛、价格、储能、SOC和负荷图。

历史：[原利润纯SAC随机场景参数试验](sac_parameter_revision_20261018/README.md)。保留未达标结果，不与当前固定日500+1混用。

历史：[五组500回合SAC试验](sac_500ep_20261018/README.md)。其中修改奖励的方案已撤回推荐，冻结记录保留用于追溯，不能混作当前原利润版本的结果。

## 历史 EQN 实验

这里保存三组已经完成的 5000 轮单因素实验，以及 2026-10-18 的历史模型 CPU 复查摘要。实验原始记录与冻结源码按字节复制，没有重新训练，也没有用当前开发代码替换历史实现。

当前开发入口在仓库根目录的 `train.py` 和 `eqnsac/`。历史实验的检查点版本为 `clean-1`，当前开发版本为 `clean-2`；历史结果不能直接作为修复后代码的训练结果。当前代码的修改、测试和审查说明请从[项目说明](../README.md)进入。

## 内容导航

| 实验 | 整日扰动 | 原始记录 |
|---|---|---|
| 负荷 | 名义负荷乘以 `1 + U(-0.05, 0.05)` | [20261010_original_load_scale_5000ep](historical/20261010_original_load_scale_5000ep/) |
| 风电 | 固定参考风电乘以 `1 + U(-0.05, 0.05)` | [20261011_original_wind_scale_5000ep](historical/20261011_original_wind_scale_5000ep/) |
| 购电价 | 名义购电价乘以 `1 + U(-0.03, 0.03)` | [20261011_original_price_scale_5000ep](historical/20261011_original_price_scale_5000ep/) |

每组目录保留了原始文件结构：

- `frozen_code/`：该实验实际使用的源码、配置、输入 JSON 和测试。三组版本存在差异，应分别读取。
- `run/`：配置、来源 manifest、参考场景、5000 轮训练 CSV、全部阶段评价 JSON 和完成记录。
- `figures*/`：原有图表及用于绘图的指标 CSV。
- 启动与训练日志；风电实验还保留 `before_resume_01/training.csv` 和恢复日志。

冻结源码和日志中的历史路径、说明及 `run/manifest.json` 均保持原文。因此其中出现的旧 Windows 路径、旧入口或原运行环境，是历史来源记录，不是当前仓库的使用说明。

汇总入口：

- [三组最终结果与平台](summary/三组最终结果与平台.csv)：原有的最终结果和平台判断摘要。
- [历史模型 CPU 复查汇总](summary/复查汇总.csv)：各组最终模型固定 100 案例的原结果、CPU 复查值和差值。
- [负荷逐案例差异](summary/20261010_original_load_scale_5000ep_case_differences.csv)、[风电逐案例差异](summary/20261011_original_wind_scale_5000ep_case_differences.csv)、[购电价逐案例差异](summary/20261011_original_price_scale_5000ep_case_differences.csv)：各 100 行，差值方向为 CPU 复查值减原值。

CPU 复查使用每组自己的冻结源码和最终检查点，没有训练。原记录来自 Windows/CUDA，复查来自 Mac/CPU，不保证逐位一致。这 100 个固定案例用于历史结果复查，不是新增的独立最终测试集。

## 检查点与来源核验

模型文件 `.pt` 保留在本机归档，没有复制到此 Git 目录。最终检查点的本地路径模式为：

```text
archive/local/EQN-SAC_项目交接_20261018/05_近三次实验完整记录/<实验目录名>/run/checkpoint.pt
```

风电实验的中途检查点位于同组的 `before_resume_01/checkpoint.pt`。`archive/local/` 是本地归档路径，不是线上下载链接；只克隆此仓库不会自动取得模型。需要重新评价历史模型时，应配合对应的 `frozen_code` 使用，并先核对检查点 SHA256。

[source_manifest.json](source_manifest.json) 记录了每个复制文件的源路径、仓库内目标路径、字节数和 SHA256；还记录了四个未复制检查点的大小与 SHA256。所有复制文件已逐一验证与本地来源完全一致。三个实验共复制 274 个文件，摘要复制 5 个 CSV，总计 **279 个文件、19,165,275 字节（约 18.28 MiB）**，不含本说明和来源清单本身。

复制源目录为 `archive/local/EQN-SAC_项目交接_20261018/05_近三次实验完整记录`。原结果摘要来自交接包的 `06_结果图与审查报告`，CPU 复查 CSV 来自 `archive/local/代码审查_20261018/历史模型复查`。只排除了模型 `.pt`、`__pycache__` 和 `.DS_Store`，没有改写冻结源码或 `run` 文件。

## 简单读取

在仓库根目录运行以下命令，无需加载模型即可读取三个最终评价摘要：

```bash
python - <<'PY'
import json
from pathlib import Path

for path in sorted(Path("experiments/historical").glob("*/run/evaluation_05000.json")):
    result = json.loads(path.read_text(encoding="utf-8"))
    print(path.parent.parent.name)
    print({key: result[key] for key in ("n", "mean_profit", "cvar10", "mean_storage_net")})
PY
```

CSV 使用 UTF-8，可用表格软件打开；通过 Python 读取带 BOM 的摘要时可指定 `encoding="utf-8-sig"`。`training.csv` 是训练过程记录，`evaluation_*.json` 是固定案例的阶段评价，二者统计口径应分别解释。

核验仓库中的复制文件：

```bash
python - <<'PY'
import hashlib
import json
from pathlib import Path

manifest = json.loads(Path("experiments/source_manifest.json").read_text(encoding="utf-8"))
for item in manifest["files"]:
    path = Path(item["destination"])
    assert path.stat().st_size == item["bytes"], path
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], path
print(f"Verified {len(manifest['files'])} copied files")
PY
```
