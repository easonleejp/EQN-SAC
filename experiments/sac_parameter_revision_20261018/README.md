# 原利润纯 SAC：参数、状态和训练复核

本轮严格保留原响应公式、原利润目标、原价格边界和原动作执行规则，只实验响应参数、SAC超参数、状态表达及新能源/储能参数。此前修改奖励的方案已撤回推荐。

完成16次训练、共23000回合（6次500回合，10次2000回合），并对基线及固定候选做新108场景评价。**候选未全面达标，不替换原基线**：部分储能价差改善，但TOU层次与购电峰值没有稳定改善，多训练seed下仍有储能亏损。完整参数含义、期刊核对、单位假设及结果解释见[审查报告](../../docs/sac/parameter-review-20261018.md)。

## 展示图

- [训练与开发场景曲线](figures/convergence.png) · [矢量PDF](figures/convergence.pdf)
- [价格、储能动作与SOC](figures/price_storage_soc.png) · [矢量PDF](figures/price_storage_soc.pdf)
- [用户负荷与购电变化](figures/load_changes.png) · [矢量PDF](figures/load_changes.pdf)

展示图另有[输入、脚本及输出哈希清单](figures/manifest.json)。重绘入口是`tools/plot_sac_convergence.py`和`tools/plot_sac_dispatch.py`；后者默认使用macOS的PingFang SC，可通过`--font`指定其他已安装中文字体。

## 先看这些文件

| 内容 | 入口 |
|---|---|
| 开发与最终评价分开的完整汇总 | [study_summary/summary.json](study_summary/summary.json) |
| 16组开发结果 | [development.csv](study_summary/development.csv) |
| 4个2000回合检查点的最终评价 | [holdout.csv](study_summary/holdout.csv) |
| 两组500回合检查点的最终评价摘要 | [final_500_checkpoints.json](analysis/final_500_checkpoints.json) |
| 查看最终场景前的候选固定记录 | [final_selection.json](final_selection.json) |
| 6次评价的检查点、实际回合及SHA | [final_evaluation_plan.json](analysis/final_evaluation_plan.json) |
| 响应幅度与购电峰值会计分解 | [final_response_effects.json](analysis/final_response_effects.json) |
| 候选逐时图与数值 | [图](analysis/final_candidate_analysis/price_response.png)、[JSON](analysis/final_candidate_analysis/price_response.json) |
| 原基线逐时图与数值 | [图](analysis/final_baseline_analysis/price_response.png)、[JSON](analysis/final_baseline_analysis/price_response.json) |
| 文献参数依据 | [parameter_evidence.json](literature/parameter_evidence.json)、[新能源20%补充证据](literature/renewable_share_supplement.json) |
| 82项测试日志 | [final_tests.log](analysis/final_tests.log) |
| 全部归档文件来源与哈希 | [archive_manifest.json](archive_manifest.json) |

表中的利润原始单位未变；研究报告另按“电价为澳分/kWh、功率kW、能量kWh”的工程假设转换成AUD。原始数据的澳洲来源和单位尚未恢复，这不是已证实的历史实测经济收益。

最终评价固定seed950000000起的108个新场景，27种三源档位组合各4个；开发seed910000000起的27个场景曾用于选择。最终评价后未据此继续调参。各图展示跨场景均值，正变化表示实际量大于基线；JSON中的`response_reduction`字段仍按“削减为正”的原始定义，不能混用符号。跨场景充放曲线同一小时都非零，不表示单场景同时充放电。平均日峰值是逐场景取峰后平均，不是均值曲线取峰。

## 归档范围与结构

每组位于`runs/sac_revision_<试验名>/`：

- `config.json`、`effective_environment/profile/generator.json`：计划配置与实际生效输入。
- `training.csv`、`complete.json`、`manifest.json`：训练日记录、完成步骤、依赖及源码哈希。
- `source/`：训练时源码及输入，按字节复制；各次版本可以不同。
- `evaluation_<最终回合>.json`：最终开发评价，保留完整逐时数据。
- `development_summaries/`：其他开发阶段的派生摘要，去除逐时数据，保留来源哈希与派生说明；没有冒充原文件。
- `holdout_108.json`、`holdout_500_108.json`：存在时原样复制最终评价。后一文件来自已训练500回合的检查点，模型配置的2000是计划总预算。
- `checkpoints.json`：本地模型路径、大小和SHA256；不包含`.pt`权重。

本归档含416个原样复制文件、157个标明来源的派生文件，共84654005字节（不含本说明、总manifest及独立登记的figures）。不包含期刊PDF/XML、论文图片或全文摘录。诊断目录只保存必要探针及结果；中间试验的重复分析可由原始最终开发数据重新生成。

## 复核与复现

在仓库根目录运行：

```sh
python tools/summarize_profit_only_study.py --runs-root experiments/sac_parameter_revision_20261018/runs --output checks/recheck_profit_only
python tools/analyze_price_response.py --evaluation experiments/sac_parameter_revision_20261018/runs/sac_revision_forecast18_scale3_2000/holdout_108.json --environment experiments/sac_parameter_revision_20261018/runs/sac_revision_forecast18_scale3_2000/effective_environment.json --profile experiments/sac_parameter_revision_20261018/runs/sac_revision_forecast18_scale3_2000/effective_profile.json --output checks/recheck_candidate
```

输出目录使用新名称。汇总会检查完整性、冻结源码、实际参数、逐日收益、保存的逐时奖励和物理结算；训练小时轨迹未保存，因此不能逐小时回溯所有训练步骤。

仅有Git仓库时，可按冻结版本重新训练；例：

```sh
python experiments/sac_parameter_revision_20261018/runs/sac_revision_forecast18_scale3_2000/source/train_sac.py train --config experiments/sac_parameter_revision_20261018/runs/sac_revision_forecast18_scale3_2000/config.json --output runs/reproduce_candidate
```

本机原检查点保留在`runs/`，使用对应`source/train_sac.py evaluate`和`analysis/final_evaluation_plan.json`中的参数复查。CLI没有完整恢复回放与随机状态的续训功能，不应将加载评价解释为精确续训。

诊断探针按原字节保存，部分以`parents[2]`定位仓库。重新运行前，应将所需探针复制回`checks/sac_revision/`原路径，再执行；不要直接从本归档目录运行。归档脚本位于[diagnostics/archive_study.py](diagnostics/archive_study.py)，可用`--root`显式指定仓库，但重新归档仍需要本地完整`runs/`和`checks/`源记录。

验证所有托管归档文件：

```python
import hashlib, json
from pathlib import Path
root = Path('experiments/sac_parameter_revision_20261018')
manifest = json.loads((root / 'archive_manifest.json').read_text())
for item in manifest['files']:
    data = (root / item['destination']).read_bytes()
    assert len(data) == item['bytes']
    assert hashlib.sha256(data).hexdigest() == item['sha256']
print('Verified', len(manifest['files']), 'files')
```
