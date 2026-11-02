# DNA diversity model

本目录对应 **DNA序列丰富度的统计建模与估计方法 v1.1**，只保留计算方法。主估计器为 first-order jackknife；不均匀丰度使用原始分子丰度的Gini约束。对数正态权重仅作为同一raw Gini约束下的丰度分布族对照。

## 文件与方法对应关系

| 文件 | 内容 |
|---|---|
| `richness.py` | 精确核苷酸序列计数、同一样品独立文库的序列并集计数、一阶刀切估计、嵌套无放回reads下采样 |
| `capture_model.py` | 分子预算、等丰度映射、最低丰度1的结构边界、Dirichlet参数校准、解析捕获均值及Beta积分数值替代 |
| `simulation.py` | Gamma概率分组、固定总量归一化、条件二项捕获、预测分位数、对数正态均值对照 |
| `inference.py` | 前向均值反演、分位数偏移插值、95%条件模型反演区间、跨样品范围与中位数 |
| `run_analysis.py` | 模块化命令行入口；导入代码不会运行分析 |
| `config.json` | 物理参数、Gini网格、下采样与模拟设置 |
| `source_map.json` | 原项目实现的相对位置、来源标识及本次移植检查记录；不依赖原项目目录运行 |

这里不包含原始数据、结果快照、绘图、其他丰富度估计器、其他Gini定义、历史试算或文稿生成工具。

## 环境

Python 3.10或以上；建议使用独立环境。固定的NumPy、SciPy、pandas版本对应原分析记录。

```bash
python -m pip install -r requirements.txt
python run_analysis.py --help
```

路径均通过命令行输入。默认配置从本代码目录读取，不依赖个人电脑绝对路径。`--config`须放在子命令前；`--output`放在子命令后。每个模块请使用独立的输出目录，避免覆盖自己的结果表。

## 1. 文库丰富度与测序深度敏感性

meta为带表头的TSV：`id,file1,file2,file3`（制表符分列），也支持独立文库的`id,file`。同一行多个文件的序列计数相加；文件名必须精确匹配。相对路径从`--data-dir`解析，绝对路径也可读取。每个文件只能属于一个样品。

原始输入是一条read一行的TSV，最后一列为`Seq`，可带`ID\tSeq`表头，也支持单列序列和gzip压缩。**不支持将Seq/Count计数表误作原始read文件输入。** 保留大小写、序列长度差异和singleton的精确字符串身份，允许ACGTN及其小写；空序列跳过，非法字符报错，不执行错误校正。编辑及建库操作的实验方法不在本代码中规定。

```bash
python run_analysis.py richness --meta metadata.tsv --data-dir data --output results/richness
```

默认20次独立重复，每次包含10%至100%的嵌套无放回子集。总reads少于10时默认最小比例可能为空，不适用该默认网格。`D_library_replicates.csv`保存逐重复结果，`D_library_summary.csv`保存均值、技术SD及经验分位数，`full_depth_estimators.csv`保存完整深度jack1输入。100%重复相同，技术SD为零并不意味着估计器误差为零。

下采样基种子为20261010；`sample_seed_overrides`中的ID 10使用20261020，对应原分析增加第十个样品时的随机流。对其他项目使用时，应在自己的配置中明确样品ID和种子。序列计数的首次出现顺序及合并文件顺序会影响固定随机种子的逐次下采样实现。

## 2. 解析均值与结构范围

```bash
python run_analysis.py curves --output results/curves
```

输出`mean_curves.csv`。`CaptureModel.structural_bounds(D_library)`给出最低丰度1的结构上下界，条件估计表也包含这两列；结构范围不是置信区间。

物理参数由`physical`统一设置：N0=25M、p=0.2、扩增8倍、C=1、eta=1、已知建库前取样1/30。程序由此计算Ns=N0*p、M=8*C*Ns、beta=eta/30。p为模型中的编辑效率，并被视为筛选后进入扩增的细胞比例；这些是计算输入与假设，不是程序测出的实验参数。合并文库的eta解释为有效并集系数。

## 3. 条件预测区间

完整模拟需要显式执行；不会随其他命令自动启动：

```bash
python run_analysis.py simulate --output results/predictions
```

默认Gini=0至0.8、步长0.1，2,048组；每个Gini有43个D位置，一般位置4,000次、三个关键位置10,000次，合计1,710,000次实现。分块大小256。原项目该部分历史运行约需十几至二十分钟，实际耗时随CPU及数学库变化；本次代码整理没有重跑它。默认只保留汇总CSV，避免保存大规模模拟重复；需要重复数组时加`--keep-replicates`。随机流由基种子20261015和配置标识生成；改变分块大小也可能改变逐次随机结果。

局部加密使用显式节点CSV（列为`D_cells,target_gini`）：

```bash
python run_analysis.py simulate --nodes local_nodes.csv --output results/refinement
```

输出`prediction_intervals.csv`，包括解析均值、模拟均值、q025/q975及模拟检查量。95%区间为固定模型下单次捕获种类数的预测区间，随机来源为丰度分配和条件捕获，不是均值的置信区间。原函数还保留用于诊断的99%分位数，它们不改变Methods中的95%口径。

## 4. D_cells点估计及95%条件反演区间

有现成的完整深度丰富度表和预测表时，直接使用，无需重新计数或模拟：

```bash
python run_analysis.py infer --library-table results/richness/full_depth_estimators.csv --predictions results/predictions/prediction_intervals.csv results/refinement/prediction_intervals.csv --output results/inference
```

输入丰富度表需要`sample_id`及`D_library`或`D_library_mean`；如果存在`estimator`列则只采用`jack1`，存在`fraction`列则只采用完整深度。每个样品须有一条有效输入。预测表支持直接读取原项目同名列的正式表格；`--predictions`按“主网格、局部加密”顺序传入，重复节点以最后提供的记录为准。

不提供预测表仍可运行`infer`：计算点估计及G=0的二项分布区间，非零Gini区间标记为`prediction_grid_not_supplied`，不会偷偷启动模拟。

输出：

- `sample_D_cells_by_gini_with_intervals.csv`：条件点估计、95%反演端点、可行性和边界标志、结构范围。
- `D_cells_ranges.csv`：各Gini下有效样品点估计的最小值、最大值、中位数、有效样品数及区间端点包络。

D_cells为细胞群中**序列种类数**，不是阳性细胞数。超过模型均值容量的输入保留无均值解状态；均值无解与预测区间无解分别处理。95%条件反演不包括jack1估计误差、p、eta、目标Gini或分布族不确定性；跨样品区间包络不表示同时95%覆盖。

插值沿用原实现：先取模拟分位数减解析均值的偏移，对偏移作线性插值，再恢复解析均值，并限制于[0,D]。网格外采用最近端点的偏移，因此应提供覆盖相关反演位置的网格及必要的局部加密；程序拒绝未覆盖细胞数上限的预测网格，并在求区间根前检查分位数带的数值单调性。参数M、beta不一致的预测表会被拒绝。

原十样品正式区间沿用了原九样品的既有结果，只对新增样品补充节点。本代码提供同一计算方法的统一入口，不再保留“追加第十个样品”的历史分支。因此，用不同的合并节点集合统一重算时，不承诺逐位复现历史冻结区间；要核对原表，应沿用对应的节点集合、随机配置和原有记录，而不是默认重跑。

## 5. 丰度分布族均值对照

```bash
python run_analysis.py sensitivity --output results/sensitivity
```

默认用512次丰度实现校准归一化对数正态权重，独立4,000次实现比较均值；Gini=0.2/0.5/0.8，七个D位置。`lognormal_mean_sensitivity.csv`仅输出均值对照，不把该分布族的尾部区间用于论文报告。

每次命令均写入`run_config.json`，记录配置、输入参数、依赖版本和运行状态。该文件不复制输入数据。原始数据标识和实验记录由仓库所属项目另外管理。
