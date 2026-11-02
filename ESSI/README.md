# ESSI

与最终 ESSI 主方法及补充方法对应的最小计算实现。入口从已按样品拆分的目标区 DNA、氨基酸序列或已鉴定标签开始，输出单轮及联合的 `Z_total` 和 `theta_total`。不包含 FASTQ 前处理、参考库比对或重组鉴定。

## 文件与环境

- `multi_round_pipeline.py`：输入计数、DNA 翻译、单轮 ESSI、多轮联合和 XLSX 导出入口。
- `isc.py`：按双尾名义 alpha 设置 k、分别拟合右尾背景的 ISC 函数；C140 直接复用。
- `xlsx_stream.py`、`assets/xlsx_styles.xml`：流式 XLSX 导出所需文件，迁移时一起保留。
- `test_essi.py`：公式、固定联合权重、缺失方差修正和同义编码合并检查。

Python >=3.9。评分入口无需第三方包；`isc.py` 使用 NumPy：

```bash
python -m pip install -r ESSI/requirements.txt
python ESSI/test_essi.py
```

上述命令从包含 `ESSI/` 与 `C140/` 的代码仓库根目录运行。依赖范围是整理版的兼容要求，不用于追认历史分析环境。

## 运行

```bash
python ESSI/multi_round_pipeline.py --meta meta.csv --data /path/to/data --rules rule1.txt rule2.txt --input-type dna --reading-frame 0 --stop-policy exclude --dual-absent legacy --output results/essi
```

`--rules` 可以接收一个或多个文件，`--batch exp1,exp2` 可以指定批次。也可用 `--input-type aa` 或 `label` 从已准备的序列/标签开始。

DNA 文本可为单列，或带 `ID\tSeq` 表头的双列；每行一条观测，ID 不去重。frame 0、标准遗传密码表、完整翻译；默认剔除终止、非 A/C/G/T 和不完整密码子，同义 DNA 按氨基酸汇总。

meta 第一列是唯一 batch，后续具名列是样品 ID。例如：

```text
batch,pre pool,pos lib2,pos lib3
exp1,Q0,Q1,Q2
```

rules 每行左侧为 post、右侧为 pre，按行顺序联合：

```text
pos lib2 vs pre pool
pos lib3 vs pos lib2
```

文件名默认 `{sample}_45.txt`，可用 `--file-template` 调整。样品合并语法如 `pos lib(2+3)` 沿用原实现。

## 计算与结果

每轮 pre/post 并集大小 K，保留读数 N，伪计数 1：

```text
f = (count + 1) / (N + K)
S = log2(f_post / f_pre)
v = [1/(count_post+1) + 1/(count_pre+1)] / ln(2)^2
Z = S / sqrt(v)
tau2 = max(0, population_variance(S) - mean(v))
theta = tau2 / (tau2 + v) * S
```

多轮先对标准 S、v 求和，估计联合 tau2 和权重，再作双端缺失补值。默认 `legacy` 同时调整 S 和 v，固定补值前的联合权重；最终 theta 不是逐轮 theta 之和。具体近似条件以 ESSI 补充方法为准。

主要结果在 `results/essi/<batch>/<rule>/analysis.xlsx` 的 `results` sheet。另有逐轮表、统计表和运行参数 JSON。全部表格为 XLSX，不输出 TSV。

通用 CLI 原有的 `filtered_by_threshold.xlsx` 使用 `Z_total>0` 加末轮计数门槛；它不是 C140 的论文阳性集合。`significance` 星号来自固定 Z 常数，`ISC_enriched` 默认 k=2，二者均不能替代 C140 的 alpha=0.0001 右尾筛选。C140 的正式入口见相邻目录。

如需以 alpha 调用 ISC：

```python
from isc import right_isc
result, history = right_isc(values, alpha=0.0001, tolerance=1e-4, max_iterations=100)
positive = values > result['threshold']
```

输出目录必须为空；不会覆盖历史结果。`--check-only` 只检查配置并展示样品计划。
