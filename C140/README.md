# C140

与最终 C140 Methods 对应：DNA → ESSI 联合 Z → 各实验右尾 ISC 阳性 → 阳性并集 → 1aa 家族 → ASOP 图及三个代表家族的树和信息量 logo。

## 结构与环境

保持 `C140/` 与 `ESSI/` 为同级目录；评分、ISC 和 XLSX 导出直接复用 ESSI，不复制第二套统计实现。

- `run.py`：命令入口，`score` 与 `analyze` 两个阶段。
- `analysis.py`：只读取完整结果中必要列、构建阳性并集、Hamming 连通分量、确定性最小生成树和代表家族选择。
- `figures.py`：最终 ASOP 图、代表家族圆形树及 bits logo。
- `config/`：唯一的项目参数入口、最小样品映射和 rule1。
- `test_c140.py`：ISC 停止规则、传递聚类、ASOP 定义和 logo 单位检查。

Python >=3.9。从包含这两个目录的仓库根目录运行：

```bash
python -m pip install -r C140/requirements.txt
python C140/test_c140.py
```

代码使用 NumPy 与 Matplotlib。原项目 zsx 工具的 Hamming 计算、根选择和圆形布局规则已保留为局部实现；DC 配色采用原确认的十六进制值，不再依赖个人工具包或历史缓存。

## 运行

第一阶段从原始目标区 DNA 计算完整结果：

```bash
python C140/run.py score --data /path/to/C140/data --output results/essi
```

只读取 rule1 所需七个物理文件，exp1/exp2 共享 pre pool。不会为未参与 rule1 的数据另做翻译导出。meta 和 rules 路径相对于配置文件，`--data` 与 `--output` 相对于命令运行位置。可先加 `--check-only` 查看样品计划。

第二阶段筛选、聚类和绘图：

```bash
python C140/run.py analyze --exp1 results/essi/exp1/rule1/analysis.xlsx --exp2 results/essi/exp2/rule1/analysis.xlsx --output results/c140
```

已有完整 `exp1_rule1.xlsx`、`exp2_rule1.xlsx` 时可直接填入这两个路径，跳过第一阶段。必须提供完整 `results`，不能用已筛选阳性表重新拟合 ISC。结果列名默认 `label`、`Z_total`，集中配置在 `config/c140.json`。

## 计算口径

右尾 ISC 使用总体标准差，双尾名义 alpha=0.0001 设置 `k=Phi^-1(1-alpha/2)`；最多100次，均值和标准差变化均小于1e-4时停止，使用停止迭代更新前的阈值，严格 `Z_total>threshold` 入选。该 alpha 是阈值参数。

完整 15aa 序列按字典序排列。Hamming<=1 连边的连通分量定义家族，按节点数降序、最小成员索引升序编号。家族具有传递性，不要求家族内所有成员两两仅相差1aa。全局 Hamming 最小生成树使用相同的确定性 Prim 规则，切断>1aa的边得到家族树。

`A=exp1_only`、`B=exp2_only`、`O=overlap`：

```text
ASOP = max(A, B)/(A+B+O)
X = sign(A-B)*ASOP
Y = A+B  (log axis)
```

仅总节点数>=10的家族进入主图，Y=0的全 overlap 家族不画。点面积52 pt²，无边框；原 DC 红→绿→蓝连续色带；±0.5参考线。仅真实坐标完全一致的点按一个名义半径确定性展开，所有指标、颜色与示例选择用真实坐标。保留最终已确认的坐标轴与排版。

代表家族按距X=-1、0、+1最近选择，再依次按专属节点数、总节点数降序、簇编号升序破除并列。树根取距等权 one-hot 中心最近的已观测序列。Logo 每条唯一序列等权，信息量为 `log2(20)-entropy`，字母高度为频率乘信息量，单位 bits，无伪计数或小样本修正；I 使用可见横杠字体。

## 输出与原结果核对

- `exp1_positive.xlsx`、`exp2_positive.xlsx`：阳性序列及联合 Z。
- `c140_summary.xlsx`：阈值、迭代过程、并集、全部家族、成员、真实/展示坐标、代表家族、树边及 logo 逐位点值。
- `family_size_signed_asop_annotated.png/.pdf`：最终 ASOP 图。
- `selected_families/*_tree_with_logo.png/.pdf`：三个代表家族。
- `run.json`：配置、输入位置、完成状态和统计摘要。

表格仅 XLSX，图件仅 PNG/PDF。没有缓存依赖，不输出距离矩阵或中间迭代图。

原 C140 记录的预期结果：exp1 阈值2.005727056624045、阳性2254；exp2 阈值1.8395718171443114、阳性3391；并集4328，三类为937/2074/1317；全部380个家族，>=10节点127个、共3717条序列，实绘126个。代表家族依次为 `AA1_C0015`、`AA1_C0001`、`AA1_C0023`。这些是核对锚点，不作为代码强制输出常量。

本次整理保留最终方法范围，不包括2aa/3aa、UMAP、全景网络、置换对照、距离曲线、饼图、DNA丰度分析和文稿生成。Hamming 矩阵约18 MiB（4328序列），临时广播数组约8 MiB；完整结果读取和ESSI评分的内存/时间取决于样品规模。本次未重新运行全量 DNA 评分。
