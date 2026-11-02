# SPT-MoE

与`SPT-MoE_Methods_v5`对应的最小计算方法代码。仅包含SPT-MoE最终方法，不包含其他模型、调优队列、历史图件、ESM特征或权重。湿实验不在本包范围内。

## 目录

- `src/`：数据方法、ESM特征、模型、训练、推理与评价。
- `config.json`：最终模型和训练设置。6专家、两通道逐残基ESM均值/最大值、融合维度64；P12轮→T1 39轮。
- `metadata/`：固定DMS/HuSP成员、标签、分组、输入片段和划分；NLUC排除名单和三共享对照对齐依据。没有大型特征。
- `METHODS_MAP.md` / `SOURCE_MAP.json`：方法与原实现对应。
- `verification.json`：本次实际核验结果。

## 环境

建议Python3.11。安装`requirements.txt`；根据机器CUDA版本安装合适的PyTorch。`environment_reference.json`记录核验环境，不是跨平台锁文件。ESM2 3B提取需要独立取得官方checkpoint与足够显存，推理和训练不自动下载模型。

以下命令从本目录运行。所有数据/输出路径通过参数传入；目标输出使用新目录。不要把受限实验数据和大型特征直接提交到公开仓库。metadata中的序列及标签属于本项目数据，公开范围由作者确认。

## 1. 数据与特征

历史成员以`metadata/{dms,dpf}_members.csv`为准，其中`dpf`指HuSP。DMS输入为信号肽前20aa，HuSP为实际片段。`sample_id`、`group_id`、`entity_group`、`split`不能随意重排或替换。label冲突的两条DMS记录已经不在成员表中。

重新构建HuSP阴性时：

```bash
python src/prepare.py truncate --root INPUT_FASTA_ROOT --output NEW_NEGATIVE_DIR
```

输入根目录含`T1/`和`T2/`FASTA文件。沿用用户原始算法：seed42，长度从[8,min(30,L)]均匀整数采样，取N端；不足8aa明确排除。整理版固定文件名排序、保留FASTA记录顺序。历史脚本的st.listdir顺序未保存，因此**历史逐条片段复现以成员表/原FASTA为准**。随后调用`select_husp_positives`、`match_husp_negatives`完成已记录筛选与两阶段匹配。阳性注释输入应已限定human reviewed SIGNAL；阴性来自人源无SIGNAL类别池，不声称全部reviewed。该包不含数据库下载器。

ISC入口是`data_methods.process_scheme_b`，传入含`exp_id,theta_total`的表，使用p=1e-6及k=norm.ppf(1-p/2)。按实验分段归一化后全局ISC，阳性取≥阈值，阴性取全局等量最低值。不要调用旧逐记录随机划分；复现最终数据使用固定成员表。

从成员表`sample_id,input_sequence`生成FASTA，ID保持唯一；特征文件用ID的SHA256命名，兼容Windows特殊字符；ESM每次一条序列：

```bash
python src/extract_esm.py --fasta INPUT.fasta --checkpoint ESM2_3B.pt --output NEW_ESM_DIR
python src/prepare.py pack --manifest metadata/dms_members.csv --features DMS_ESM_DIR --output DATA/dms
python src/prepare.py pack --manifest metadata/dpf_members.csv --features HUSP_ESM_DIR --output DATA/dpf
```

输出每个split一份CSV和NPZ。NPZ含`ids,esm,tokens,mask`；esm为(N,2560)，tokens为(N,34,8)兼容原特征接口，仅前两通道用于本模型，其余通道置0。BOS/EOS及padding不参与池化；mask=True表示真实残基。逐条`<SHA256(sample_id)>.npy`为实际残基×2560，不含BOS/EOS。

## 2. 最终两阶段训练

```bash
python src/train.py --data DATA --output NEW_MODEL_DIR --device cuda
```

仅读取两个域train成员；scaler按域等权、域内按唯一序列等权拟合。P用HuSP标签训练；T1复制P参数和HuSP头初始化DMS头，冻结HuSP头及输入分支/E1–E5，仅更新gating、FFN、E6及DMS头。每个DMS批次配同量HuSP回放，两损失等权；DMS逆重复权重、HuSP按唯一序列权重采样。最终12/39轮沿用原60/40轮cosine日程，不在历史val/test上早停。每阶段只保存最终选定权重。

如需理解训练内长度选择，`train_stage(validation=...)`保留原checkpoint排序逻辑：P最大化HuSP AUC；T1第16轮起先满足HuSP AUC≥.95，再最大化DMS AUC；无可行候选按HuSP/DMS AUC选取，patience10/8。此函数只接受训练内折验证，不允许传入历史val/test或NLUC。本包不含完整外5/内3调度，固定最终训练入口不重新搜索。

最终产物为P.pth、T1.pth、scaler.pkl、配置/训练loss及pooled_reference.npy。参考CDF以DMS1492+HuSP4552条唯一训练序列的双头MC64分数合并得到，域按实际数量合并，不是两域等权CDF。

## 3. 冻结预测与外部评价

```bash
python src/evaluate.py --model MODEL_DIR --features EXTERNAL.npz --metadata EXTERNAL.csv --output NEW_PREDICTIONS.csv
```

CSV必须含sample_id且与NPZ ids逐条一致。相同输入/层/抽样编号绑定Philox采样流；普通dropout关闭，MC64平均margin。可另用`--sampling-group independent`进行复核，不能按相关性选采样组。

百分位定义为`(n_< + n_≤ + 1)/(2(N+1))`，分别映射两个头后取均值。CDF只来自训练序列。融合分数不是物理活性或校准概率。

NLUC评价调用`evaluate.merge_external`与`correlations`：输入列为sample_id、input_sequence、batch（REF/Recom）、activity、score、signalp_sp。先排除B2M/P61769及metadata列出的4条REF；只在Recom批次内平均合并相同序列。REF38、Recom30、All68；Recom活性乘`metadata/batch_alignment.json`的recom_to_ref_multiplier后与REF合并。B2M可仅作展示，不进入统计或KDE。批次内用原活性，All用对齐活性。SignalP对照使用Eukarya SP(Sec/SPI)，输入已有线上预测，不调用在线服务。

## 检查与限制

```bash
python check.py
```

该检查覆盖导入、短序列/N端截取、ISC、特征形状、两头推理复现、CDF并列值。`verification.json`另记录整理时与原实现/封存权重的核对。未重新训练，不承诺不同GPU/软件版本参数逐位相同。不把小规模检查称为全流程训练复现。完整数据收集及数据库release元数据仍需配套原始输入。本包不附授权条款或虚构数据库版本。

