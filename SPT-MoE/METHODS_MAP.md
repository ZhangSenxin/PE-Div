# V5 Methods与代码对应

| V5内容 | 代码 |
|---|---|
| ISC、分实验归一化和全局等量阴性 | src/data_methods.py: isc_threshold / normalize_theta_per_exp / process_scheme_b |
| 阴性N端均匀随机截取 | src/prepare.py: truncate_sequence / truncate_fasta（用户2026-11-02提供原始算法） |
| HuSP reviewed阳性精确匹配、ATG/编码框/去重和排除 | src/prepare.py: select_husp_positives |
| 阴性冲突排除、两阶段长度/类别匹配 | src/prepare.py: match_husp_negatives; src/data_methods.py |
| 实体分组划分 | src/data_methods.py: assign_entities_to_row_targets；实际成员由metadata固定 |
| ESM2 3B第33层、实际残基池化 | src/extract_esm.py; src/prepare.py: feature_arrays |
| SPT-MoE与伯努利注意力 | src/model_configs.py |
| 两域等权scaler、P预训练、T1有限解冻与回放 | src/train.py |
| 训练内checkpoint选择 | src/train.py: train_stage(validation=...)；最终训练不早停，使用固定12/39轮 |
| 双头固定MC64推理 | src/inference.py; bernoulli_validation.py; cached_dms_mc.py |
| 合并唯一训练参考CDF、两头等权 | src/inference.py: percentile / pooled_fusion |
| NLUC重复合并、批次对齐、相关系数 | src/evaluate.py |

来源符号记录见SOURCE_MAP.json。迁移头保持原参数键名以兼容封存权重；dpf为HuSP的历史内部字段，不是另一个数据集。四列margin接口中后两列仅兼容接口，不含回归模型。

边界：这是从已执行实现整理的最小方法代码，不含搜索队列、其他模型、完整嵌套评估调度或论文绘图。已发布成员优先由metadata加载；不能用重新划分冒充原划分。NLUC后处理及相似度规则存在探索性复用的时间线，代码整理不改变该事实。
