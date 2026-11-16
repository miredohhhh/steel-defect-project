# 第一阶段论文参考文献与工艺关联核查

日期：2026-11-09。范围：本轮 MLPTA.tex 的六类表及全部 13 条参考文献。分类识别资料只能支持类别、图像任务或模型方法，不能单独证明工业缺陷的形成机制。

## 六类表的证据处理

| 类别 | 本轮处理 | 证据边界 |
| --- | --- | --- |
| Cr | 改引 [Bahrami 等，Metals 2019](https://doi.org/10.3390/met9070801) 的热轧钢板表面裂纹案例 | 研究对象是重板而非 NEU-CLS 图像，文中明确为案例背景和待查方向 |
| In | 保留 [Guo 等，Scientific Reports 2019](https://doi.org/10.1038/s41598-018-36766-6) 的连铸坯夹杂表征研究 | 需显微与成分证据，不能由 NEU 类别直接认定夹杂来源 |
| Pa | 删除“碎屑压入”等原先依赖分类论文的具体成因 | 目前无足够直接证据，明确标为工艺关联待核实 |
| PS | 改引 [Pimenov 等，Steel in Translation 2019](https://doi.org/10.3103/S0967091219100115) 的热轧机组“pitted surface”类缺陷案例 | 可查到摘要对轧辊/带钢冷却和氧化铁皮压入的分析；尚未核阅全文，不能迁移成 NEU 样本的根因 |
| RS | 保留 [Dey 等，Engineering Failure Analysis 2013](https://doi.org/10.1016/j.engfailanal.2013.07.018) 的氧化铁皮表征案例 | 直接讨论氧化层、除鳞和随后的轧制；具体成因仍需材料与工况证据 |
| Sc | 改引 [AIST 热轧工艺培训资料](https://www.aist.org/AIST/aist/AIST/Conferences_Exhibitions/Training_Seminars/Training%20Presentations/10-Hot-Rolling-Processes-msts-20.pdf) 的划痕缺陷部分 | 是专业培训资料，不是 NEU 样本现场记录；仅用于提出接触路径核查 |

## 文献元数据复核

| 论文键 | 已核对的作者、刊物/资料、年份与 DOI/链接 | 在正文中的用途 |
| --- | --- | --- |
| li-review | 李跃、王子铭、李鑫林、岳强；《钢铁研究学报》35(8):950–962，2023；[10.13228/j.boyuan.issn1001-0963.20220363](https://doi.org/10.13228/j.boyuan.issn1001-0963.20220363) | 检测方法与工业背景 |
| tang-review | **Bo Tang、Li Chen、Wei Sun、Zhong-kang Lin**；IET Image Processing 17(2):303–322，2023；[10.1049/ipr2.12647](https://doi.org/10.1049/ipr2.12647) | 机器视觉综述；已纠正旧稿的“C. Li”误署 |
| wen-survey | Xin Wen、Jvran Shan、Yu He、Kechen Song；Coatings 13(1):17，2023；[10.3390/coatings13010017](https://doi.org/10.3390/coatings13010017) | 钢材缺陷识别综述 |
| neu-database | 东北大学宋克臣团队[官方数据页](https://faculty.neu.edu.cn/songkechen/zh_CN/zhym/263269/)；无 DOI | NEU-CLS 来源、类别和图像性质 |
| wu-transfer | Hao Wu、Quanquan Lv；Journal of Sensors 2021:6637252；[10.1155/2021/6637252](https://doi.org/10.1155/2021/6637252) | 迁移学习识别研究 |
| resnet | Kaiming He、Xiangyu Zhang、Shaoqing Ren、Jian Sun；CVPR 2016:770–778；[10.1109/CVPR.2016.90](https://doi.org/10.1109/CVPR.2016.90) | ResNet 方法 |
| dey-scale | Arthita Dey、Tanmay Bhattacharyya、Sharmistha Dhara；Engineering Failure Analysis 34:478–487，2013；[10.1016/j.engfailanal.2013.07.018](https://doi.org/10.1016/j.engfailanal.2013.07.018) | RS 工艺案例 |
| guo-inclusion | Jing Guo、Shu-sen Cheng、Hanjie Guo、Yaguang Mei；Scientific Reports 9:2929，2019；[10.1038/s41598-018-36766-6](https://doi.org/10.1038/s41598-018-36766-6) | 连铸坯夹杂来源与表征 |
| guo-light | Naixuan Guo、Haonan Fan、Qin Dong、Rongchen Gu、Sen Xu；Sensors 26(5):1618，2026；[10.3390/s26051618](https://doi.org/10.3390/s26051618) | 仅作为 NEU-CLS 分类研究背景，已从工艺成因引文中移除 |
| pytorch-transfer | [TorchVision ResNet18 官方文档](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)；在线文档，无 DOI | 权重与预处理 |
| bahrami-crack | Abbas Bahrami、Mahdi Kiani Khouzani、Seyed Amirmohammad Mokhtari、Shahin Zareh、Maryam Yazdan Mehr；Metals 9(7):801，2019；[10.3390/met9070801](https://doi.org/10.3390/met9070801) | Cr 相关热加工裂纹案例，非同一数据集 |
| pimenov-pitted | V. A. Pimenov、A. I. Dagman、A. K. Pogodaev、N. N. Zhovnodii、D. A. Kovalev；Steel in Translation 49(10):703–708，2019；[10.3103/S0967091219100115](https://doi.org/10.3103/S0967091219100115) | 按[期刊卷期目录](https://journals.rcsi.science/0967-0912/issue/view/11262)调整末两位作者顺序；PS 相关机组案例，全文待核。其他二级索引给出不同顺序，提交前宜核原文首页 |
| aist-rolling | Association for Iron & Steel Technology，Making, Shaping and Treating of Steel: 101，热轧培训资料，2020，相关内容为幻灯片编号 57、65；[原始 PDF](https://www.aist.org/AIST/aist/AIST/Conferences_Exhibitions/Training_Seminars/Training%20Presentations/10-Hot-Rolling-Processes-msts-20.pdf)，无 DOI | 57 为氧化铁皮，65 为划痕；PDF 为每页两张幻灯片，故不写作 PDF 页码 |

## 待人工复核

1. 核阅 Pimenov 等论文首页与全文，确认作者最终顺序，以及摘要中的“pitted surface”是否与 NEU-CLS 的 PS 具有可比的缺陷定义；若不具可比性，应进一步弱化或删除表中关联。
2. 由具备材料/工艺背景的组员复核 Cr、In、PS、RS、Sc 的具体工序适用范围，并继续查找 Pa 的直接工艺证据。
3. 公开发布包含原始 NEU-CLS 样例图的论文前，核对数据提供者的图像再发布许可。
