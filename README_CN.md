<div align="center">

# <img src="assets/onevl_logo_new.png" alt="OneVL Logo" height="48" style="vertical-align:middle"/> OneVL：融合图文解释的单步隐式推理规划视觉语言动作模型

[![技术报告](https://img.shields.io/badge/Tech%20Report-arXiv-red?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2604.18486/)
[![项目主页](https://img.shields.io/badge/Project%20Page-blue?style=flat-square&logo=googlechrome)](https://xiaomi-embodied-intelligence.github.io/OneVL/)
[![模型权重](https://img.shields.io/badge/Model%20Weights-HuggingFace-yellow?style=flat-square&logo=huggingface)](https://huggingface.co/collections/xiaomi-research/onevl-models/)
[![开源协议](https://img.shields.io/badge/License-Apache%202.0-green?style=flat-square)](LICENSE)


</div>

---

[English](README.md)、[简体中文](README_CN.md)

## 项目概述

OneVL是面向自动驾驶场景的**视觉-语言-动作（VLA）**框架，轨迹预测精度达到业界顶尖水准，推理延迟与仅输出结果的自回归模型持平。
该框架创新性引入双模态辅助解码器，解决传统隐式思维链算法的固有缺陷，通过约束紧凑隐式表征同时编码语言推理逻辑与场景动态变化规律。

### 三类思维链技术范式

<div align="center">
<img src="assets/comparison.png" alt="三类思维链范式对比" width="90%"/>
</div>

> (a) 显式思维链：先完整生成推理过程再输出结果，可解释性强但推理速度慢
> 
> (b) 隐式思维链：将推理压缩为不可解读的隐式向量，速度快但无法溯源分析
> 
> (c) 本文OneVL模型：设置视觉隐式令牌`v`与语言隐式令牌`l`；训练阶段借助双辅助解码器，分别还原未来画面与推理文本；推理阶段舍弃解码器，将隐式表征预填充至提示词。兼顾隐式模型的高速特性，同时保留图文双维度可解释能力。

### 模型架构

<div align="center">
<img src="assets/framework.png" alt="OneVL模型架构" width="90%"/>
</div>

> 训练过程中，视觉隐式位置的隐藏特征送入**视觉辅助解码器**，预测0.5秒、1秒后的场景画面令牌；语言隐式位置特征输入**语言辅助解码器**，还原推理逻辑文本。推理阶段移除两类解码器，隐式令牌一次性预载入上下文，推理耗时等同于纯结果输出模型。

模型基于**通义千问3-VL-4B-指令模型**扩展搭建，新增核心模块：

- **隐式令牌交互层**：回复区预置4个视觉隐式令牌、2个语言隐式令牌，复用现有词表，无需新增特殊标记
- **视觉辅助解码器**：依托13.1万码本的Emu3.5图像量化模型，依据视觉隐特征预判后续画面，充当场景世界模型监督信号
- **语言辅助解码器**：结合视觉特征，从语言隐表征还原完整推理话术
- **预填充推理机制**：推理剔除解码器，隐式令牌并行处理，仅轨迹序列自回归生成

### 核心创新点

1. **双模态辅助解码**：语言解码器还原可读推理逻辑，视觉解码器预判场景画面，让隐式表征贴合真实物理场景规律
2. **预填充加速推理**：隐式表征单次并行载入上下文，在NAVSIM数据集推理速度较显式思维链提升1.5倍，道路施工场景提速2.3倍，延迟对标极简输出模型
3. **压缩表征提升泛化性**：业内首个在四项测试基准中，综合性能全面超越显式自回归思维链的隐式推理算法

---

## 开源资源

| 模块 | 开放状态 |
|----|----|
| 📄 技术论文 | ✅ [查看论文](https://arxiv.org/abs/2604.18486) |
| ⚖️ 模型权重 | ✅ [下载权重](https://huggingface.co/collections/xiaomi-research/onevl-models) |
| 🔍 推理代码 | ✅ [代码仓库](https://github.com/xiaomi-research/onevl) |
| 🏋️ 训练代码 | ✅ [训练源码](https://github.com/GeorgeLuImmortal/OneVL_training/tree/main) |

---

## 实验结果

### 精度与效率最优权衡（NAVSIM/ROADWork 道路施工数据集）

<div align="center">
<img src="assets/teaser_bar.png" alt="基准数据集精度效率对比" width="90%"/>
</div>

> OneVL在两项测试中均处于**最优区间**，延迟最低、预测指标最佳。过往隐式思维链算法在自动驾驶任务中表现均不及基础输出模型，本模型彻底突破该瓶颈。

### NAVSIM数据集完整测评

| 算法                 | 模型参数量 | 轨迹匹配分数↑ | 推理耗时(秒)↓ | 可解释维度 |
|--------------------|:----:|:----:|:----:|:----:|
| AdaThinkDrive      | 80亿 | 86.20 | — | 语言文本 |
| LaST-VLA           | 80亿 | 87.30 | — | 无 |
| 纯结果输出模型(AR Answer) | 40亿 | 87.47 | 4.49 | 无 |
| 显式推理+结果模型(AR CoT+Answer)        | 40亿 | 88.29 | 6.58 | 语言文本 |
| COCONUT算法          | 40亿 | 84.84 | 5.93 | 无 |
| CODI算法             | 40亿 | 83.92 | 8.62 | 无 |
| SIM-CoT算法          | 40亿 | 84.21 | 10.86 | 语言文本 |
| **OneVL本文模型**      | **40亿** | **88.84** | **4.46** | **视觉+语言** |

### ROADWork 道路施工场景数据集测评
| 算法                 | 平均位移误差(像素)↓ | 终点位移误差(像素)↓ | 推理耗时(秒)↓ | 可解释维度 |
|--------------------|:----:|:----:|:----:|:----:|
| YNet模型             | 22.68 | 80.78 | — | 无 |
| 纯结果输出模型(AR Answer) | 15.98 | 40.29 | 4.74 | 无 |
| 显式推理+结果模型(AR CoT+Answer)        | 13.18 | 29.98 | 10.74 | 语言文本 |
| COCONUT算法          | 15.44 | 38.60 | 6.06 | 无 |
| CODI算法             | 16.45 | 44.28 | 6.73 | 无 |
| SIM-CoT算法          | 16.49 | 44.32 | 6.19 | 语言文本 |
| **OneVL本文模型**      | **12.49** | **28.80** | **4.71** | **视觉+语言** |

### Impromptu 即兴驾驶场景数据集测评
| 算法 | 平均位移误差(米)↓ | 终点位移误差(米)↓ | 推理耗时(秒)↓ | 可解释维度 |
|----|:----:|:----:|:----:|:----:|
| Impromptu VLA模型 | 1.60 | 4.28 | 6.10 | 无 |
| 纯结果输出模型 | 1.46 | 4.03 | 4.24 | 无 |
| 显式推理+结果模型 | 1.42 | 3.96 | 6.84 | 语言文本 |
| COCONUT算法 | 1.49 | 4.07 | 5.27 | 无 |
| CODI算法 | 1.86 | 5.18 | 5.24 | 无 |
| SIM-CoT算法 | 2.43 | 6.10 | 5.09 | 语言文本 |
| **OneVL本文模型** | **1.34** | **3.70** | **4.02** | **视觉+语言** |

### APR1 路况数据集测评
| 算法 | 平均位移误差(米)↓ | 终点位移误差(米)↓ | 推理耗时(秒)↓ | 可解释维度 |
|----|:----:|:----:|:----:|:----:|
| Cosmos-Reason模型 | 2.86 | 7.42 | — | 语言文本 |
| 纯结果输出模型 | 3.27 | 9.59 | 3.06 | 无 |
| 显式推理+结果模型 | 2.99 | 8.54 | 3.51 | 语言文本 |
| COCONUT算法 | 3.29 | 9.48 | 3.76 | 无 |
| CODI算法 | 3.22 | 9.25 | 3.85 | 无 |
| SIM-CoT算法 | 3.40 | 9.85 | 3.78 | 语言文本 |
| **OneVL本文模型** | **2.62** | 7.53 | **3.26** | **视觉+语言** |

### 推理文本质量评测（NAVSIM）
| 算法          | 动作匹配准确率↑ | 语义相似度分数↑ | 大模型评审得分↑ | 综合均分↑ | 推理耗时(秒)↓ |
|-------------|:----:|:----:|:----:|:----:|:----:|
| 显式推理模型(AR CoT+Answer)    | 73.20 | 79.75 | 81.86 | 78.27 | 6.58 |
| SIM-CoT算法   | 67.20 | 76.25 | 78.73 | 74.06 | 10.86 |
| **OneVL模型** | 71.00 | 78.26 | 79.13 | 76.13 | **4.46** |

OneVL语言辅助解码器还原的推理文本质量可达显式推理模型的97%，同时保持极速推理性能。

### 模块消融实验（NAVSIM轨迹分数）

| 模型变体 | 语言辅助解码器 | 视觉辅助解码器 | 分阶段训练 | 轨迹匹配分数↑ |
|----|:----:|:----:|:----:|:----:|
| 移除视觉解码器 | 启用 | 关闭 | 启用 | 87.97 |
| 移除语言解码器 | 关闭 | 启用 | 启用 | 88.53 |
| 无分阶段训练 | 启用 | 启用 | 关闭 | 67.13 |
| **完整OneVL模型** | **启用** | **启用** | **启用** | **88.84** |

两类辅助解码器均对性能有正向增益，分阶段训练为模型核心必备策略，缺失后性能大幅暴跌。

---

## 可视化效果示例

### NAVSIM驾驶场景

<div align="center">
<img src="assets/navsim_example1.png" alt="NAVSIM场景可视化效果" width="95%"/>
</div>

> 图中绿色为真实行驶轨迹，红色为模型预测轨迹；同步展示视觉解码器生成的0.5秒、1秒预判画面，以及语言模块输出的驾驶推理逻辑。

### 道路施工区域通行场景

<div align="center">
<img src="assets/roadwork_example1.png" alt="施工路段可视化效果" width="95%"/>
</div>

---

## 环境部署

**运行要求**：Python 3.10及以上版本，英伟达CUDA显卡；推理建议显存不低于16GB

```bash
# 1. 使用 venv 创建并激活虚拟环境
uv venv venv/onevl --python 3.12
source venv/onevl/bin/activate

# 或使用 conda 
conda create -n onevl python=3.12 -y
conda activate onevl

# 2. 安装依赖库(推荐使用加速源镜像)

# 清华源(推荐)
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 阿里云
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 豆瓣
pip install -r requirements.txt -i http://pypi.douban.com/simple/
```

核心依赖清单(`requirements/framework.txt`)：

```
transformers>=4.57.6,<5.4.0   # 模型运行最低版本要求
trl>=0.15,<0.29
peft>=0.11,<0.19
deepspeed<0.19
qwen_vl_utils
timm
datasets>=3.0,<4.0
safetensors
einops
omegaconf
numpy
pillow
```

单独安装模型加速工具(see [Training → Quick Start](#quick-start))：

```bash
pip install git+https://github.com/modelscope/ms-swift.git#egg=ms-swift[all]
```

**快速注意力组件**：根据本机CUDA、PyTorch版本，在官方发行页下载对应安装包部署 [flash-attention releases page](https://github.com/Dao-AILab/flash-attention/releases) 。

---

## 模型训练

### 快速上手

模型依托 [ms-swift](https://github.com/modelscope/ms-swift) 框架采用**三阶段训练流程**，脚本自动识别显卡数量，支持多机分布式训练。

#### 前期准备

1. **Install ms-swift** (and its dependencies) / 部署框架依赖

```bash
pip install -e .
# 适配版本安装快速注意力组件 Install flash-attn matching your CUDA version from:
# https://github.com/Dao-AILab/flash-attention/releases
```

2. 下载基础模型权重(base VLM + visual aux decoder):

| 模型名称 | 开源地址                                                                                         |
|----|----------------------------------------------------------------------------------------------|
| 通义千问3-VL-4B指令模型 | [Qwen3-VL-4B-Instruct 模型下载](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)                |
| OneVL预训练权重 | [OneVL model weights 模型合集](https://huggingface.co/collections/xiaomi-research/onevl-models/) |

3. 测试数据集：工程目录内置100条样本测试集，可快速验证训练流程。

#### 阶段0：预热监督微调

仅针对结果或推理文本做基础微调，暂不启用隐式令牌结构。

```bash
# 常规训练脚本
bash run_script/train/navsim/sft_distributed_stage0_vis4_txt2_bs64.sh

# 显式推理基线训练
bash run_script/train/navsim/sft_distributed_qwen3vl_cot_64.sh

# 纯结果输出基线训练
bash run_script/train/navsim/sft_distributed_qwen3vl_answer_bs64.sh
```

#### 阶段1：辅助解码器专项训练

初始化隐式推理结构，冻结主干大模型，仅训练图文辅助解码器。

```bash
bash run_script/train/navsim/sft_distributed_stage1_vis4_txt2_bs64.sh
```

#### 阶段2：全模型联合微调

解锁所有模块参数，整体协同优化模型性能。

```bash
bash run_script/train/navsim/sft_distributed_stage2_vis4_txt2_bs64.sh
```

#### 多机分布式训练

双节点、每节点8卡训练示例：

```bash
# 主节点执行
NNODES=2 NODE_RANK=0 MASTER_ADDR=主节点IP bash run_script/train/navsim/sft_distributed_stage2_vis4_txt2_bs64.sh

# 从节点执行
NNODES=2 NODE_RANK=1 MASTER_ADDR=主节点IP bash run_script/train/navsim/sft_distributed_stage2_vis4_txt2_bs64.sh
```

#### 训练阶段汇总

| 训练阶段 | 启动脚本 | 冻结模块 | 训练模块 | 优化策略 |
|----|----|----|----|----|
| 预热微调 | sft_distributed_qwen3vl_answer_bs64.sh | 无 | 全模型 | 显存分级优化2 |
| 解码器初始化 | sft_distributed_stage1_vis4_txt2_bs64.sh | 主干模型、视觉编码器 | 双辅助解码器 | 显存分级优化2 |
| 全局联合调优 | sft_distributed_stage2_vis4_txt2_bs64.sh | 无 | 全模型 | 显存分级优化2 |

---

## 引用格式

如若使用本项目成果，引用标注如下：

```bibtex
@article{lu2026onevl,
  title={OneVL: One-Step Latent Reasoning and Planning with Vision-Language Explanation},
  author={Lu, Jinghui and Guan, Jiayi and Huang, Zhijian and Li, Jinlong and Li, Guang and Kong, Lingdong and Li, Yingyan and Wang, Han and Xu, Shaoqing and Luo, Yuechen and others},
  journal={arXiv preprint arXiv:2604.18486},
  year={2026},
  url={https://arxiv.org/abs/2604.18486}
}
```

---

## 开源协议

本项目遵循 [Apache 2.0 License](LICENSE).

Model weights are built on [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) and the visual tokenizer is from [Emu3.5-VisionTokenizer](https://huggingface.co/BAAI/Emu3.5-VisionTokenizer); 使用时需同步遵守对应原始许可条款.

---

## 致谢

- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) — backbone VLM / 主干视觉语言模型底座
- [Emu3.5](https://github.com/baaivision/Emu3) — IBQ visual tokenizer / 图像量化编码工具
- [AdaThinkDrive](https://github.com/luo-yc17/AdaThinkDrive/tree/main) — NAVSIM CoT annotations / 驾驶推理标注数据集
- [NAVSIM](https://github.com/autonomousvision/navsim), [ROADWork](https://github.com/vita-epfl/roadwork), [Impromptu](https://github.com/Xiaomi-CHI/Impromptu) — evaluation benchmarks / 算法评测基准平台
