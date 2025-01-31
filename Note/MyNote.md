# Large Model Learning Note

## 手撕大模型

### Transformer

- Transformer 完全基于**注意力机制**，没有递归或卷积结构，所以不具备处理序列信息的能力，其通过**位置编码**处理序列。
- Transformer 是由两种模块组合构建的模块化网络结构。两种模块分别为：（1）**注意力**（Attention）模块；（2）**全连接前馈**（Fully-Connected Feedforward）模块。其中，
自注意力模块由自注意力层（Self-AttentionLayer）、残差连接（ResidualConnections）和层正则化（LayerNormalization）组成。全连接前馈模块由全连接前馈层（占总参数2/3），残差连接和层正则化组成。
![](Fig/Transformer模块结构.png)
- Feedforward 用于增强模型非线性能力 LayerNormalization 用以加速神经网络训练过程并取得更好的泛化性能。
- Transformer 结构示意图（以 Encoder 和 Decoder 为主）。![](Fig/Transformer结构示意图.jpg)原始的 Transformer 采用 Encoder-Decoder 架构，
其包含 Encoder 和 Decoder 两部分。这两部分都是由自注意力模块和全连接前馈模块重复连接构建而成。
其中，**Encoder 部分由六个级联的 encoder layer 组成，每个 encoder layer 包含一个注意力模块和一个全连接前馈模块**。
其中的注意力模块为自注意力模块（query，key，value 的输入是相同的）。** Decoder 部分由六个级联的
decoder layer 组成，每个 decoder layer 包含两个注意力模块和一个全连接前馈模块**。
其中，**第一个注意力模块为自注意力模块，第二个注意力模块为交叉注意力模块**
（query，key，value 的输入不同）。Decoder 中第一个 decoder layer 的自注意力模块的输入模型的输出。其后的decoder layer的自注意力模块的输入为上一个 decoder
layer 的输出。Decoder 交叉注意力模块的输入分别是自注意力模块的输出（query）和最后一个 encoder layer 的输出（key，value）。
- [Code](../Code/Transformer.py)中 class 调用结构示意图。![](Fig/Transformer代码class层级.jpg)
- **优缺点：** 相较于 RNN 模型串行的循环迭代模式，Transformer 并行输入的特性，使其容易进行并行计算。但是，Transformer 并行输入的范式也导致网络模型的规模随输入序列长度的增长而平方次增长。这为应用 Transformer 处理长序列带来挑战。
---

### WebGPT
手撕AutoGPT和LangChain+GPT
https://blog.csdn.net/bobwww123/article/details/138948884
https://bbs.csdn.net/topics/618157820

---

### LLaMa

---

### DeepSeek
https://zhuanlan.zhihu.com/p/14953285242（论文解析）

**整体架构**：MLA + DeepSeekMoE + MTP

**训练架构**：精细化分块量化 + 提升乘累精度 + 在线量化” 的 FP8 

- **专家混合**（Mixture-of-Experts, MoE）架构自然语言生成模型

- **多头潜在注意力**（Multi-head Latent Attention, MLA）机制和**DeepSeekMoE架构**
  
- **多token预测** （Multi-token Prediction，MTP）

---

### LLM、VLM基本框架

- Encoder-Only: BERT; Encoder-Decoder: T5; Decoder-Only: GPT-3; 非Transformer结构（需要一个思维导图）
---

## 微调技术 PEFT

### LoRA

---

### P-Tuning

---

## 知识检索增强

--- 

### RAG

---


### 向量数据库

---


### agent

---

### embedding

---


## 大模型工具


### DevOps工具

---

###  langchain

---
