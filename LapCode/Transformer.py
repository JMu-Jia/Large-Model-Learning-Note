#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/1/30 16:06
# @Author : JMu
# @ProjectName : 手撕Transformer  # https://blog.csdn.net/xiaoh_7/article/details/140019530

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoder(nn.Module): # 位置编码
    def __init__(self, d_model, max_seq_len=80):    # d_model: 模型维度, max_seq_len: 序列最大长度
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)  # 零矩阵，用于存储位置编码
        for pos in range(max_seq_len):  # 遍历序列
            for i in range(0, d_model, 2):  # 遍历维度
                pe[pos, i] = math.sin(pos / (10000**((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000**((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)    # 增加批次维度，pe: (max_seq_len, d_model)->(1, max_seq_len, d_model)
        self.register_buffer('pe', pe)  # 将pe注册为一个缓冲区，这样它会在模型保存和加载时被保存，但不会被优化器更新。

    def forward(self, x):
        x = x * math.sqrt(self.d_model)  # 放大输入嵌入向量x，以确保嵌入向量的值不会被位置编码淹没
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]    # 将位置编码添加到输入嵌入向量中，其中seq_len 是输入序列的实际长度。
        return x

class MultiHeadAttention(nn.Module):    # 多头注意力机制
    def __init__(self, heads, d_model, dropout=0.1):    # heads: 注意力的头数, d_model: 输入和输出维度
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads  # 每个注意力头的维度
        self.h = heads
        self.q_linear = nn.Linear(d_model, d_model) # 线性层，用于将Q, K, V分别映射到d_model维度
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)  # 防止过拟合
        self.out = nn.Linear(d_model, d_model)  # 将拼接后的多头注意力输出映射回d_model维度

    def attention(self, q, k, v, d_k, mask=None, dropout=None): # q: 查询(Query), k: 键值(Key), v: 值(Value)，前两者用来计算自注意力权重，后者是对输入的编码
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # 计算注意力分数: 通过矩阵乘法计算q和k的点积(计算q和k的相似度)，然后除以sqrt(d_k)进行缩放，以防止梯度消失或爆炸
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)    # 则将掩码中为0的位置对应的score设置为一个非常小的值（如 -1e9），以确保这些位置在softmax后为0
        scores = F.softmax(scores, dim=-1)  # 对分数进行softmax操作，使其成为一个概率分布
        if dropout is not None:
            scores = dropout(scores)    # 应用Dropout层
        output = torch.matmul(scores, v)    # 通过矩阵乘法将注意力分数与值相乘，得到加权的值
        return output

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k).transpose(1, 2)  # 对输入的q、k和v分别进行线性变换，重塑为多头形式，将这些张量进行转置，以便在注意力计算中正确对齐
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)  # 多头注意力的输出进行转置和拼接
        output = self.out(concat)   # 输出通过线性层进行整合
        return output

class FeedForward(nn.Module):   # 全连接前馈层    # 包含两层，两层之间由ReLU作为激活函数
    def __init__(self, d_model, d_ff=2048, dropout=0.1):    # d_model: 输入/输出维度, d_ff: 中间层维度,
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)    # 第一个线性层 d_model->d_ff
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)    # 第二个线性层 d_ff->d_model

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class NormLayer(nn.Module):     # 层正则化
    def __init__(self, d_model, eps=1e-6):  # d_model: 输入和输出维度, eps: 防止除零错误
        super().__init__()
        self.size = d_model
        self.alpha = nn.Parameter(torch.ones(self.size))    # 可学习参数量 alpha, 初始化为1
        self.bias = nn.Parameter(torch.zeros(self.size))    # 可学习参数量 bias, 初始化为0
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

class EncoderLayer(nn.Module):  #每个EncoderLayer包含一个多头注意力机制和一个前馈神经网络，以及相应的归一化层和丢弃层
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = NormLayer(d_model)    # 在多头注意力机制和前馈神经网络之前，对输入进行层归一化
        self.norm_2 = NormLayer(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout) # 多头注意力机制
        self.ff = FeedForward(d_model, dropout=dropout) # 前馈神经网络
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x

class Encoder(nn.Module):   # 将输入序列（例如一段文本）转换成一系列高维特征向量，这些特征向量可以被解码器用来生成输出序列
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = nn.Embedding(vocab_size, d_model)  # 嵌入层
        self.pe = PositionalEncoder(d_model)    # 位置编码器
        self.layers = nn.ModuleList([EncoderLayer(d_model, heads, dropout) for _ in range(N)])  # 编码器层，由N个EncoderLayer组成的列表
        self.norm = NormLayer(d_model)  # 在所有编码器层之后，对输出进行层归一化，以稳定训练过程

    def forward(self, src, mask):
        x = self.embed(src)  # 输入通过嵌入层转化为词嵌入向量
        x = self.pe(x)  # 通过位置编码器添加位置信息
        for layer in self.layers:   # 每个编码器层的输出作为下一个编码器层的输入
            x = layer(x, mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = NormLayer(d_model)
        self.norm_2 = NormLayer(d_model)
        self.norm_3 = NormLayer(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)   # 自注意力机制，自注意力机制允许模型在处理每个位置的输入时，考虑到序列中所有其他位置的信息。
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)   # 编码器-解码器注意力机制，允许解码器在生成每个位置的输出时，考虑到编码器的输出（即源语言的上下文信息）。
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x

class Decoder(nn.Module):   # 解码器，生成输出序列
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = nn.Embedding(vocab_size, d_model)  # 嵌入层，每个索引对应d_model维度
        self.pe = PositionalEncoder(d_model)    # 位置编码器
        self.layers = nn.ModuleList([DecoderLayer(d_model, heads, dropout) for _ in range(N)])  # 解码器层
        self.norm = NormLayer(d_model)  # 归一化层

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, e_outputs, src_mask, trg_mask) # 和Encoder不同之处 每个DecoderLayer包含两个多头注意力机制和一个前馈神经网络，以及相应的归一化层和丢弃层
        return self.norm(x)

class Transformer(nn.Module):   # Transformer主体结构
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads, dropout)
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab)

    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output

