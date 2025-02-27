#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/2/7 10:17
# @Author : JMu
# @ProjectName : 手撕LLaMA https://github.com/wdndev/llama3-from-scratch-zh?tab=readme-ov-file

# 仅使用2层Transformer，所以这里的预测结果和教程不同


from pathlib import Path
import tiktoken # 用于OpenAI模型的快速BPE标记器
from tiktoken.load import load_tiktoken_bpe
import torch
import json # 将python对象编码为json格式输出或存储，以及将json格式对象解码为python对象
import matplotlib.pyplot as plt


# ============================tokenizer（BPE分词器的实现）====================
tokenizer_path = "Meta-Llama-3-8B-Instruct-2layers/tokenizer.model" # 加载分词器模型路径 # 调用的是两层小模型中的BPE分词器
special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",  # end of turn
        ] + [f"<|reserved_special_token_{i}|>" for i in range(5, 256 - 5)]  # 添加特殊字节
mergeable_ranks = load_tiktoken_bpe(tokenizer_path) # 导入本地存储的BPE分词器
tokenizer = tiktoken.Encoding(
    name=Path(tokenizer_path).name,
    pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    mergeable_ranks=mergeable_ranks,
    special_tokens={token: len(mergeable_ranks) + i for i, token in enumerate(special_tokens)},
)

# 测试分词器编码和解码功能
en = tokenizer.encode("hello world!")
de = tokenizer.decode(en)

# ==============================读取模型文件=================================
model = torch.load("Meta-Llama-3-8B-Instruct-2layers/consolidated_2layers.pth") # 加载模型权重    # 调用的是两层小模型
# print(json.dumps(list(model.keys())[:20], indent=4))

with open("Meta-Llama-3-8B-Instruct-2layers/params.json", "r") as f:    # 获取模型配置参数
    config = json.load(f)

dim = config["dim"] # 从配置文件中提取模型参数
n_layers = config["n_layers"]
n_heads = config["n_heads"]
n_kv_heads = config["n_kv_heads"]
vocab_size = config["vocab_size"]
multiple_of = config["multiple_of"]
ffn_dim_multiplier = config["ffn_dim_multiplier"]
norm_eps = config["norm_eps"]
rope_theta = torch.tensor(config["rope_theta"])

# ==============================将文本转化为token=================================
prompt = "the answer to the ultimate question of life, the universe, and everything is "

tokens = [128000] + tokenizer.encode(prompt)    # 编码为token  # 一句prompt->tokens实现分词
# print(tokens)
tokens = torch.tensor(tokens)   # list转tensor

prompt_split_as_tokens = [tokenizer.decode([token.item()]) for token in tokens] # 将每个 token 解码为对应的文本 查看分词是否有问题
# print(prompt_split_as_tokens)

# ==============================将token转换为embedding=================================
# 加载嵌入层并复制权重
embedding_layer = torch.nn.Embedding(vocab_size, dim)   # 根据配置文件模型参数定义Embedding层尺寸
embedding_layer.weight.data.copy_(model["tok_embeddings.weight"])   # 复制权重

# 获取未归一化的 token 嵌入
token_embeddings_unnormalized = embedding_layer(tokens).to(torch.bfloat16)  # tokens->embedding嵌入
print(token_embeddings_unnormalized.shape)

# ==============================定义RMS函数，以归一化embedding=================================
def rms_norm(tensor, norm_weights): # norm_eps用来避免RMS为0和除0情况
    return (tensor * torch.rsqrt(tensor.pow(2).mean(-1, keepdim=True) + norm_eps)) * norm_weights


# ==============================构建第一个Transformer层（以第一层一个头为案例理解）=================================
# token_embeddings = rms_norm(token_embeddings_unnormalized, model["layers.0.attention_norm.weight"]) # 归一化token嵌入
# print(token_embeddings.shape)
#
# print(
#     model["layers.0.attention.wq.weight"].shape,    # [4096, 4096]
#     model["layers.0.attention.wk.weight"].shape,    # [1024, 4096]
#     model["layers.0.attention.wv.weight"].shape,    # [1024, 4096]
#     model["layers.0.attention.wo.weight"].shape     # [4096, 4096]
# )   # 为并行注意力计算，Q, K, V, Output均是合并在一起
#
# # --------------------Q的展开与旋转位置编码--------------
# # 展开query
# q_layer0 = model["layers.0.attention.wq.weight"]
# head_dim = q_layer0.shape[0] // n_heads
# q_layer0 = q_layer0.view(n_heads, head_dim, dim)    # reshape query 权重为[头数，头维度，嵌入维度]
# print(q_layer0.shape)   # [32, 128, 4096]
#
# # 查询第一层第一个Q的权重矩阵
# q_layer0_head0 = q_layer0[0]
#
# # 计算每个token的Q
# q_per_token = torch.matmul(token_embeddings, q_layer0_head0.T)
# print(q_per_token.shape)    # [17, 128]
#
# # 为每个Q生成位置编码 RoPE(Rotary Positional Embeddings) # LLaMA特有功能
# q_per_token_split_into_pairs = q_per_token.float().view(q_per_token.shape[0], -1, 2)
# print(q_per_token_split_into_pairs.shape)   # [17, 64, 2]
#
# # 使用复数点积计算旋转向量
# zero_to_one_split_into_64_parts = torch.tensor(range(64))/64
# freqs = 1.0 / (rope_theta ** zero_to_one_split_into_64_parts)
# freqs_for_each_token = torch.outer(torch.arange(17), freqs)
# freqs_cis = torch.polar(torch.ones_like(freqs_for_each_token), freqs_for_each_token)
#
# # 将Q拆分成对转换为复数
# q_per_token_as_complex_numbers = torch.view_as_complex(q_per_token_split_into_pairs)    # 返回复数张量
# print(q_per_token_as_complex_numbers.shape)     # [17, 64]
# # 然后进行点积以根据位置旋转查询
# q_per_token_as_complex_numbers_rotated = q_per_token_as_complex_numbers * freqs_cis
# print(q_per_token_as_complex_numbers_rotated.shape) # [17, 64]
#
# # 得到旋转向量后通过通过再次将复数看作实数来返回成对的Q
# q_per_token_split_into_pairs_rotated = torch.view_as_real(q_per_token_as_complex_numbers_rotated)   # 返回实数
# print(q_per_token_split_into_pairs_rotated.shape)   # [17, 64, 2]
#
# # 将旋转对合并为新的Q
# q_per_token_rotated = q_per_token_split_into_pairs_rotated.view(q_per_token.shape)
# print(q_per_token_rotated.shape)    # [17, 128]
#
# # --------------------K的展开与旋转位置编码--------------
# # K的生成和旋转编码几乎与Q一模一样，
# # keys 生成的 key 向量的维度也是 128
# # keys 的权重只有 query 的 1/4，因为 keys 的权重在 4 个头之间共享，以减少计算量
# # keys 也像 query 一样被旋转以添加位置信息，其原因相同
#
# k_layer0 = model["layers.0.attention.wk.weight"]
# k_layer0 = k_layer0.view(n_kv_heads, k_layer0.shape[0] // n_kv_heads, dim)
# print(k_layer0.shape)   # [8, 128, 4096]
#
# k_layer0_head0 = k_layer0[0]
# print(k_layer0_head0.shape) # [128, 4096]
#
# k_per_token = torch.matmul(token_embeddings, k_layer0_head0.T)
# print(k_per_token.shape)    # [17, 128]
#
# k_per_token_split_into_pairs = k_per_token.float().view(k_per_token.shape[0], -1, 2)
# print(k_per_token_split_into_pairs.shape)   # [17, 64, 2]
#
# k_per_token_as_complex_numbers = torch.view_as_complex(k_per_token_split_into_pairs)
# print(k_per_token_as_complex_numbers.shape)   # [17, 64]
#
# k_per_token_split_into_pairs_rotated = torch.view_as_real(k_per_token_as_complex_numbers * freqs_cis)
# print(k_per_token_split_into_pairs_rotated.shape)   # [17, 64, 2]
#
# k_per_token_rotated = k_per_token_split_into_pairs_rotated.view(k_per_token.shape)
# print(k_per_token_rotated.shape)    # [17, 128]
#
# # -------------旋转后的Q和K相乘-------------------
# # 这个分数描述了每个token的query与每个token的key的相关度。这就是自注意力
# qk_per_token = torch.matmul(q_per_token_rotated, k_per_token_rotated.T)/(head_dim)**0.5
# print(qk_per_token.shape)   # [17, 17] 17是token的数量
#
# # ---------------------屏蔽未来的Q和K分数----------------------
# # 在llama3的训练过程中，未来的token的QK分数被屏蔽。
# # 为什么？因为在训练过程中，只学习使用过去的token来预测token 。
# # 因此，在推理过程中，将未来的token设置为零。
#
# mask = torch.full((len(tokens), len(tokens)), float("-inf"), device=tokens.device)  # 创建个全是-INF的MASK矩阵
# mask = torch.triu(mask, diagonal=1) # 返回上三角元素，其余位置为0
# print(mask)
#
# qk_per_token_after_masking = qk_per_token + mask    # 屏蔽QK的上三角
#
# qk_per_token_after_masking_after_softmax = torch.nn.functional.softmax(qk_per_token_after_masking, dim=1).to(torch.bfloat16)    # Softmax
#
# # -----------------V（注意力机制的最后部分）--------------------
# v_layer0 = model["layers.0.attention.wv.weight"]
# v_layer0 = v_layer0.view(n_kv_heads, v_layer0.shape[0] // n_kv_heads, dim)  # 并行进行切分
# print(v_layer0.shape)   # [8, 128, 4096]
#
# v_layer0_head0 = v_layer0[0]    # 读取第一个头
# print(v_layer0_head0.shape)    # [128, 4096]
#
# # 使用V权重获取每个token的注意力值
# v_per_token = torch.matmul(token_embeddings, v_layer0_head0.T)
# print(v_per_token.shape)    # [17, 128]
#
# # 注意力   # Softmax(QK)V
# qkv_attention = torch.matmul(qk_per_token_after_masking_after_softmax, v_per_token)
# print(qkv_attention.shape)  # [17, 128]

# ==================多头注意力机制（将上面的第一层第一个头的计算重新整合为第一层所有头的计算）=======================
# token_embeddings = rms_norm(token_embeddings_unnormalized, model["layers.0.attention_norm.weight"]) # 归一化token嵌入
#
# # 根据token数使用复数点积计算旋转向量
# zero_to_one_split_into_64_parts = torch.tensor(range(64))/64
# freqs = 1.0 / (rope_theta ** zero_to_one_split_into_64_parts)
# freqs_for_each_token = torch.outer(torch.arange(17), freqs)
# freqs_cis = torch.polar(torch.ones_like(freqs_for_each_token), freqs_for_each_token)
#
# # 导入第一层的Q,K,V
# q_layer0 = model["layers.0.attention.wq.weight"]
# k_layer0 = model["layers.0.attention.wk.weight"]
# v_layer0 = model["layers.0.attention.wv.weight"]
#
# # 将并行保存的Q,K,V按照头进行解耦保存
# q_layer0 = q_layer0.view(n_heads, q_layer0.shape[0] // n_heads, dim)
# k_layer0 = k_layer0.view(n_kv_heads, k_layer0.shape[0] // n_kv_heads, dim)
# v_layer0 = v_layer0.view(n_kv_heads, v_layer0.shape[0] // n_kv_heads, dim)
#
#
# qkv_attention_store = []
# for head in range(n_heads):
#     q_layer0_head = q_layer0[head]
#     k_layer0_head = k_layer0[head//4] # key weights are shared across 4 heads
#     v_layer0_head = v_layer0[head//4] # value weights are shared across 4 heads
#     q_per_token = torch.matmul(token_embeddings, q_layer0_head.T)
#     k_per_token = torch.matmul(token_embeddings, k_layer0_head.T)
#     v_per_token = torch.matmul(token_embeddings, v_layer0_head.T)
#
#     q_per_token_split_into_pairs = q_per_token.float().view(q_per_token.shape[0], -1, 2)
#     q_per_token_as_complex_numbers = torch.view_as_complex(q_per_token_split_into_pairs)
#     q_per_token_split_into_pairs_rotated = torch.view_as_real(q_per_token_as_complex_numbers * freqs_cis[:len(tokens)])
#     q_per_token_rotated = q_per_token_split_into_pairs_rotated.view(q_per_token.shape)
#
#     k_per_token_split_into_pairs = k_per_token.float().view(k_per_token.shape[0], -1, 2)
#     k_per_token_as_complex_numbers = torch.view_as_complex(k_per_token_split_into_pairs)
#     k_per_token_split_into_pairs_rotated = torch.view_as_real(k_per_token_as_complex_numbers * freqs_cis[:len(tokens)])
#     k_per_token_rotated = k_per_token_split_into_pairs_rotated.view(k_per_token.shape)
#
#     qk_per_token = torch.matmul(q_per_token_rotated, k_per_token_rotated.T)/(128)**0.5
#     mask = torch.full((len(tokens), len(tokens)), float("-inf"), device=tokens.device)
#     mask = torch.triu(mask, diagonal=1)
#     qk_per_token_after_masking = qk_per_token + mask
#     qk_per_token_after_masking_after_softmax = torch.nn.functional.softmax(qk_per_token_after_masking, dim=1).to(torch.bfloat16)
#     qkv_attention = torch.matmul(qk_per_token_after_masking_after_softmax, v_per_token)
#     qkv_attention_store.append(qkv_attention)
#
# print(len(qkv_attention_store))    # 32
#
# # 将32个头的注意力分数合并成一个大矩阵
# stacked_qkv_attention = torch.cat(qkv_attention_store, dim=-1)
# print(stacked_qkv_attention.shape)
#
# # 将权重矩阵相乘
# w_layer0 = model["layers.0.attention.wo.weight"]
# print(w_layer0.shape)   # [4096, 4096]
#
# embedding_delta = torch.matmul(stacked_qkv_attention, w_layer0.T)
# print(embedding_delta.shape)   # [17, 4096]
#
# # embedding的变化更新到原始的embedding中
# embedding_after_edit = token_embeddings_unnormalized + embedding_delta
# print(embedding_after_edit.shape)   # [17, 4096]
#
# # 新的embedding进行归一化
# embedding_after_edit_normalized = rms_norm(embedding_after_edit, model["layers.0.ffn_norm.weight"])
# print(embedding_after_edit_normalized.shape)   # [17, 4096]
#
# # LLaMA中使用SwiGLU前馈神经网络
# w1 = model["layers.0.feed_forward.w1.weight"]
# w2 = model["layers.0.feed_forward.w2.weight"]
# w3 = model["layers.0.feed_forward.w3.weight"]
# output_after_feedforward = torch.matmul(torch.functional.F.silu(torch.matmul(embedding_after_edit_normalized, w1.T)) * torch.matmul(embedding_after_edit_normalized, w3.T), w2.T)
# print(output_after_feedforward.shape)   # [17, 4096]
#
# # 前向和后向更新后的embedding整合在一起
# layer_0_embedding = embedding_after_edit+output_after_feedforward
# print(layer_0_embedding.shape)   # [17, 4096]

# ==================每层每个头代码整合在一起====================================
# 根据token数使用复数点积计算旋转向量  # 可替换为下列函数
# zero_to_one_split_into_64_parts = torch.tensor(range(64))/64
# freqs = 1.0 / (rope_theta ** zero_to_one_split_into_64_parts)
# freqs_for_each_token = torch.outer(torch.arange(17), freqs)
# freqs_cis= torch.polar(torch.ones_like(freqs_for_each_token), freqs_for_each_token)

def precompute_freqs_cis(fre_dim, tokens_len, theta):
    freqs = 1.0 / (theta ** (torch.tensor(range(fre_dim))/fre_dim)) # type: ignore
    freqs_for_each_token = torch.outer(torch.arange(tokens_len), freqs)  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs_for_each_token), freqs_for_each_token)  # complex64
    return freqs_cis

freqs_cis = precompute_freqs_cis(dim//n_heads//2, len(tokens), rope_theta)  # 根据tokens计算freqs_cis

final_embedding = token_embeddings_unnormalized
for layer in range(n_layers):
    qkv_attention_store = []
    layer_embedding_norm = rms_norm(final_embedding, model[f"layers.{layer}.attention_norm.weight"])
    q_layer = model[f"layers.{layer}.attention.wq.weight"]
    q_layer = q_layer.view(n_heads, q_layer.shape[0] // n_heads, dim)
    k_layer = model[f"layers.{layer}.attention.wk.weight"]
    k_layer = k_layer.view(n_kv_heads, k_layer.shape[0] // n_kv_heads, dim)
    v_layer = model[f"layers.{layer}.attention.wv.weight"]
    v_layer = v_layer.view(n_kv_heads, v_layer.shape[0] // n_kv_heads, dim)
    w_layer = model[f"layers.{layer}.attention.wo.weight"]
    for head in range(n_heads):
        q_layer_head = q_layer[head]
        k_layer_head = k_layer[head//4]
        v_layer_head = v_layer[head//4]
        q_per_token = torch.matmul(layer_embedding_norm, q_layer_head.T)
        k_per_token = torch.matmul(layer_embedding_norm, k_layer_head.T)
        v_per_token = torch.matmul(layer_embedding_norm, v_layer_head.T)
        q_per_token_split_into_pairs = q_per_token.float().view(q_per_token.shape[0], -1, 2)
        q_per_token_as_complex_numbers = torch.view_as_complex(q_per_token_split_into_pairs)
        q_per_token_split_into_pairs_rotated = torch.view_as_real(q_per_token_as_complex_numbers * freqs_cis)
        q_per_token_rotated = q_per_token_split_into_pairs_rotated.view(q_per_token.shape)
        k_per_token_split_into_pairs = k_per_token.float().view(k_per_token.shape[0], -1, 2)
        k_per_token_as_complex_numbers = torch.view_as_complex(k_per_token_split_into_pairs)
        k_per_token_split_into_pairs_rotated = torch.view_as_real(k_per_token_as_complex_numbers * freqs_cis)
        k_per_token_rotated = k_per_token_split_into_pairs_rotated.view(k_per_token.shape)
        qk_per_token = torch.matmul(q_per_token_rotated, k_per_token_rotated.T)/(128)**0.5
        mask = torch.full((len(token_embeddings_unnormalized), len(token_embeddings_unnormalized)), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        qk_per_token_after_masking = qk_per_token + mask
        qk_per_token_after_masking_after_softmax = torch.nn.functional.softmax(qk_per_token_after_masking, dim=1).to(torch.bfloat16)
        qkv_attention = torch.matmul(qk_per_token_after_masking_after_softmax, v_per_token)
        qkv_attention_store.append(qkv_attention)

    stacked_qkv_attention = torch.cat(qkv_attention_store, dim=-1)
    w_layer = model[f"layers.{layer}.attention.wo.weight"]
    embedding_delta = torch.matmul(stacked_qkv_attention, w_layer.T)
    embedding_after_edit = final_embedding + embedding_delta
    embedding_after_edit_normalized = rms_norm(embedding_after_edit, model[f"layers.{layer}.ffn_norm.weight"])
    w1 = model[f"layers.{layer}.feed_forward.w1.weight"]
    w2 = model[f"layers.{layer}.feed_forward.w2.weight"]
    w3 = model[f"layers.{layer}.feed_forward.w3.weight"]
    output_after_feedforward = torch.matmul(torch.functional.F.silu(torch.matmul(embedding_after_edit_normalized, w1.T)) * torch.matmul(embedding_after_edit_normalized, w3.T), w2.T)
    final_embedding = embedding_after_edit+output_after_feedforward

# =====================输出========================

final_embedding = rms_norm(final_embedding, model["norm.weight"])   # 输出embedding进行归一化
print(final_embedding.shape)    # [17, 4096]

logits = torch.matmul(final_embedding[-1], model["output.weight"].T)    # 使用最后一个token的embedding预测下一个值
next_token = torch.argmax(logits, dim=-1)
print(next_token)

out = tokenizer.decode([next_token.item()]) # 对预测的token进行解码
print(out)




