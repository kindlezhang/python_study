import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, dropout = 0.1):
        super(SelfAttention, self).__init__()
        self.dropout = nn.Dropout(dropout) # 对10%的神经元做一个随机失活
        self.softmax = nn.Softmax(dim = -1) # 讲得分转成概率分布，在最后一个维度进行

    def forward(self, Q, K, V, mask = None):
        # x:batch, seq_len, d_model kkkkk
        # batch: 一次送到模型的句子个数，seq_len: 一个句子中的token数量， d_model:embedding向量的维度(一般512)
        # Q, query向量 维度： batch, heads 头数一般为8, seq_len_q 一般来自token数量，但也有例外比如cross attention, d_k
        # K, key向量 维度：batch, heads, seq_len_k, d_k
        # V, value向量 维度： batch, heads, seq_len_v, d_v
        # mask的目的是为了告诉模型哪些位置需要忽略
        d_k = Q.size(-1) # q的最后一维是对每个query向量的维度，代表我们对每个query进行缩放
        # batch,heads,seq_len_q,d_k %*% batch,heads,d_k,seq_len_k -> batch,heads,seq_len_q,seq_len_k
        scores = torch.matmul(Q,K.transpose(-2, -1))/math.sqrt(d_k) # -2和-1维度做一个交换
        # 进行缩放使梯度更稳定
        # 如果提供mask，则通过mask == 0，来找到需要屏蔽的位置。 masked_fill会将这些位置的值改为-inf
        # 经过softmax之后，这些位置的值会约等于0 (被忽略)
        if mask is not None:
            scores = scores.masked_fill(mask==0, float("-inf"))

        # batch, heads, seq_len_q, seq_len_k 对最后一维进行softmax，即对key进行，得到注意力权重矩阵，对每一个query的key权重之和为1
        attn = self.softmax(scores)
        attn = self.dropout(attn) # 对注意力权重进行dropout防止过拟合
        # attn: batch, heads, seq_len_q, seq_len_k; V: batch, heads, seq_len_v, d_v -> batch, heads, seq_len_q, d_v
        out = torch.matmul(attn, V)
        return out, attn
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        # n_heads就是多头注意力的头数 8， d_model embedding的维度 512
        # d_model 需要被 n_heads 整除， 结果为64
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads # 每个头的维度
        self.n_heads = n_heads
        
        # 将输入映射到Q, K, V三个变量， 通过线性映射让模型具有学习能力(前馈神经网络)
        self.W_q = nn.Linear(d_model, d_model) # query的线性映射，唯独不需要改变，翻遍后续的维度拆分
        self.W_k = nn.Linear(d_model, d_model) # key的线性映射
        self.W_v = nn.Linear(d_model, d_model) # value的线性映射
        self.fc = nn.linear(d_model, d_model) # 多头拼接后再映射回原来的d_model, 让模型融合不同头的信息

        self.attention = SelfAttention(dropout) # 使用我们定义好的selfattn
        self.dropout = nn.Dropout(dropout) # 防止过拟合
        self.norm = nn.LayerNorm(d_model) # 用于残差后的归一化

    def forward (self, q, k, v, mask = None):
        batch_size = q.size(0) # 获取batch的大小
        # Q 的维度 batch, seq_len, d_model -> batch, seq_len, self.n_heads, self.d_k -> batch, self.n_heads, seq_len, self.d_k
        # 为了让每个注意力头独立的处理整个序列， 方便后续计算注意力权重
        Q = self.W_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
        K = self.W_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2) # batch, self.nheads, seq_len, self.d_k
        V = self.W_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2) 

        # 计算注意力
        out, attn = self.attention(Q,K,V,mask) # atten为注意力权重, out is the outcome
        # batch, heads, sq_len_q, d_v -> batch, seq_len_q, heads, d_v -> batch, seq_len, d_model
        # contiguous 目的是让tensor在内存中连续存储，避免view的时候报错
        out = out.transpose(1,2).contiguous().view(batch_size, -1, self.n_heads * self.d_k) # out: batch, seq_len, d_model
        # multi concat
        out = self.fc(out) # make output and input same, convenient for concat
        out = self.dropout(out) # training data and drop randomly
        return self.norm(out+q), attn
    
    # 
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(). __init__()

    
