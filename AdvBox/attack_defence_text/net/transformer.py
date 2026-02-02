import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math



class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        #mask = mask.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        #mask = mask.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2) 

        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            expanded_mask = mask.unsqueeze(1).unsqueeze(2).expand(-1, scores.size(1), scores.size(2), -1)
            scores = scores.masked_fill(expanded_mask == 0, -1e9)
        attention = torch.softmax(scores, dim=-1)

        x = torch.matmul(attention, v).transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        x = self.out_linear(x)
        return x

# 前馈神经网络
class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x

# 编码器层
class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.feed_forward = FeedForward(embed_dim, ff_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

# 编码器
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ff_dim, dropout, max_len):
        super(Encoder, self).__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        self.layers = nn.ModuleList([EncoderLayer(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        seq_len = x.size(1)
        pos = torch.arange(0, seq_len).unsqueeze(0).repeat(x.size(0), 1).to(x.device)
        x = self.embedding(x) + self.pos_embedding(pos)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask)
        return x

# Transformer模型
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ff_dim, num_classes, dropout, max_len):
        super(TransformerClassifier, self).__init__()
        self.encoder = Encoder(vocab_size, embed_dim, num_heads, num_layers, ff_dim, dropout, max_len)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask): 
        encoder_output = self.encoder(input_ids, attention_mask)
        pooled_output = encoder_output.mean(dim=1)
        output = self.dropout(pooled_output)
        return self.fc(output)
