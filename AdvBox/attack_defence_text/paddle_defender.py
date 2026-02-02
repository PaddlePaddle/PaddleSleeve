import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
import math


class PinyinEmbedding(nn.Layer):
    def __init__(self, hidden_dim=128, pinyin_out_dim=256, dropout_prob=0.1, **kwargs):
        super(PinyinEmbedding, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.pinyin_out_dim = pinyin_out_dim

        # 用于将 one-hot 编码转为 8 * hidden_dim 向量
        self.embedding = nn.Embedding(31, hidden_dim)

        # 1D 卷积，输入通道数为 hidden_dim，输出通道数为 pinyin_out_dim，卷积核大小为 2
        self.conv = nn.Conv1D(
            in_channels=hidden_dim,
            out_channels=self.pinyin_out_dim,
            kernel_size=2,
            stride=1,
            padding='valid'
        )

    def forward(self, x, pinyin_mask):
        embed = self.embedding(x)  # (bs, sentence_length, pinyin_locs, hidden_dim)

        # 调整维度顺序以适应 Conv1D 的输入格式 (batch_size, channels, length)
        embed_shape = embed.shape
        bs, sentence_length, pinyin_locs, embed_size = embed_shape

        # 重塑为 (bs * sentence_length, pinyin_locs, embed_size)
        view_embed = paddle.reshape(embed, [-1, pinyin_locs, embed_size])
        # 调整为 Conv1D 需要的格式: (batch_size, channels, length)
        # 注意: Paddle Conv1D 期望输入形状为 (batch_size, channels, length)
        view_embed = paddle.transpose(view_embed, [0, 2, 1])  # (bs*sentence_length, embed_size, pinyin_locs)

        # 扩展掩码以匹配嵌入维度
        expanded_mask = paddle.unsqueeze(pinyin_mask, axis=-1)
        expanded_mask = paddle.tile(expanded_mask, [1, 1, 1, self.hidden_dim])

        # 确保掩码与嵌入形状匹配
        expanded_mask = paddle.reshape(expanded_mask, [-1, pinyin_locs, embed_size])
        expanded_mask = paddle.transpose(expanded_mask, [0, 2, 1])

        # 应用掩码
        view_embed = view_embed * expanded_mask

        # 卷积和池化
        pinyin_conv = self.conv(view_embed)  # [(bs*sentence_length), pinyin_out_dim, pinyin_locs-2+1]
        pinyin_conv = paddle.transpose(pinyin_conv, [0, 2, 1])  # [(bs*sentence_length), pinyin_locs-1, pinyin_out_dim]
        pinyin_embed = paddle.max(pinyin_conv, axis=1)  # [(bs*sentence_length), pinyin_out_dim]

        # 重塑回原始批次形状
        return paddle.reshape(pinyin_embed, [bs, sentence_length, self.pinyin_out_dim])


class MultiHeadAttention(nn.Layer):
    def __init__(self, embed_dim, num_heads, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]

        # 线性变换并重塑为多头
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)

        # 重塑为 (batch_size, seq_len, num_heads, head_dim)
        q = paddle.reshape(q, [batch_size, -1, self.num_heads, self.head_dim])
        k = paddle.reshape(k, [batch_size, -1, self.num_heads, self.head_dim])
        v = paddle.reshape(v, [batch_size, -1, self.num_heads, self.head_dim])

        # 转置为 (batch_size, num_heads, seq_len, head_dim)
        q = paddle.transpose(q, [0, 2, 1, 3])
        k = paddle.transpose(k, [0, 2, 1, 3])
        v = paddle.transpose(v, [0, 2, 1, 3])

        # 计算注意力分数
        scores = paddle.matmul(q, paddle.transpose(k, [0, 1, 3, 2])) / math.sqrt(self.head_dim)

        if mask is not None:
            # 扩展掩码以匹配注意力分数形状
            expanded_mask = paddle.unsqueeze(paddle.unsqueeze(mask, axis=1), axis=2)
            expanded_mask = paddle.tile(expanded_mask, [1, self.num_heads, scores.shape[2], 1])
            # 将 mask 转换为 bool 类型
            mask_bool = (expanded_mask == 0)
            scores = paddle.where(mask_bool, paddle.full_like(scores, -1e9), scores)

        attention = F.softmax(scores, axis=-1)
        x = paddle.matmul(attention, v)

        # 转置并重塑回原始形状
        x = paddle.transpose(x, [0, 2, 1, 3])
        x = paddle.reshape(x, [batch_size, -1, self.embed_dim])

        return self.out_linear(x)


class FeedForward(nn.Layer):
    def __init__(self, embed_dim, ff_dim, **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        return self.linear2(x)


class EncoderLayer(nn.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.feed_forward = FeedForward(embed_dim, ff_dim)
        self.norm1 = nn.LayerNorm(embed_dim, epsilon=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim, epsilon=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, training=False):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class Encoder(nn.Layer):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ff_dim, dropout, max_len, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        self.layers = nn.LayerList([EncoderLayer(embed_dim, num_heads, ff_dim, dropout)
                                    for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, training=False):
        seq_len = x.shape[1]
        pos = paddle.tile(paddle.unsqueeze(paddle.arange(0, seq_len), 0), [x.shape[0], 1])

        x = self.embedding(x) + self.pos_embedding(pos)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask, training=training)

        return x


class TransformerClassifier(nn.Layer):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ff_dim, num_classes, dropout, max_len, **kwargs):
        super(TransformerClassifier, self).__init__(**kwargs)
        self.encoder = Encoder(vocab_size, embed_dim, num_heads, num_layers, ff_dim, dropout, max_len)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, training=False):
        encoder_output = self.encoder(input_ids, attention_mask, training=training)
        pooled_output = paddle.mean(encoder_output, axis=1)
        output = self.dropout(pooled_output)
        return self.fc(output)


class EncoderImg(nn.Layer):
    def __init__(self, img_size, embed_dim, num_heads, num_layers, ff_dim, dropout, max_len, **kwargs):
        super(EncoderImg, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.img_embed = nn.Linear(img_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        self.layers = nn.LayerList([EncoderLayer(embed_dim, num_heads, ff_dim, dropout)
                                    for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, training=False):
        seq_len = x.shape[1]
        pos = paddle.tile(paddle.unsqueeze(paddle.arange(0, seq_len), 0), [x.shape[0], 1])

        x = self.img_embed(x) + self.pos_embedding(pos)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask, training=training)

        return x


class TransformerClassifierImg(nn.Layer):
    def __init__(self, img_size, embed_dim, num_heads, num_layers, ff_dim, num_classes, dropout, max_len, **kwargs):
        super(TransformerClassifierImg, self).__init__(**kwargs)
        self.encoder = EncoderImg(img_size, embed_dim, num_heads, num_layers, ff_dim, dropout, max_len)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, training=False):
        encoder_output = self.encoder(input_ids, attention_mask, training=training)
        pooled_output = paddle.mean(encoder_output, axis=1)
        output = self.dropout(pooled_output)
        return self.fc(output)


class EncoderPinyin(nn.Layer):
    def __init__(self, pinyin_hid_dim, embed_dim, num_heads, num_layers, ff_dim, dropout, max_len, **kwargs):
        super(EncoderPinyin, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.embedding = PinyinEmbedding(pinyin_hid_dim, embed_dim)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        self.layers = nn.LayerList([EncoderLayer(embed_dim, num_heads, ff_dim, dropout)
                                    for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pinyin_mask, mask, training=False):
        seq_len = x.shape[1]
        pos = paddle.tile(paddle.unsqueeze(paddle.arange(0, seq_len), 0), [x.shape[0], 1])

        x = self.embedding(x, pinyin_mask) + self.pos_embedding(pos)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask, training=training)

        return x


class TransformerClassifierPinyin(nn.Layer):
    def __init__(self, pinyin_hid_dim, embed_dim, num_heads, num_layers, ff_dim, num_classes, dropout, max_len,
                 **kwargs):
        super(TransformerClassifierPinyin, self).__init__(**kwargs)
        self.encoder = EncoderPinyin(pinyin_hid_dim, embed_dim, num_heads, num_layers, ff_dim, dropout, max_len)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, pinyin_mask, attention_mask, training=False):
        encoder_output = self.encoder(input_ids, pinyin_mask, attention_mask, training=training)
        pooled_output = paddle.mean(encoder_output, axis=1)
        output = self.dropout(pooled_output)
        return self.fc(output)


class PolicyBlock(nn.Layer):
    def __init__(self, vocab_size, img_size, embedding_dim, hidden_dim, max_len, dropout, num_action=3, **kwargs):
        super(PolicyBlock, self).__init__(**kwargs)
        self.char_embed = nn.Embedding(vocab_size, embedding_dim)
        self.pinyin_embed = PinyinEmbedding(pinyin_out_dim=embedding_dim)
        self.glyph_embed = nn.Linear(img_size, embedding_dim)
        self.pos_embed = nn.Embedding(max_len, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 3, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, embedding_dim)
        self.decision = nn.Linear(embedding_dim, num_action)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_char, x_glyph, x_pinyin, pinyin_mask, training=False):

        x_char = self.char_embed(x_char)
        x_glyph = self.glyph_embed(x_glyph)

        x_pinyin = self.pinyin_embed(x_pinyin, pinyin_mask)

        # 连接三个嵌入
        x_cat = paddle.concat([x_char, x_glyph, x_pinyin], axis=2)

        seq_len = x_cat.shape[1]
        pos = paddle.tile(paddle.unsqueeze(paddle.arange(0, seq_len), 0), [x_cat.shape[0], 1])

        x = self.fc1(x_cat) + self.pos_embed(pos)
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        output = paddle.mean(x, axis=1)
        output = self.dropout(output)
        output = self.decision(output)

        return output


class Defender(nn.Layer):
    def __init__(self, vocab_size, img_size, embedding_dim, hidden_dim, max_len, ff_dim, num_classes, num_heads,
                 num_layers, num_action=3, dropout=0.1, pinyin_hid_dim=128, **kwargs):
        super(Defender, self).__init__(**kwargs)
        self.policy = PolicyBlock(vocab_size, img_size, embedding_dim, hidden_dim, max_len, dropout)
        self.char_net = TransformerClassifier(vocab_size, embedding_dim, num_heads, num_layers, ff_dim, num_classes,
                                              dropout, max_len)
        self.glyph_net = TransformerClassifierImg(img_size, embedding_dim, num_heads, num_layers, ff_dim, num_classes,
                                                  dropout, max_len)
        self.pronunciation_net = TransformerClassifierPinyin(pinyin_hid_dim, embedding_dim, num_heads, num_layers,
                                                             ff_dim, num_classes, dropout, max_len)

    def forward(self, x_char, x_glyph, x_pinyin, pinyin_mask, mask, training=False):
        action = self.policy(x_char, x_glyph, x_pinyin, pinyin_mask, training=training)
        prob_char = self.char_net(x_char, mask, training=training)
        prob_glyph = self.glyph_net(x_glyph, mask, training=training)
        prob_pronunciation = self.pronunciation_net(x_pinyin, pinyin_mask, mask, training=training)

        # 对动作进行加权
        weighted_prob_char = prob_char * paddle.unsqueeze(action[:, 0], axis=1)  # (batch_size, 2)
        weighted_prob_glyph = prob_glyph * paddle.unsqueeze(action[:, 1], axis=1)  # (batch_size, 2)
        weighted_prob_pronunciation = prob_pronunciation * paddle.unsqueeze(action[:, 2], axis=1)  # (batch_size, 2)

        # 对加权后的向量进行逐元素求和
        x = weighted_prob_char + weighted_prob_glyph + weighted_prob_pronunciation  # (batch_size, 2)

        return x


defender = Defender(34765, 4096, 256, 128, 512, 1024, 2, 8,
                    6, num_action=3, dropout=0.1, pinyin_hid_dim=128)

# 测试代码
if __name__ == "__main__":
    # 创建测试数据
    batch_size = 2
    seq_len = 10
    pinyin_locs = 8

    # 创建输入数据
    x_char = paddle.randint(0, 34765, [batch_size, seq_len])
    x_glyph = paddle.randn([batch_size, seq_len, 4096])
    x_pinyin = paddle.randint(0, 30, [batch_size, seq_len, pinyin_locs])
    pinyin_mask = paddle.ones([batch_size, seq_len, pinyin_locs])
    mask = paddle.ones([batch_size, seq_len])

    # 前向传播
    output = defender(x_char, x_glyph, x_pinyin, pinyin_mask, mask)
    print(f"Output shape: {output.shape}")
    print(f"Defender model created successfully!")