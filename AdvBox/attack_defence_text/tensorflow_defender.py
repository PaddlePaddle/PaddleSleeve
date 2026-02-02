import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import math


class PinyinEmbedding(tf.keras.layers.Layer):
    def __init__(self, hidden_dim=128, pinyin_out_dim=256, dropout_prob=0.1, **kwargs):
        super(PinyinEmbedding, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.pinyin_out_dim = pinyin_out_dim

        # 用于将 one-hot 编码转为 8 * hidden_dim 向量
        self.embedding = layers.Embedding(31, hidden_dim)

        # 1D 卷积，输入通道数为 hidden_dim，输出通道数为 pinyin_out_dim，卷积核大小为 2
        self.conv = layers.Conv1D(
            filters=self.pinyin_out_dim,
            kernel_size=2,
            strides=1,
            padding='valid'
        )

    def call(self, x, pinyin_mask):
        embed = self.embedding(x)  # (bs, sentence_length, pinyin_locs, hidden_dim)

        # 扩展掩码以匹配嵌入维度
        expanded_mask = tf.expand_dims(pinyin_mask, axis=-1)
        expanded_mask = tf.tile(expanded_mask, [1, 1, 1, self.hidden_dim])
        embed = embed * expanded_mask

        # 获取形状
        bs, sentence_length, pinyin_locs, embed_size = tf.shape(embed)[0], tf.shape(embed)[1], tf.shape(embed)[2], \
            tf.shape(embed)[3]

        # 重塑以进行卷积操作
        view_embed = tf.reshape(embed, [-1, pinyin_locs, embed_size])  # [(bs*sentence_length), pinyin_locs, embed_size]

        # 卷积和池化
        pinyin_conv = self.conv(view_embed)  # [(bs*sentence_length), pinyin_locs-2+1, pinyin_out_dim]
        pinyin_embed = tf.reduce_max(pinyin_conv, axis=1)  # [(bs*sentence_length), pinyin_out_dim]

        # 重塑回原始批次形状
        return tf.reshape(pinyin_embed, [bs, sentence_length, self.pinyin_out_dim])


class MultiHeadAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_linear = layers.Dense(embed_dim)
        self.k_linear = layers.Dense(embed_dim)
        self.v_linear = layers.Dense(embed_dim)
        self.out_linear = layers.Dense(embed_dim)

    def call(self, q, k, v, mask=None):
        batch_size = tf.shape(q)[0]

        # 线性变换并重塑为多头
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)

        # 重塑为 (batch_size, seq_len, num_heads, head_dim)
        q = tf.reshape(q, [batch_size, -1, self.num_heads, self.head_dim])
        k = tf.reshape(k, [batch_size, -1, self.num_heads, self.head_dim])
        v = tf.reshape(v, [batch_size, -1, self.num_heads, self.head_dim])

        # 转置为 (batch_size, num_heads, seq_len, head_dim)
        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])

        # 计算注意力分数
        scores = tf.matmul(q, k, transpose_b=True) / math.sqrt(self.head_dim)

        if mask is not None:
            # 扩展掩码以匹配注意力分数形状
            expanded_mask = tf.expand_dims(tf.expand_dims(mask, axis=1), axis=2)
            expanded_mask = tf.tile(expanded_mask, [1, self.num_heads, tf.shape(scores)[2], 1])
            scores = tf.where(expanded_mask == 0, -1e9, scores)

        attention = tf.nn.softmax(scores, axis=-1)
        x = tf.matmul(attention, v)

        # 转置并重塑回原始形状
        x = tf.transpose(x, [0, 2, 1, 3])
        x = tf.reshape(x, [batch_size, -1, self.embed_dim])

        return self.out_linear(x)


class FeedForward(layers.Layer):
    def __init__(self, embed_dim, ff_dim, **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.linear1 = layers.Dense(ff_dim)
        self.linear2 = layers.Dense(embed_dim)

    def call(self, x):
        x = self.linear1(x)
        x = tf.nn.relu(x)
        return self.linear2(x)


class EncoderLayer(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.feed_forward = FeedForward(embed_dim, ff_dim)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(dropout)

    def call(self, x, mask, training=False):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output, training=training))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output, training=training))
        return x


class Encoder(layers.Layer):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ff_dim, dropout, max_len, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.embedding = layers.Embedding(vocab_size, embed_dim)
        self.pos_embedding = layers.Embedding(max_len, embed_dim)
        self.layers = [EncoderLayer(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        self.dropout = layers.Dropout(dropout)

    def call(self, x, mask, training=False):
        seq_len = tf.shape(x)[1]
        pos = tf.tile(tf.expand_dims(tf.range(0, seq_len), 0), [tf.shape(x)[0], 1])

        x = self.embedding(x) + self.pos_embedding(pos)
        x = self.dropout(x, training=training)

        for layer in self.layers:
            x = layer(x, mask, training=training)

        return x


class TransformerClassifier(Model):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ff_dim, num_classes, dropout, max_len, **kwargs):
        super(TransformerClassifier, self).__init__(**kwargs)
        self.encoder = Encoder(vocab_size, embed_dim, num_heads, num_layers, ff_dim, dropout, max_len)
        self.fc = layers.Dense(num_classes)
        self.dropout = layers.Dropout(dropout)

    def call(self, input_ids, attention_mask, training=False):
        encoder_output = self.encoder(input_ids, attention_mask, training=training)
        pooled_output = tf.reduce_mean(encoder_output, axis=1)
        output = self.dropout(pooled_output, training=training)
        return self.fc(output)


class EncoderImg(layers.Layer):
    def __init__(self, img_size, embed_dim, num_heads, num_layers, ff_dim, dropout, max_len, **kwargs):
        super(EncoderImg, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.img_embed = layers.Dense(embed_dim)
        self.pos_embedding = layers.Embedding(max_len, embed_dim)
        self.layers = [EncoderLayer(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        self.dropout = layers.Dropout(dropout)

    def call(self, x, mask, training=False):
        seq_len = tf.shape(x)[1]
        pos = tf.tile(tf.expand_dims(tf.range(0, seq_len), 0), [tf.shape(x)[0], 1])

        x = self.img_embed(x) + self.pos_embedding(pos)
        x = self.dropout(x, training=training)

        for layer in self.layers:
            x = layer(x, mask, training=training)

        return x


class TransformerClassifierImg(Model):
    def __init__(self, img_size, embed_dim, num_heads, num_layers, ff_dim, num_classes, dropout, max_len, **kwargs):
        super(TransformerClassifierImg, self).__init__(**kwargs)
        self.encoder = EncoderImg(img_size, embed_dim, num_heads, num_layers, ff_dim, dropout, max_len)
        self.fc = layers.Dense(num_classes)
        self.dropout = layers.Dropout(dropout)

    def call(self, input_ids, attention_mask, training=False):
        encoder_output = self.encoder(input_ids, attention_mask, training=training)
        pooled_output = tf.reduce_mean(encoder_output, axis=1)
        output = self.dropout(pooled_output, training=training)
        return self.fc(output)


class EncoderPinyin(layers.Layer):
    def __init__(self, pinyin_hid_dim, embed_dim, num_heads, num_layers, ff_dim, dropout, max_len, **kwargs):
        super(EncoderPinyin, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.embedding = PinyinEmbedding(pinyin_hid_dim, embed_dim)
        self.pos_embedding = layers.Embedding(max_len, embed_dim)
        self.layers = [EncoderLayer(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        self.dropout = layers.Dropout(dropout)

    def call(self, x, pinyin_mask, mask, training=False):
        seq_len = tf.shape(x)[1]
        pos = tf.tile(tf.expand_dims(tf.range(0, seq_len), 0), [tf.shape(x)[0], 1])

        x = self.embedding(x, pinyin_mask) + self.pos_embedding(pos)
        x = self.dropout(x, training=training)

        for layer in self.layers:
            x = layer(x, mask, training=training)

        return x


class TransformerClassifierPinyin(Model):
    def __init__(self, pinyin_hid_dim, embed_dim, num_heads, num_layers, ff_dim, num_classes, dropout, max_len,
                 **kwargs):
        super(TransformerClassifierPinyin, self).__init__(**kwargs)
        self.encoder = EncoderPinyin(pinyin_hid_dim, embed_dim, num_heads, num_layers, ff_dim, dropout, max_len)
        self.fc = layers.Dense(num_classes)
        self.dropout = layers.Dropout(dropout)

    def call(self, input_ids, pinyin_mask, attention_mask, training=False):
        encoder_output = self.encoder(input_ids, pinyin_mask, attention_mask, training=training)
        pooled_output = tf.reduce_mean(encoder_output, axis=1)
        output = self.dropout(pooled_output, training=training)
        return self.fc(output)


class PolicyBlock(Model):
    def __init__(self, vocab_size, img_size, embedding_dim, hidden_dim, max_len, dropout, num_action=3, **kwargs):
        super(PolicyBlock, self).__init__(**kwargs)
        self.char_embed = layers.Embedding(vocab_size, embedding_dim)
        self.pinyin_embed = PinyinEmbedding(pinyin_out_dim=embedding_dim)
        self.glyph_embed = layers.Dense(embedding_dim)
        self.pos_embed = layers.Embedding(max_len, embedding_dim)
        self.fc1 = layers.Dense(embedding_dim)
        self.fc2 = layers.Dense(hidden_dim)
        self.fc3 = layers.Dense(embedding_dim)
        self.decision = layers.Dense(num_action)
        self.dropout = layers.Dropout(dropout)

    def call(self, x_char, x_glyph, x_pinyin, pinyin_mask, training=False):

        pinyin_mask = tf.cast(pinyin_mask, tf.float32)
        x_char = self.char_embed(x_char)
        x_glyph = self.glyph_embed(x_glyph)
        x_pinyin = tf.cast(x_pinyin, tf.float32)
        x_pinyin = self.pinyin_embed(x_pinyin, pinyin_mask)

        # 连接三个嵌入
        x_cat = tf.concat([x_char, x_glyph, x_pinyin], axis=2)

        seq_len = tf.shape(x_cat)[1]
        pos = tf.tile(tf.expand_dims(tf.range(0, seq_len), 0), [tf.shape(x_cat)[0], 1])

        x = self.fc1(x_cat) + self.pos_embed(pos)
        x = self.dropout(x, training=training)
        x = tf.nn.relu(self.fc2(x))
        x = self.dropout(x, training=training)
        x = self.fc3(x)

        output = tf.reduce_mean(x, axis=1)
        output = self.dropout(output, training=training)
        output = self.decision(output)

        return output


class Defender(Model):
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

    def call(self, x_char, x_glyph, x_pinyin, pinyin_mask, mask, training=False):
        action = self.policy(x_char, x_glyph, x_pinyin, pinyin_mask, training=training)
        mask = tf.cast(mask, tf.float32)
        pinyin_mask = tf.cast(pinyin_mask, tf.float32)
        prob_char = self.char_net(x_char, mask, training=training)
        prob_glyph = self.glyph_net(x_glyph, mask, training=training)
        prob_pronunciation = self.pronunciation_net(x_pinyin, pinyin_mask, mask, training=training)

        # 对动作进行加权
        weighted_prob_char = prob_char * tf.expand_dims(action[:, 0], axis=1)  # (batch_size, 2)
        weighted_prob_glyph = prob_glyph * tf.expand_dims(action[:, 1], axis=1)  # (batch_size, 2)
        weighted_prob_pronunciation = prob_pronunciation * tf.expand_dims(action[:, 2], axis=1)  # (batch_size, 2)

        # 对加权后的向量进行逐元素求和
        x = weighted_prob_char + weighted_prob_glyph + weighted_prob_pronunciation  # (batch_size, 2)

        return x


defender = Defender(34765, 4096, 256, 128, 512, 1024, 2, 8,
                 6, num_action=3, dropout=0.1, pinyin_hid_dim=128)