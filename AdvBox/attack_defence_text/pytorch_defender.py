
#from transformers import BertTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import net.transformer as tf
import net.transformer_img as tf_img
import net.transformer_pinyin as tf_pinyin

class PinyinEmbedding(nn.Module):
    def __init__(self, hidden_dim=128, pinyin_out_dim=256, dropout_prob=0.1):
        super(PinyinEmbedding, self).__init__()
        self.hidden_dim = hidden_dim
        self.pinyin_out_dim = pinyin_out_dim

        # 用于将 one-hot 编码转为 8 * hidden_dim 向量
        self.embedding = nn.Embedding(31, hidden_dim)

        # 1D 卷积，输入通道数为 hidden_dim，输出通道数为 hidden_dim * 6，卷积核大小为 2
        self.conv = nn.Conv1d(in_channels=hidden_dim, out_channels=self.pinyin_out_dim, kernel_size=2,
                              stride=1, padding=0)


    def forward(self, x, pinyin_mask):


        embed = self.embedding(x)
        expanded_mask = pinyin_mask.unsqueeze(-1).expand(-1, -1, -1, self.hidden_dim)
        embed = embed * expanded_mask
        bs, sentence_length, pinyin_locs, embed_size = embed.shape


        view_embed = embed.view(-1, pinyin_locs, embed_size)
        input_embed = view_embed.permute(0, 2, 1)
        pinyin_conv = self.conv(input_embed)  # [(bs*sentence_length),pinyin_out_dim,H]
        pinyin_embed = F.max_pool1d(pinyin_conv, pinyin_conv.shape[-1])  # [(bs*sentence_length),pinyin_out_dim,1]
        return pinyin_embed.view(bs, sentence_length, self.pinyin_out_dim)  # [bs,sentence_length,pinyin_out_dim]


class PolicyBlock(nn.Module):
    def __init__(self, vocab_size, img_size, embedding_dim, hidden_dim, max_len, dropout, num_action=3,):
        super(PolicyBlock, self).__init__()
        self.char_embed = nn.Embedding(vocab_size, embedding_dim)
        self.pinyin_embed = PinyinEmbedding(pinyin_out_dim=embedding_dim)
        self.glyph_embed = nn.Linear(img_size, embedding_dim)
        self.pos_embed = nn.Embedding(max_len, embedding_dim)
        self.fc1 = nn.Linear(num_action*embedding_dim, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, embedding_dim)
        self.decision = nn.Linear(embedding_dim, num_action)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_char, x_glyph, x_pinyin, pinyin_mask):

        x_char = self.char_embed(x_char)
        x_glyph = self.glyph_embed(x_glyph)
        x_pinyin = self.pinyin_embed(x_pinyin,pinyin_mask)
        x_cat= torch.cat((x_char, x_glyph, x_pinyin), dim=2)
        seq_len =x_cat.size(1)
        pos = torch.arange(0, seq_len).unsqueeze(0).repeat(x_cat.size(0), 1).to(x_cat.device)
        x = self.fc1(x_cat) + self.pos_embed(pos)
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        output = x.mean(dim=1)
        output = self.dropout(output)
        output = self.decision(output)

        return output

class Defender(nn.Module):
    def __init__(self, vocab_size, img_size, embedding_dim, hidden_dim, max_len, ff_dim, num_classes, num_heads, num_layers, num_action=3, dropout=0.1, pinyin_hid_dim=128):
        super(Defender, self).__init__()
        self.policy = PolicyBlock(vocab_size, img_size, embedding_dim, hidden_dim, max_len, dropout=dropout)
        self.char_net = tf.TransformerClassifier(vocab_size, embedding_dim, num_heads, num_layers, ff_dim, num_classes, dropout, max_len)
        self.glyph_net = tf_img.TransformerClassifier(img_size, embedding_dim, num_heads, num_layers, ff_dim, num_classes, dropout, max_len)
        self.pronunciation_net = tf_pinyin.TransformerClassifier(pinyin_hid_dim, embedding_dim, num_heads, num_layers, ff_dim, num_classes, dropout, max_len)

    def forward(self,x_char, x_glyph, x_pinyin, pinyin_mask, mask):
        action = self.policy(x_char, x_glyph, x_pinyin, pinyin_mask)
        prob_char = self.char_net(x_char, mask)
        prob_glyph = self.glyph_net(x_glyph, mask)
        prob_pronunciation = self.pronunciation_net(x_pinyin, pinyin_mask, mask)

        weighted_prob_char = prob_char * action[:, 0].unsqueeze(1)  # (batch_size, 2)
        weighted_prob_glyph = prob_glyph * action[:, 1].unsqueeze(1)  # (batch_size, 2)
        weighted_prob_pronunciation = prob_pronunciation * action[:, 2].unsqueeze(1)  # (batch_size, 2)

        # 对加权后的向量进行逐元素求和
        x = weighted_prob_char + weighted_prob_glyph + weighted_prob_pronunciation  # (batch_size, 2)

        return x





