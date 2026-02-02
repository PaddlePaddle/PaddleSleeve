import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from transformers import BertTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
from pypinyin import lazy_pinyin, Style
import pygame

# 假设你的TensorFlow Defender模型定义在这个文件中
from tensorflow_defender import Defender  # 你需要创建TensorFlow版本的Defender


# 读json文件
def load_from_json(filename):
    with open(filename, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    return data


def TextToImgAndPinyin(texts, tokenizer, pinyin_map_path="pinyin_map.json"):
    # 1. 提前加载拼音映射表
    with open(pinyin_map_path, encoding="utf-8") as f:
        pinyin_map = json.load(f)

    # 2. Tokenize 输入的文本
    batch_tokens = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="np")

    batch_imgs = []
    batch_pinyins = []
    batch_pinyin_masks = []

    # 3. 创建字体对象（假设已初始化 pygame）
    font = pygame.font.Font("STSong.ttf", 128)

    for tokens_ids in batch_tokens['input_ids']:
        token_texts = tokenizer.convert_ids_to_tokens(tokens_ids)  # 获取每个token的文本
        token_imgs = []  # 存储每个token的图像特征
        token_pinyins = []  # 存储每个token的拼音编码
        pinyin_masks = []  # 存储每个token的拼音掩码

        for token_text in token_texts:
            # 4. 处理图像特征
            t_img = font.render(token_text, True, (0, 0, 0), (255, 255, 255))
            t_img = pygame.transform.scale(t_img, (64, 64))  # 缩放到 64x64 大小
            t_img = pygame.surfarray.array2d(t_img).flatten().astype(np.float32)  # 转为一维浮点数数组
            token_imgs.append(t_img)

            # 5. 处理拼音特征，使用拼音转换，将错误的字符处理为'-'
            t_pinyin = lazy_pinyin(token_text, style=Style.TONE3, errors=lambda item: '-')
            item_pinyin = t_pinyin[0].ljust(8, '-')  # 将拼音补齐到8个字符

            # 6. 生成拼音索引和掩码
            pinyin_index = [pinyin_map.get(char, pinyin_map['-']) for char in item_pinyin]
            pinyin_mask = [0 if char == '-' else 1 for char in item_pinyin]

            token_pinyins.append(pinyin_index)
            pinyin_masks.append(pinyin_mask)

        batch_imgs.append(token_imgs)
        batch_pinyins.append(token_pinyins)
        batch_pinyin_masks.append(pinyin_masks)

    # 7. 转换为numpy数组
    batch_tokens['imgs'] = np.array(batch_imgs, dtype=np.float32)
    batch_tokens['pinyins'] = np.array(batch_pinyins, dtype=np.int32)
    batch_tokens['pinyin_masks'] = np.array(batch_pinyin_masks, dtype=np.int32)

    return batch_tokens


class MyDataset(keras.utils.Sequence):
    def __init__(self, filepath, col, batch_size=16):
        self.samples = self.read_excel(filepath, col)
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.samples) / self.batch_size))

    def __getitem__(self, idx):
        batch_samples = self.samples[idx * self.batch_size:(idx + 1) * self.batch_size]
        texts = [sample[0] for sample in batch_samples]
        labels = [sample[1] for sample in batch_samples]
        return texts, np.array(labels, dtype=np.int32)

    def read_excel(self, filepath, col):
        import pandas as pd
        samples = []
        df = pd.read_excel(filepath)
        for index in df.index.values:
            text, label = df.loc[index, col], df.loc[index, 'types']
            samples.append((str(text), label))
        return samples


def evaluate_model(model, dataloader, tokenizer, desc):
    total_loss = 0
    all_labels = []
    all_predictions = []

    # 定义损失函数
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    for batch_idx in tqdm(range(len(dataloader)), desc=desc):
        texts, labels = dataloader[batch_idx]

        # 准备输入数据
        tokens = TextToImgAndPinyin(texts, tokenizer, pinyin_map_path="net/pinyin_map.json")

        x_char = tf.convert_to_tensor(tokens['input_ids'])
        x_pinyin = tf.convert_to_tensor(tokens['pinyins'])
        x_glyph = tf.convert_to_tensor(tokens['imgs'])
        pinyin_mask = tf.convert_to_tensor(tokens['pinyin_masks'])
        attentions_mask = tf.convert_to_tensor(tokens['attention_mask'])

        # 转换为TensorFlow张量
        labels_tf = tf.convert_to_tensor(labels)

        # 前向传播
        with tf.GradientTape() as tape:
            outputs = model(x_char, x_glyph, x_pinyin, pinyin_mask, attentions_mask, training=False)
            loss = loss_fn(labels_tf, outputs)

        total_loss += loss.numpy()

        # 获取预测结果
        predictions = tf.argmax(outputs, axis=1).numpy()
        all_labels.extend(labels)
        all_predictions.extend(predictions)

    avg_loss = total_loss / len(dataloader)
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)

    return avg_loss, accuracy, precision, recall, f1


if __name__ == "__main__":
    # 定义分词器
    tokenizer = BertTokenizer.from_pretrained('tokenizer')

    pygame.init()

    # 配置GPU（如果有）
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # 设置GPU显存按需增长
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU is available: {gpus}")
            device = "/GPU:0"
        except RuntimeError as e:
            print(e)
    else:
        print("GPU is not available, using CPU instead")
        device = "/CPU:0"

    # 创建模型（随机初始化权重）
    model = Defender(34765, 4096, 256, 128, 512, 1024, 2, 8, 6,
                     num_action=3, dropout=0.1, pinyin_hid_dim=128)

    # 如果你有保存的TensorFlow模型权重，可以加载
    # model_save_path = '../防御方实验/data/demo/model/epoch_0_model.h5'
    # model.load_weights(model_save_path)

    col_names = ['query', '字音变体(0.1)', '字音变体(0.2)', '字音变体(0.3)', '字音变体(0.4)',
                 '字音变体(0.5)', '字音变体(0.6)', '字音变体(0.7)', '字音变体(0.8)', '字音变体(0.9)',
                 '字形变体(0.1)', '字形变体(0.2)', '字形变体(0.3)', '字形变体(0.4)', '字形变体(0.5)',
                 '字形变体(0.6)', '字形变体(0.7)', '字形变体(0.8)', '字形变体(0.9)']

    result = pd.DataFrame(index=col_names, columns=['avg_loss', 'accuracy', 'precision', 'recall', 'f1'])

    for col in col_names:
        # 要评估的测试集
        testing_data = MyDataset('data/demo/demo_set.xlsx', col, batch_size=16)

        avg_loss, accuracy, precision, recall, f1 = evaluate_model(
            model, testing_data, tokenizer, col + ' Evaluating'
        )

        result.loc[col, 'avg_loss'] = avg_loss
        result.loc[col, 'accuracy'] = accuracy
        result.loc[col, 'precision'] = precision
        result.loc[col, 'recall'] = recall
        result.loc[col, 'f1'] = f1

    result_save_path = '../防御方实验/data/demo/defense_demo.xlsx'
    result.to_excel(result_save_path)
    print(f"结果已保存到: {result_save_path}")