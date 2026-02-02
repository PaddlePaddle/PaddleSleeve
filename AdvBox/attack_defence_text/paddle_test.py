import os
import json
import numpy as np
import pandas as pd
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from transformers import BertTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
from pypinyin import lazy_pinyin, Style
import pygame

# 导入PaddlePaddle版本的Defender模型
from paddle_defender import Defender


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


class MyDataset:
    def __init__(self, filepath, col, batch_size=16):
        self.samples = self.read_excel(filepath, col)
        self.batch_size = batch_size
        self.num_samples = len(self.samples)
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        batch_samples = self.samples[idx * self.batch_size:(idx + 1) * self.batch_size]
        texts = [sample[0] for sample in batch_samples]
        labels = [sample[1] for sample in batch_samples]
        return texts, np.array(labels, dtype=np.int32)

    def read_excel(self, filepath, col):
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
    loss_fn = nn.CrossEntropyLoss()

    # 设置模型为评估模式
    model.eval()

    for batch_idx in tqdm(range(len(dataloader)), desc=desc):
        texts, labels = dataloader[batch_idx]

        # 准备输入数据
        tokens = TextToImgAndPinyin(texts, tokenizer, pinyin_map_path="pinyin_map.json")

        # 转换为PaddlePaddle张量
        x_char = paddle.to_tensor(tokens['input_ids'], dtype='int64')
        x_pinyin = paddle.to_tensor(tokens['pinyins'], dtype='int64')
        x_glyph = paddle.to_tensor(tokens['imgs'], dtype='float32')
        pinyin_mask = paddle.to_tensor(tokens['pinyin_masks'], dtype='float32')
        attention_mask = paddle.to_tensor(tokens['attention_mask'], dtype='float32')

        # 转换为PaddlePaddle张量
        labels_tensor = paddle.to_tensor(labels, dtype='int64')

        # 前向传播（不计算梯度）
        with paddle.no_grad():
            outputs = model(x_char, x_glyph, x_pinyin, pinyin_mask, attention_mask)
            loss = loss_fn(outputs, labels_tensor)

        total_loss += loss.item()

        # 获取预测结果
        predictions = paddle.argmax(outputs, axis=1).numpy()
        all_labels.extend(labels)
        all_predictions.extend(predictions)

    avg_loss = total_loss / len(dataloader)
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))

    # 计算其他指标（确保有至少两个类别）
    unique_labels = np.unique(all_labels)
    if len(unique_labels) > 1:
        precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    else:
        # 如果只有一个类别，设置指标为0
        precision = 0.0
        recall = 0.0
        f1 = 0.0

    return avg_loss, accuracy, precision, recall, f1


if __name__ == "__main__":
    # 定义分词器
    tokenizer = BertTokenizer.from_pretrained('tokenizer')

    pygame.init()

    # 配置设备
    device = 'gpu' if paddle.device.is_compiled_with_cuda() else 'cpu'
    paddle.set_device(device)
    print(f"Using device: {device}")

    # 创建模型（随机初始化权重）
    model = Defender(34765, 4096, 256, 128, 512, 1024, 2, 8, 6,
                     num_action=3, dropout=0.1, pinyin_hid_dim=128)



    # 如果你有保存的PaddlePaddle模型权重，可以加载
    # model_save_path = '../防御方实验/data/demo/model/epoch_0_model.pdparams'
    # model_state_dict = paddle.load(model_save_path)
    # model.set_state_dict(model_state_dict)

    # 定义要评估的列名
    col_names = ['query', '字音变体(0.1)', '字音变体(0.2)', '字音变体(0.3)', '字音变体(0.4)',
                 '字音变体(0.5)', '字音变体(0.6)', '字音变体(0.7)', '字音变体(0.8)', '字音变体(0.9)',
                 '字形变体(0.1)', '字形变体(0.2)', '字形变体(0.3)', '字形变体(0.4)', '字形变体(0.5)',
                 '字形变体(0.6)', '字形变体(0.7)', '字形变体(0.8)', '字形变体(0.9)']

    result = pd.DataFrame(index=col_names, columns=['avg_loss', 'accuracy', 'precision', 'recall', 'f1'])

    for col in col_names:
        print(f"\n{'=' * 60}")
        print(f"Evaluating on column: {col}")
        print('=' * 60)

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
    print(f"\n结果已保存到: {result_save_path}")


    pygame.quit()