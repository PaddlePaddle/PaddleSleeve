import os
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import torch
import torch.nn as nn
import net.transformer as tf
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import json
import numpy as np
import pygame
from tqdm import tqdm
from pypinyin import lazy_pinyin, Style
import net.defender as defender

#读json文件
def load_from_json(filename):
    with open(filename, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    return data

def TextToImgAndPinyin(texts, pinyin_map_path="pinyin_map.json"):
    # 1. 提前加载拼音映射表
    with open(pinyin_map_path, encoding="utf-8") as f:
        pinyin_map = json.load(f)

    # 2. Tokenize 输入的文本
    batch_tokens = tokenizer(texts, padding=True, truncation=True, max_length=512)

    batch_imgs = []
    batch_pinyins = []
    batch_pinyin_masks = []

    # 3. 创建字体对象（假设已初始化 pygame）
    font = pygame.font.Font("STSong.ttf", 128)

    for tokens_ids in batch_tokens.input_ids:
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

    # 7. 将拼音、拼音掩码、和图像特征添加到批量token结果中
    batch_tokens['imgs'] = np.array(batch_imgs)  # 图像特征
    batch_tokens['pinyins'] = batch_pinyins  # 拼音特征
    batch_tokens['pinyin_masks'] = batch_pinyin_masks  # 拼音掩码

    return batch_tokens


class MyDataset(Dataset):
    def __init__(self, filepath, col):
        self.samples = self.read_excel(filepath, col)

    def __getitem__(self, item):
        text, label = self.samples[item]
        return text, label

    def __len__(self):
        return len(self.samples)

    def read_excel(self, filepath, col):
        import pandas as pd
        samples = []
        df = pd.read_excel(filepath)
        for index in df.index.values:
            text, label = df.loc[index, col], df.loc[index, 'types']
            # label = 0 if label == '否' else 1
            samples.append((str(text), label))  # 用pandas读取文件注意
        return samples





def evaluate_model(model, dataloader, device, desc):
    model.eval()  # 设置模型为评估模式
    total_loss = 0
    all_labels = []
    all_predictions = []
    criterion = torch.nn.CrossEntropyLoss()  # 损失函数

    with torch.no_grad():  # 不计算梯度
        for text, labels in tqdm(dataloader, desc=desc, unit='batch'):
            tokens = TextToImgAndPinyin(text, pinyin_map_path="net/pinyin_map.json")
            x_char = torch.tensor(tokens.input_ids).to(device)
            x_pinyin = torch.tensor(tokens.pinyins).to(device)
            x_glyph = torch.tensor(tokens.imgs).to(device)
            pinyin_mask = torch.tensor(tokens.pinyin_masks).to(device)
            attentions_mask = torch.tensor(tokens.attention_mask).to(device)
            outputs = model(x_char, x_glyph, x_pinyin, pinyin_mask, attentions_mask)

            labels = labels.clone().detach().to(device)



            # 前向传播

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # 获取预测结果
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = (torch.tensor(all_predictions) == torch.tensor(all_labels)).float().mean().item()
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    return avg_loss, accuracy, precision, recall, f1

if __name__ == "__main__":

    # 定义分词器
    tokenizer = BertTokenizer.from_pretrained('tokenizer')

    pygame.init()

    #读配置文件
    #config_path = 'model/defender1.0/config.json'
    #config = load_from_json(config_path)


    # 选择用GPU训练
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU is not available, using CPU instead")

    #待评估的模型
    model_save_path = 'data/demo/model/epoch_0_model.pt'
    model = torch.load(model_save_path, weights_only= False)
    print(f"PyTorch version: {torch.__version__}")

    col_names = ['query','字音变体(0.1)','字音变体(0.2)','字音变体(0.3)','字音变体(0.4)','字音变体(0.5)','字音变体(0.6)','字音变体(0.7)','字音变体(0.8)','字音变体(0.9)',
                 '字形变体(0.1)','字形变体(0.2)','字形变体(0.3)','字形变体(0.4)','字形变体(0.5)','字形变体(0.6)','字形变体(0.7)','字形变体(0.8)','字形变体(0.9)']

    result = pd.DataFrame(index=col_names)

    for col in col_names:

        #要评估的测试集
        testing_data = MyDataset('data/demo/demo_set.xlsx', col)
        test_dataloader = DataLoader(testing_data, batch_size=16, shuffle=True)

        avg_loss, accuracy, precision, recall, f1 = evaluate_model(model, test_dataloader, device, col+' Evaluating')
        result.loc[col, 'avg_loss'] = avg_loss
        result.loc[col, 'accuracy'] = accuracy
        result.loc[col, 'precision'] = precision
        result.loc[col, 'recall'] = recall
        result.loc[col, 'f1'] = f1

    result_save_path ='data/demo/defense_demo.xlsx'
    result.to_excel(result_save_path)


