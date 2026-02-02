import os
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import torch
import torch.nn as nn
import json
import pygame
from pypinyin import lazy_pinyin, Style
import numpy as np
from tqdm import tqdm
import defender as defender

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
    def __init__(self, filepath):
        self.samples = self.read_excel(filepath)

    def __getitem__(self, item):
        text, label = self.samples[item]
        return text, label

    def __len__(self):
        return len(self.samples)

    def read_excel(self, filepath):
        import pandas as pd
        samples = []
        df = pd.read_excel(filepath)
        for _, row in df.iterrows():
            text, label = row
            # label = 0 if label == '否' else 1
            samples.append((str(text), label))  # 用pandas读取文件注意
        return samples





def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for text, label in dataloader:
            tokens = TextToImgAndPinyin(text)
            x_char = torch.tensor(tokens.input_ids).to(device)
            x_pinyin = torch.tensor(tokens.pinyins).to(device)
            x_glyph = torch.tensor(tokens.imgs).to(device)
            pinyin_mask = torch.tensor(tokens.pinyin_masks).to(device)
            attentions_mask = torch.tensor(tokens.attention_mask).to(device)
            pred = model(x_char, x_glyph, x_pinyin, pinyin_mask, attentions_mask)
            labels = label.clone().detach().to(device)
            test_loss += loss_fn(pred, labels).item()
            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()


    test_loss /= num_batches
    correct /= size
    print(f" Test Error:\n Accuracy:{(100 * correct):>0.2f}%, Avg loss:{test_loss:>8f} \n")


if __name__ == "__main__":

    # 定义分词器
    tokenizer = BertTokenizer.from_pretrained('tokenizer')



    # 选择用GPU训练
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU is not available, using CPU instead")

    # dataset返还的数据类型：字典，字典中的内容如下所示
    # 'input_ids':(batch_size * max_size ) 二维矩阵，元素为token在词表中对应的id
    # 'attention_mask':(batch_size * max_len) 位置掩码，遮掩掉[PAD]token
    # 'token_type_ids'
    # 'imgs':(batch_size * max_len * 64 * 64)，numpy数组，表示每个token图像的二维矩阵
    # ''
    batch_size = 16
    # 读数据集
    training_data = MyDataset('/home/students/xgq25/代码/防御方实验/data/baidu/train_set.xlsx')
    testing_data = MyDataset('/home/students/xgq25/代码/防御方实验/data/baidu/val_set.xlsx')
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle = True)

    # 模型构建
    vocab_size = len(tokenizer)
    embedding_dim = 256
    hidden_dim = 128
    num_heads = 8
    num_layers = 6
    d_ff = 1024
    max_len = 512
    dropout = 0.2
    num_classes = 2
    img_size = 4096
    pinyin_hid_dim = 128
    print(vocab_size)
    model = defender.Defender(vocab_size, img_size, embedding_dim, hidden_dim, max_len, d_ff, num_classes, num_heads, num_layers)

    pretrained_char_net = torch.load('../data/demo/model/pretrained/char_net/epoch_40_model.pt')
    pretrained_glyph_net = torch.load('../data/demo/model/pretrained/glyph_net/epoch_25_model.pt')
    pretrained_pronunciation_net = torch.load('../data/demo/model/pretrained/pronunciation_net/epoch_40_model.pt')


    model.char_net.load_state_dict(pretrained_char_net)
    model.glyph_net.load_state_dict(pretrained_glyph_net)
    model.pronunciation_net.load_state_dict(pretrained_pronunciation_net)



    # 冻结三部分的参数
    for param in model.char_net.parameters():
        param.requires_grad = False
    for param in model.glyph_net.parameters():
        param.requires_grad = False
    for param in model.pronunciation_net.parameters():
        param.requires_grad = False


    # 训练过程
    learning_rate = 2e-5
    epochs = 50
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    config = {
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'num_layers': num_layers,
        'dropout': dropout,
        'vocab_size': vocab_size,
        'num_classes':num_classes,
        'd_ff':d_ff,
        'num_heads':num_heads,
        'embedding_dim':embedding_dim,
        'hidden_dim':hidden_dim,
        'max_len':max_len,
        'img_size':img_size,
        'pinyin_hid_dim':pinyin_hid_dim
    }
    print(config)
    config_path = '../data/demo/model/transformer_pinyin_model/pinyin2/config.json'
    f = open(config_path, "w")
    config = json.dumps(config)
    f.write(config)
    f.close()

    model.to(device)
    pygame.init()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
        if epoch % 5 == 0 and epoch != 0:
            save_path = "model/transformer_pinyin_model/pinyin2/epoch_" + str(epoch) + "_model.pt"
            torch.save(model, save_path)
    print('Done!')
