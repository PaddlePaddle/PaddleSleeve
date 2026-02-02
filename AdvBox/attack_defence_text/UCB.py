import numpy as np
import random
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import BertTokenizer
import json
import pygame
from pypinyin import lazy_pinyin, Style
import pandas as pd
def load_from_json(filename):
    with open(filename, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    return data

dict_filename1 = '形近字语料库.json'  # 替换为你的 JSON 文件名
dict_filename2 = '音近字语料库.json'


glyph_dict = load_from_json(dict_filename1)
pronunciation_dict = load_from_json(dict_filename2)

criterion = nn.CrossEntropyLoss()  # 假设使用交叉熵作为损失函数
tokenizer = BertTokenizer.from_pretrained('tokenizer')

pinyin_map_path = 'pinyin_map.json'
with open(pinyin_map_path, encoding="utf-8") as f:
    pinyin_map = json.load(f)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU is not available, using CPU instead")

def compute_reward( input_text, labels, model, pinyin_map=pinyin_map):

    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 在评估模式下不需要计算梯度
        # 假设 input_text 是已经预处理过的，转换为张量
        tokens = TextToImgAndPinyin(input_text, pinyin_map)
        print(tokens)
        x_char = torch.tensor(tokens.input_ids).to(device)
        x_pinyin = torch.tensor(tokens.pinyins).to(device)
        x_glyph = torch.tensor(tokens.imgs).to(device)
        pinyin_mask = torch.tensor(tokens.pinyin_masks).to(device)
        attentions_mask = torch.tensor(tokens.attention_mask).to(device)
        pred = model(x_char, x_glyph, x_pinyin, pinyin_mask, attentions_mask)
        labels = torch.tensor(labels, dtype=torch.long)
        labels = labels.clone().detach().to(device)
        loss = criterion(pred, labels)

        # 将损失直接作为奖励，损失越大奖励越大
        reward = loss.item()

    return reward


def load_from_json(filename):
    with open(filename, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    return data


def TextToImgAndPinyin(texts, pinyin_map):
    # 1. 提前加载拼音映射表
    #with open(pinyin_map_path, encoding="utf-8") as f:
    #    pinyin_map = json.load(f)

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


defender_model_path = '../防御方实验/data/demo/model/epoch_0_model.pt'
defender_model = torch.load(defender_model_path, weights_only= False)


with open('../防御方实验/pinyin_map.json', encoding="utf-8") as f:
    pinyin_map = json.load(f)

class UCBJoint:
    def __init__(self, text, label, n_arms_per_machine, c, model=defender_model, tokenizer=tokenizer, pronunciation_dict=pronunciation_dict, glyph_dict=glyph_dict, tolerance=1e-3, max_steps=1000):
        self.tokenizer = tokenizer
        self.label = label
        self.tokens = self.tokenizer.tokenize(text)
        self.n_machines = len(self.tokens)
        self.n_arms_per_machine = n_arms_per_machine
        self.c = c
        self.tolerance = tolerance
        self.max_steps = max_steps
        self.machine_rewards = np.zeros((self.n_machines, n_arms_per_machine))
        self.machine_counts = np.ones((self.n_machines, n_arms_per_machine))  # 初始化为1，防止初始时除零错误
        self.total_counts = 0
        self.rewards_history = []
        self.pronunciation_dict = pronunciation_dict
        self.glyph_dict = glyph_dict
        self.model = model

    def get_newword(self, word, replace_dict):
        try:
            replace_words = replace_dict[word]
        except KeyError:
            replace_words = word
        return replace_words

    def word_replacement(self, token, replace_dict):

        replaced_words = ''

        for word in token:
            replace_words = self.get_newword(word, replace_dict)
            replace_word = random.choice(replace_words)
            replaced_words = replaced_words + replace_word

        return replaced_words

    def select_arms(self):
        selected_arms = []
        for machine_idx in range(self.n_machines):
            ucb_values = np.zeros(self.n_arms_per_machine)
            for arm in range(self.n_arms_per_machine):
                average_reward = self.machine_rewards[machine_idx, arm] / self.machine_counts[machine_idx, arm]
                ucb = average_reward + self.c * np.sqrt((2 * np.log(self.total_counts + 1)) / self.machine_counts[machine_idx, arm])
                ucb_values[arm] = ucb
            # 为每个老虎机选择具有最高UCB的拉臂
            selected_arms.append(np.argmax(ucb_values))
        return selected_arms

    def text_transform(self, selected_arms):
        text_variants = ''
        for token, action in zip(self.tokens, selected_arms):
            if action == 1:
                new_token = self.word_replacement(token, self.glyph_dict)
            elif action == 2:
                new_token = self.word_replacement(token, self.pronunciation_dict)
            else:
                new_token = token
            text_variants = text_variants + new_token

        return text_variants




    def update(self, selected_arms, total_reward):
        for machine_idx in tqdm(range(self.n_machines)):
            arm_idx = selected_arms[machine_idx]
            self.machine_counts[machine_idx, arm_idx] += 1
            self.machine_rewards[machine_idx, arm_idx] += total_reward  # 每个老虎机的拉臂共享总收益
        self.total_counts += 1

    def run(self, reward_function):

        for step in range(self.max_steps):
            # 选择每个老虎机的拉臂
            selected_arms = self.select_arms()

            action = self.text_transform(selected_arms)

            # 获取所有拉臂组合的总收益
            total_reward = reward_function([action], [self.label], self.model, pinyin_map)

            # 基于总收益更新UCB值
            self.update(selected_arms, total_reward)

            # 计算当前平均收益
            average_reward = np.sum(self.machine_rewards) / np.sum(self.machine_counts)
            self.rewards_history.append(average_reward)

            # 检查是否收敛
            if len(self.rewards_history) > 1 and abs(self.rewards_history[-1] - self.rewards_history[-2]) < self.tolerance:
                print(f"Converged at step {step}")
                break

        return self.text_transform(self.select_arms())


if __name__ == "__main__":

    pygame.init()
    text_set_path = '../防御方实验/data/demo/attack_demo.xlsx'
    text_set = pd.read_excel(text_set_path)

    for index in text_set.index.values:
        text = text_set.loc[index, 'query']
        label = text_set.loc[index, 'types']
        agent = UCBJoint(text, label, 3, 1.4)
        text_set.loc[index,'purturbed_text_1'] = agent.run(compute_reward)

    text_set.to_excel(text_set_path)



'''

def joint_reward_function(selected_arms):
    # 定义某个固定拉臂组合的收益（真实分布可根据实际情况设定）
    true_combinations = {
        (0, 0, 0): 0.5,
        (0, 1, 2): 0.9,
        (2, 2, 1): 0.7,
        # 可以定义更多组合...
    }
    return true_combinations.get(tuple(selected_arms), 0.2)  # 未定义组合返回默认值

# 初始化联合UCB算法，3个老虎机，每个老虎机有3个拉臂
ucb_joint = UCBJoint(n_arms_per_machine=3, n_machines=3, c=1.0)

# 运行UCB算法，观察收益变化
rewards_history = ucb_joint.run(joint_reward_function)

# 显示结果
import pandas as pd
df_rewards_history = pd.DataFrame(rewards_history, columns=["Average Reward"])
print(df_rewards_history)
'''
