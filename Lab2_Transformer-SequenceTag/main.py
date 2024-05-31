import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from TorchCRF import CRF
from gensim.models import KeyedVectors 
from tqdm import tqdm
from torch.utils.data import DataLoader

# scipy==1.10.1

class NerDataset(Dataset):
    def __init__(self, corpus_list, tags_list):
        """
        初始化方法
        :param corpus_list: 转换为词向量形式的语料
        :param tags_list: 转换为id形式的标签
        """
        self.corpus_list = corpus_list
        self.tags_list = tags_list

    def __len__(self):
        return len(self.tags_list)

    def __getitem__(self, item):
        """
        :param item: 获取句子的下标
        :return: 返回转换为tensor后的句子，标签以及句子的长度
        """
        corpus = torch.tensor(self.corpus_list[item], dtype=torch.float32)
        # tags = torch.tensor(self.tags_list[item], dtype=torch.int32)
        tags = torch.tensor(self.tags_list[item], dtype=torch.long)
        return corpus, tags, len(self.tags_list[item])


def my_collate_fn(batch):
    corpus_list = []
    tag_list = []
    len_list = []
    for bt in batch:
        corpus_list.append(bt[0])
        tag_list.append(bt[1])
        len_list.append(bt[2])
    corpus_list = pad_sequence(corpus_list, batch_first=True, padding_value=0.0)
    tag_list = pad_sequence(tag_list, batch_first=True, padding_value=0)
    return corpus_list, tag_list, len_list



class BiLstmCrf(nn.Module):
    def __init__(self,device, output_size, emb_size=100, hidden_dim=256):
        """
        模型的初始化方法
        :param device: 当前使用的设备（'gpu'或'cpu'）
        :param output_size: 输出的维度
        :param emb_size: 编码的维度
        :param hidden_dim: 隐含层的维度
        """
        super(BiLstmCrf, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.device = device
        self.lstm = nn.LSTM(input_size=emb_size, hidden_size=self.hidden_dim // 2, num_layers=2, batch_first=True,
                            bidirectional=True)
        self.fc = nn.Linear(self.hidden_dim, self.output_size)
        self.crf = CRF(self.output_size)

    def forward(self, x, target, sentence_len, is_predict=False):
        """
        前向传播
        x: 输入的向量 [batch_size, sentence_len, embedding_size]
        target: 标签向量[batch_size, sentence_len]
        sentence_len: batch 中每个句子的长度 [batch_size]
        is_predict: 当前是否为预测，若不为预测，返回的为loss，否则为解码的结果
        """
        mask = torch.tensor([[True if j < length else 0 for j in range(x.shape[1])]
                             for length in sentence_len], dtype=torch.bool).to(self.device)
        x, _ = self.lstm(x)
        x = self.fc(x)

        if not is_predict:
            loss = -self.crf.forward(x, target, mask=mask)
            return loss
        else:
            decode = self.crf.viterbi_decode(x, mask=mask)
            return decode


# 预处理类
class Preprocess:
    def __init__(self, input_path='data/'):
        self.data_path = input_path
        # 标签与id相互转换的字典
        self.tag2id = {
            'PAD': 0,  # pad为填充的标签
            'B_PER': 1,
            'I_PER': 2,
            'B_T': 3,
            'I_T': 4,
            'B_ORG': 5,
            'I_ORG': 6,
            'B_LOC': 7,
            'I_LOC': 8,
            'O': 9,
        }
        self.id2tag = {}
        for tag, id in self.tag2id.items():
            self.id2tag[id] = tag
        tencent_file_path = "data/tencent-ailab-embedding-zh-d100-v0.2.0-s.txt"
        # self.word2vector = KeyedVectors.load_word2vec_format(tencent_file_path, binary=True)
        # self.word2vector = KeyedVectors.load_word2vec_format(tencent_file_path, binary=True, encoding='latin1')
        # self.word2vector = KeyedVectors.load_word2vec_format(tencent_file_path, binary=True, unicode_errors='ignore')
        
        # tmp_w2v = KeyedVectors.load_word2vec_format(tencent_file_path, binary=False)
        # tmp_w2v.save("data/tencent-ailab-embedding-zh-d100-v0.2.0-s.bin")
        tmp_w2v = KeyedVectors.load("data/tencent-ailab-embedding-zh-d100-v0.2.0-s.bin")
        self.word2vector = tmp_w2v


    def read_corpus(self, corpus_name: str):
        """
        读取语料的方法
        :param corpus_name:
        :return:
        """
        result_list = []
        with open(self.data_path + corpus_name, 'r', encoding='UTF-8') as f:
            corpus_lines = f.readlines()
            for line in corpus_lines:
                result_list.append(line.split())
        return result_list

    @staticmethod
    def cut_corpus(corpus_list: list, tag_list: list = None, has_tag=False):
        """
        切分语料的方法
        :param corpus_list: 待切分的语料list
        :param tag_list: 传入corpus_list对应的标签列表
        :param has_tag: 是否传入标签
        :return: 切分后的语料（与标签）
        """
        new_corpus_list = []
        new_tag_list = []
        # 遍历语料的每一行
        for i in range(len(corpus_list)):
            # 每行截断的起始位置
            start_index = 0
            # 遍历每一个字符
            for j in range(len(corpus_list[i])):
                word = corpus_list[i][j]  # 当前字符
                # 如果一句话结束或者该行语料结束，则可以截断
                if word == ';' or word == '。' or j + 1 == len(corpus_list[i]):
                    new_corpus_list.append(corpus_list[i][start_index:j + 1])
                    if has_tag:
                        new_tag_list.append(tag_list[i][start_index:j + 1])
                    start_index = j + 1  # 更新下一个句子的起始index
        if has_tag:
            return new_corpus_list, new_tag_list
        else:
            return new_corpus_list

    def word_to_id(self, corpus_list):
        """
        将字转换为对应向量的方法
        :param corpus_list: 待转换的语料列表
        :return: 转换后的向量
        """
        id_list = []
        for sentence in corpus_list:
            id_sentence = []
            for word in sentence:
                if self.word2vector.has_index_for(word):
                    id_sentence.append(self.word2vector[word])
                else:
                    id_sentence.append(self.word2vector[0])
            id_list.append(id_sentence)
        return id_list

    def tag_to_id(self, tag_list):
        """
        将标签转换为id的方法
        :param tag_list: 待转换的标签列表
        :return: 转换后的id列表
        """
        id_list = []
        for sentence in tag_list:
            id_sentence = []
            for tag in sentence:
                id_sentence.append(self.tag2id[tag])
            id_list.append(id_sentence)
        return id_list

    def preprocess_corpus(self, corpus_name, tag_name='', has_tag=False):
        """
        读取语料并将其转换为id向量格式的方法
        :param corpus_name: 语料名称
        :param tag_name: 标签名称
        :param has_tag: 是否传入标签
        :return: 三维向量格式的语料以及三维id列表格式的标签
        """
        corpus = self.read_corpus(corpus_name)
        tags = None
        if has_tag:
            tags = self.read_corpus(tag_name)
            corpus, tags = self.cut_corpus(corpus, tags, has_tag)

        corpus = self.word_to_id(corpus)
        if has_tag:
            tags = self.tag_to_id(tags)
            return corpus, tags
        return corpus




def get_f1(pred, label):
    """
    计算f1值的方法
    :param pred: 预测的tag
    :param label: 真实的tag
    :return: 计算得到的f1值
    """
    assert len(pred) == len(label)
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(pred)):
        # 若真实值不为O
        if label[i] != 9:
            if pred[i] == label[i]:
                TP += 1
            else:
                FN += 1
        elif pred[i] != 9:
            FP += 1
        else:
            TN += 1
    # 准确率
    precision = 0 if (TP + FP) == 0 else TP / (TP + FP)
    # 召回率
    recall = 0 if (TP + FN) == 0 else TP / (TP + FN)
    f1 = 0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return f1


def train(max_epoch, batch_size, save_path='model/model.pth'):
    """
    训练的函数
    :param max_epoch: 最大训练轮数
    :param batch_size: 每个batch的长度
    :param save_path: 保存模型的路径
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pre = Preprocess()
    train_corpus, train_tags = pre.preprocess_corpus('train.txt', 'train_TAG.txt', has_tag=True)
    dev_corpus, dev_tags = pre.preprocess_corpus('dev.txt', 'dev_TAG.txt', has_tag=True)
    train_dataset = NerDataset(train_corpus, train_tags)
    dev_dataset = NerDataset(dev_corpus, dev_tags)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=my_collate_fn, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, collate_fn=my_collate_fn, shuffle=True)
    # 模型
    model = BiLstmCrf(device=device, output_size=len(pre.tag2id)).to(device)
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    f1_list = []
    loss_list = []

    for epoch in range(max_epoch):
        print(f'第{epoch + 1}轮...')
        print('开始训练...')
        # 训练
        tqdm_obj = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for i, (inputs, labels, lens) in tqdm_obj:
            loss = model.forward(inputs.to(device), labels.to(device), lens, is_predict=False)
            optimizer.zero_grad()
            loss.backward(torch.ones_like(loss))
            optimizer.step()
            loss = torch.mean(loss).item()
            tqdm_obj.set_postfix(loss=loss)
            # 每个Epoch结束后记录一次loss
            loss_list.append(loss)

        # 验证
        print('开始验证...')
        with torch.no_grad():
            pred_list = []
            label_list = []
            for i, (inputs, labels, lens) in enumerate(dev_dataloader):
                # 预测的pred为list格式
                pred = model.forward(inputs.to(device), labels.to(device), lens, is_predict=True)
                # 需要将labels也转换为list格式
                labels = labels.tolist()
                for j in range(len(pred)):
                    pred_list.extend(pred[j][:lens[j]])
                    label_list.extend(labels[j][:lens[j]])
            f1 = get_f1(pred_list, label_list)
            print(f'第{epoch + 1}轮训练结束，在发展集上非O标签的F1值为{f1}')
            f1_list.append(f1)

            if epoch > 1:
                # 提前终止
                if f1_list[-2] > f1 and f1_list[-2] > 0.94:
                    print('提前终止！！！')
                    break
            # 更新模型
            if epoch > 0:
                os.remove(save_path)
            torch.save(model.state_dict(), save_path)

    # 绘制loss图与f1图
    plt.plot(loss_list)
    plt.title('loss')
    plt.savefig('result/loss.jpg')
    plt.close()

    plt.plot(f1_list)
    plt.title('F1')
    plt.savefig('result/F1.jpg')
    plt.close()



# 根据训练所得模型，在测试集上进行实体抽取
def predict(model_path, save_path):
    """
    在测试集上标注的函数
    :param model_path: 模型的路径
    :param save_path: 序列标注得到的标签文件存放的路径
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pre = Preprocess()
    test_corpus = pre.preprocess_corpus('test.txt', has_tag=False)
    model = BiLstmCrf(device=device, output_size=len(pre.tag2id)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    # print(model)
    tag_list = []  # 标注结果的列表
    with torch.no_grad():
        for st in test_corpus:
            len_st = len(st)
            # 获取标签
            st = torch.tensor(st, dtype=torch.float32).unsqueeze(0).to(device)
            pred = model.forward(st, None, [len_st], is_predict=True)

            st_tag = []
            for id in pred[0]:
                # 将预测为填充的结果转换为O
                if id == 0:
                    id = 9
                st_tag.append(pre.id2tag[id])
            tag_list.append(st_tag)

    # 将预测的结果保存在磁盘中
    with open(save_path, 'w', encoding='UTF-8') as f:
        for st_tag in tag_list:
            line = ' '.join(st_tag)
            f.write(line + '\n')
        f.close()


if __name__ == '__main__':
    train(max_epoch=20, batch_size=512)
    predict('model/model.pth','result/test_TAG.txt')
