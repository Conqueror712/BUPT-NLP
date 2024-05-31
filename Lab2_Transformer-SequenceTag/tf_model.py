import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from TorchCRF import CRF
from tqdm import tqdm
from gensim.models import KeyedVectors 

class NerDataset(Dataset):
    def __init__(self, corpus_list, tags_list, tokenizer):
        """
        功能：初始化Dataset
        参数：corpus_list: list, 语料列表
              tags_list: list, 标签列表
              tokenizer: BertTokenizer, 分词器
        """
        self.corpus_list = corpus_list
        self.tags_list = tags_list
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.tags_list)

    def __getitem__(self, item):
        """
        功能：返回指定索引的样本
        参数：item: int, 索引
        返回：inputs: dict, 模型输入, 包括input_ids, attention_mask
              labels: tensor, 标签
        """
        inputs = self.tokenizer(self.corpus_list[item], return_tensors="pt", padding=True, truncation=True)
        labels = torch.tensor(self.tags_list[item], dtype=torch.long)
        return inputs, labels

def padding_collate(batch):
    """
    功能：对batch中的样本进行padding
    参数：batch: list, 每个元素为Dataset的__getitem__返回值
    返回：inputs: dict, 模型输入, 包括input_ids, attention_mask
          labels: tensor, 标签
    """
    input_ids = [item[0]['input_ids'] for item in batch]
    attention_masks = [item[0]['attention_mask'] for item in batch]
    tags = [item[1] for item in batch]
    return {'input_ids': pad_sequence(input_ids, batch_first=True, padding_value=0), 'attention_mask': pad_sequence(attention_masks, batch_first=True, padding_value=0)}, pad_sequence(tags, batch_first=True, padding_value=0)


class BertCrf(nn.Module):
    def __init__(self, output_size, tokenizer, bert_model_name='bert-base-chinese'):
        """
        功能：初始化模型
        参数：output_size: int, 输出维度
              tokenizer: BertTokenizer, 分词器
              bert_model_name: str, Bert模型名
        """
        super(BertCrf, self).__init__()
        self.tokenizer = tokenizer
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.hidden_size = self.bert.config.hidden_size
        self.fc = nn.Linear(self.hidden_size, output_size)
        self.crf = CRF(output_size)

    def forward(self, inputs, target, is_predict=False):
        """
        功能：前向传播
        参数：inputs: dict, 模型输入, 包括input_ids, attention_mask
              target: tensor, 标签
              is_predict: bool, 是否预测
        返回：loss: tensor, 损失
        """
        bert_output = self.bert(**inputs).last_hidden_state
        logits = self.fc(bert_output)

        if not is_predict:
            mask = inputs['attention_mask'].bool()
            loss = -self.crf.forward(logits, target, mask=mask)
            return loss
        else:
            mask = inputs['attention_mask'].bool()
            decode = self.crf.viterbi_decode(logits, mask=mask)
            return decode



class Preprocess:
    def __init__(self, input_path='data/', bert_model_path='bert/bert-base-chinese'):
        self.data_path = input_path
        self.tag2id = {
            'PAD': 0,
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
        tencent_file_path = "data/tencent-ailab-embedding-zh-d100-v0.2.0-s.bin"
        self.word2vector = KeyedVectors.load(tencent_file_path)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_path)

    def rd_corpus(self, corpus_name: str):
        result_list = []
        with open(self.data_path + corpus_name, 'r', encoding='UTF-8') as f:
            corpus_lines = f.readlines()
            for line in corpus_lines:
                tag = line.strip()
                if tag:
                    result_list.append(tag)
        return result_list



    @staticmethod
    def seg_corpus(corpus_list: list, tag_list: list = None, has_tag=False):
        new_corpus_list = []
        new_tag_list = []
        for i in range(len(corpus_list)):
            start_index = 0
            for j in range(len(corpus_list[i])):
                word = corpus_list[i][j]
                if word == ';' or word == '。' or j + 1 == len(corpus_list[i]):
                    new_corpus_list.append(corpus_list[i][start_index:j + 1])
                    if has_tag:
                        new_tag_list.append(tag_list[i][start_index:j + 1])
                    start_index = j + 1
        if has_tag:
            return new_corpus_list, new_tag_list
        else:
            return new_corpus_list

    def word2id(self, corpus_list):
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

    def tag2id(self, tag_list):
        id_list = []
        for sentence in tag_list:
            id_sentence = []
            for tag in sentence:
                if tag not in self.tag2id:
                    continue
                    # print(f"Unknown tag: {tag}")
                id_sentence.append(self.tag2id.get(tag, 0))
            id_list.append(id_sentence)
        return id_list


    def preprocess_corpus(self, corpus_name, tag_name='', has_tag=False):
        """
        功能：读取语料文件，分词，将词转换为id
        参数：corpus_name: str, 语料文件名
              tag_name: str, 标签文件名
              has_tag: bool, 是否有标签文件
        返回：corpus: list, 分词后的语料              
        """
        corpus = self.rd_corpus(corpus_name)
        tags = None
        if has_tag:
            tags = self.rd_corpus(tag_name)
            if ' ' in tags:
                raise ValueError("Tag file should not contain space, please check the tag file.")
            corpus, tags = self.seg_corpus(corpus, tags, has_tag)

        print("Sample corpus:", corpus[:5])
        if has_tag:
            print("Sample tags:", tags[:5])

        corpus = [self.tokenizer.tokenize(sent) for sent in corpus]
        if has_tag:
            tags = self.tag2id(tags)
            return corpus, tags
        return corpus


def calc_f1(pred, label):
    assert len(pred) == len(label)
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(pred)):
        if label[i] != 9:
            if pred[i] == label[i]:
                TP += 1
            else:
                FN += 1
        elif pred[i] != 9:
            FP += 1
        else:
            TN += 1
    precision = 0 if (TP + FP) == 0 else TP / (TP + FP)
    recall = 0 if (TP + FN) == 0 else TP / (TP + FN)
    f1 = 0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return f1


def train(max_epoch, batch_size, bert_model_path, save_path='model/model_bert.pth'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pre = Preprocess(bert_model_path=bert_model_path)
    tokenizer = pre.tokenizer
    train_corpus, train_tags = pre.preprocess_corpus('train.txt', 'train_TAG.txt', has_tag=True)
    dev_corpus, dev_tags = pre.preprocess_corpus('dev.txt', 'dev_TAG.txt', has_tag=True)
    train_dataset = NerDataset(train_corpus, train_tags, tokenizer)
    dev_dataset = NerDataset(dev_corpus, dev_tags, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=padding_collate, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, collate_fn=padding_collate, shuffle=True)
    model = BertCrf(output_size=len(pre.tag2id), tokenizer=tokenizer).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    f1_list = []
    loss_list = []

    for epoch in range(max_epoch):
        print(f'Epoch {epoch + 1}/{max_epoch}')
        print('-' * 10)

        model.train()
        tqdm_obj = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for i, (inputs, labels) in tqdm_obj:
            inputs = {key: val.to(device) for key, val in inputs.items()}
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = model(inputs, labels)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            tqdm_obj.set_postfix(loss=loss.item())

        model.eval()
        pred_list = []
        label_list = []
        with torch.no_grad():
            for inputs, labels in dev_dataloader:
                inputs = {key: val.to(device) for key, val in inputs.items()}
                labels = labels.to(device)
                pred = model(inputs, labels, is_predict=True)
                pred_list.extend(pred)
                label_list.extend(labels.tolist())


        f1 = calc_f1(pred_list, label_list)
        f1_list.append(f1)
        print(f'F1: {f1}')


    torch.save(model.state_dict(), save_path)

    print("Model saved successfully!")
    print("Model state:", model.state_dict())

    plt.plot(loss_list)
    plt.title('Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.savefig('loss_bert.jpg')
    plt.close()

    plt.plot(f1_list)
    plt.title('F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.savefig('F1_bert.jpg')
    plt.close()


def predict(model_path, save_path, bert_model_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pre = Preprocess(bert_model_path=bert_model_path)
    tokenizer = pre.tokenizer
    test_corpus = pre.preprocess_corpus('test.txt', has_tag=False)
    model = BertCrf(output_size=len(pre.tag2id), tokenizer=tokenizer).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    tag_list = []
    with torch.no_grad():
        for sentence in test_corpus:
            inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
            inputs = {key: val.to(device) for key, val in inputs.items()}
            pred = model(inputs, None, is_predict=True)
            tag_list.append(pred[0])

    with open(save_path, 'w', encoding='UTF-8') as f:
        for st_tag in tag_list:
            line = ' '.join(pre.id2tag[id] for id in st_tag)
            f.write(line + '\n')

if __name__ == '__main__':
    train(max_epoch=3, batch_size=512, bert_model_path='bert-base-chinese')
    predict('model/model_bert.pth', 'result/test_TAG_bert.txt', 'bert-base-chinese')