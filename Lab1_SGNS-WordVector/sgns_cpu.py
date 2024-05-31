import math
import torch
import random
import numpy as np
from tqdm import tqdm
from queue import PriorityQueue
from torch.utils.data import Dataset, DataLoader


class SkipGram(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size, init_range=0.1):
        """
        vocab_size: 整数，词汇表的大小
        embedding_size: 整数，词向量的维度
        init_range: 浮点数，用于初始化词向量的范围，默认为0.1
        """
        super(SkipGram, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.embedding = torch.nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_size,
            padding_idx=None,
            max_norm=None,
            norm_type=2,
            scale_grad_by_freq=False,
            sparse=False,
            _weight=torch.FloatTensor(vocab_size, embedding_size).uniform_(-init_range, init_range)
        ).cpu()

        self.embedding_out = torch.nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_size,
            padding_idx=None,
            max_norm=None,
            norm_type=2,
            scale_grad_by_freq=False,
            sparse=False,
            _weight=torch.FloatTensor(vocab_size, embedding_size).uniform_(-init_range, init_range)
        ).cpu()


    def forward(self, center_words, target_words, label):
        """
        center_words: Tensor，表示中心词的索引
        target_words: Tensor，表示目标词的索引
        label: Tensor，表示目标词是正样本（1）还是负样本（0）
        """        
        center_words_emb = self.embedding(center_words)
        target_words_emb = self.embedding_out(target_words)
        word_sim = torch.sum(center_words_emb * target_words_emb, dim=-1)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn(word_sim, label.float())
        return loss


class SkipGramDataset(Dataset):
    def __init__(self, corpus, window_size, negative_sample_num):
        """
        corpus: 列表，表示语料库
        window_size: 整数，表示上下文窗口大小
        negative_sample_num: 整数，表示负样本数量
        """
        self.corpus = corpus
        self.window_size = window_size
        self.negative_sample_num = negative_sample_num
        self.positive_target = dict()
        for index in range(len(corpus)):
            positive_word_range = (max(0, index - window_size), min(len(corpus) - 1, index + window_size))
            if not corpus[index] in self.positive_target:
                self.positive_target[corpus[index]] = set()
            for i in range(positive_word_range[0], positive_word_range[1] + 1):
                self.positive_target[corpus[index]].add(corpus[i])

    def __len__(self):
        return len(self.corpus) * (2 * self.window_size + self.negative_sample_num)

    def __getitem__(self, index):
        if index < len(self.corpus) * 2 * self.window_size:
            center_index = index // (2 * self.window_size)
            shift_index = index % (2 * self.window_size) - self.window_size
            if shift_index >= 0:
                shift_index += 1
            target_index = max(0, min(len(self.corpus) - 1, center_index + shift_index))
            if target_index == center_index:
                if target_index - 1 >= 0:
                    target_index -= 1
                else:
                    target_index += 1
            return self.corpus[center_index], self.corpus[target_index], 1
        else:
            center_index = (index - len(self.corpus) * 2 * self.window_size) // self.negative_sample_num
            while True:
                target_index = random.randint(0, len(self.corpus) - 1)
                if self.corpus[target_index] not in self.positive_target[self.corpus[center_index]]:
                    return self.corpus[center_index], self.corpus[target_index], 0


class SGNS:
    def __init__(self, train_path: str = 'data/lmtraining.txt', window_size=2, negative_sample_num=4, embedding_size=100, device='cpu', has_train=False):
        """
        train_path: 字符串，训练数据文件路径，默认为'data/lmtraining.txt'
        window_size: 整数，上下文窗口大小，默认为2
        negative_sample_num: 整数，负样本数量，默认为4
        embedding_size: 整数，词向量维度，默认为100
        device: 字符串，指定训练使用的设备，'cuda'或'cpu'，默认为'cuda'
        has_train: 布尔值，指示是否已经训练了模型，默认为False
        """
        self.embeddings = None
        self.dataset = None
        self.dataloader = None
        self.device = device
        self.corpus = []
        with open(train_path, 'r', encoding='UTF-8') as f:
            self.corpus = f.read().strip("\n")
            f.close()

        self.preprocess_corpus()

        self.id2word_dict = {}
        self.word2id_dict = {}
        self.freq_dict = {}
        self.corpus_size = 0
        self.create_id()
        self.vocab_size = len(self.word2id_dict)
        print(f'There are {self.vocab_size} words in the corpus.')

        if not has_train:
            self.subsampling()
            self.dataset = SkipGramDataset(self.corpus, window_size, negative_sample_num)
            self.dataloader = DataLoader(self.dataset, batch_size=512, shuffle=True)
        self.model = SkipGram(self.vocab_size, embedding_size).to(self.device)

    def preprocess_corpus(self):
        self.corpus = self.corpus.strip().lower()
        self.corpus = self.corpus.split(" ")

    def create_id(self):
        for word in self.corpus:
            self.corpus_size += 1
            if word not in self.freq_dict:
                self.freq_dict[word] = 1
            else:
                self.freq_dict[word] += 1

        temp_dict = sorted(self.freq_dict.items(), key=lambda x: x[1], reverse=True)

        for word, _ in temp_dict:
            my_id = len(self.word2id_dict)
            self.word2id_dict[word] = my_id
            self.id2word_dict[my_id] = word

    def subsampling(self):
        temp_corpus = self.corpus
        self.corpus = []
        for word in temp_corpus:
            drop = random.uniform(0, 1) < 1 - math.sqrt(1e-4 / self.freq_dict[word] * self.corpus_size)
            if not drop:
                self.corpus.append(self.word2id_dict[word])

    def train(self, epoch_num, save_path='model/SGNS.pth'):
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=0.001)
        loss = -1
        for epoch in range(epoch_num):
            for _, (center_word, target_word, is_positive) in enumerate(tqdm(self.dataloader)):
                center_word = center_word.to('cpu')
                target_word = target_word.to('cpu')
                is_positive = is_positive.to('cpu')
                loss = self.model(center_word, target_word, is_positive)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f'The loss of Epoch {epoch + 1} = {loss:>7f}')
        torch.save(self.model.state_dict(), save_path)
        self.embeddings = self.model.embedding.cpu().weight.data.numpy()

    def load_model(self, model_path='model/SGNS.pth'):
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.embeddings = self.model.embedding.cpu().weight.data.numpy()

    def get_cos_sim(self, word1, word2):
        if (word1 not in self.word2id_dict) or (word2 not in self.word2id_dict):
            return 0
        word1_vec = self.embeddings[self.word2id_dict[word1]]
        word2_vec = self.embeddings[self.word2id_dict[word2]]
        cos_sim = np.dot(word1_vec, word2_vec) / (np.linalg.norm(word1_vec) * np.linalg.norm(word2_vec))
        return cos_sim


def get_sgns_ans(has_train=True, epoch_num=5, model_path='model/SGNS.pth', test_path='data/wordsim353_agreed.txt', result_path='data/sgns_ans.txt'):
    """
    has_train: 布尔值，指示是否已经训练了模型，默认为True
    epoch_num: 整数，表示训练的轮数，默认为5
    model_path: 字符串，表示模型路径，默认为'model/SGNS.pth'
    test_path: 字符串，表示测试数据文件路径，默认为'data/wordsim353_agreed.txt'
    result_path: 字符串，表示结果保存路径，默认为'data/sgns_ans.txt'
    """    
    if not has_train:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        my_sgns = SGNS(device='cuda', has_train=False)
        my_sgns.train(epoch_num, save_path=model_path)
    else:
        my_sgns = SGNS(device='cuda', has_train=True)
        my_sgns.load_model(model_path)

    with open(test_path, 'r', encoding='UTF-8') as f:
        test_lines = f.readlines()
        f.close()
    f = open(result_path, 'w', encoding='UTF-8')
    for i in range(len(test_lines)):
        line = test_lines[i].strip('\n').split('\t')
        if len(line) == 0:
            continue
        word1 = line[1]
        word2 = line[2]
        true_result = line[3]
        sim_sgns = my_sgns.get_cos_sim(word1, word2)
        f.write(f'{word1}\t{word2}\t{true_result}\t{(sim_sgns + 1) * 5:.2f}\n')
    f.close()


def top5(word, has_train=True, epoch_num=5, model_path='model/SGNS.pth'):
    if not has_train:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        my_sgns = SGNS(device='cuda', has_train=False)
        my_sgns.train(epoch_num, save_path=model_path)
    else:
        my_sgns = SGNS(device='cuda', has_train=True)
        my_sgns.load_model(model_path)
    if word not in my_sgns.word2id_dict:
        print(f'Sorry, "{word}" is not found in the corpus.')
        return
    q = PriorityQueue()
    for i in range(my_sgns.vocab_size):
        word2 = my_sgns.id2word_dict[i]
        if word2 == word:
            continue
        sim_sgns = my_sgns.get_cos_sim(word, word2)
        q.put((-sim_sgns, word2))
    print(f'The most similar 5 words to "{word}" are:')
    for i in range(5):
        next_item = q.get()
        print(f'{-next_item[0]:.2f}\t{next_item[1]}')


def main():
    get_sgns_ans(has_train=True, test_path='data/wordsim353_agreed.txt', epoch_num=5, model_path='model/SGNS.pth')

main()