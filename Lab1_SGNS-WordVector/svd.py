import numpy as np
import scipy
from queue import PriorityQueue

class SVD:
    def __init__(self, train_path: str = 'data/lmtraining.txt', vocab_max_size=10000):
        """
        train_path: 字符串，指定用于训练的文本文件路径，默认为'data/lmtraining.txt'
        vocab_max_size: 整数，指定词汇表的最大大小，默认为10000
        """
        self.word_vectors = None
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
        self.vocab_size = len(self.word2id_dict) if vocab_max_size >= len(self.word2id_dict) or vocab_max_size == 0 else vocab_max_size
        print(f'There are {self.vocab_size} words in the corpus.')

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

        self.freq_dict = sorted(self.freq_dict.items(), key=lambda x: x[1], reverse=True)

        for word, _ in self.freq_dict:
            my_id = len(self.word2id_dict)
            self.word2id_dict[word] = my_id
            self.id2word_dict[my_id] = word

    def build_svd_vec(self, save_path='model/svd.npy', vector_dim=100, window_size=5):
        """
        save_path: 字符串，指定保存SVD模型的路径，默认为'model/svd.npy'
        vector_dim: 整数，指定词向量的维度，默认为100
        window_size: 整数，指定窗口大小，默认为5
        """
        co_matrix = np.zeros((self.vocab_size, self.vocab_size), dtype='uint16')
        for center_word_idx in range(len(self.corpus)):
            center_word = self.corpus[center_word_idx]
            if self.word2id_dict[center_word] >= self.vocab_size:
                continue
            context_words_list = self.corpus[max(0, center_word_idx - window_size): center_word_idx] + self.corpus[center_word_idx + 1: center_word_idx + window_size + 1]
            for context_word in context_words_list:
                if self.word2id_dict[context_word] < self.vocab_size:
                    co_matrix[self.word2id_dict[center_word], self.word2id_dict[context_word]] += 1
        print('Begin to calculate SVD...')
        co_matrix = scipy.sparse.csr_matrix(co_matrix).asfptype()
        U, S, V = scipy.sparse.linalg.svds(co_matrix, k=vector_dim)
        print(f'There are {len(S)} singular values, sum of them is {np.sum(S)}.')
        self.word_vectors = U

        np.save(save_path, np.array(self.word_vectors))

    def load_svd_vec(self, model_path='model/svd.npy'):
        self.word_vectors = np.load(model_path)

    def get_cos_sim(self, word1, word2):
        """
        word1: 字符串，指定第一个词
        word2: 字符串，指定第二个词
        """
        if (word1 not in self.word2id_dict) or (word2 not in self.word2id_dict) or (self.word2id_dict[word1] >= self.vocab_size) or (
                self.word2id_dict[word2] >= self.vocab_size):
            return 0
        word1_vec = self.word_vectors[self.word2id_dict[word1]]
        word2_vec = self.word_vectors[self.word2id_dict[word2]]
        cos_sim = np.dot(word1_vec, word2_vec) / (np.linalg.norm(word1_vec) * np.linalg.norm(word2_vec))
        return cos_sim


def get_svd_ans(has_train=True, vocab_max_size=100000, vector_dim=100, window_size=5, model_path='model/svd.npy', test_path='data/wordsim353_agreed.txt', result_path='data/svd_ans.txt'):
    """
    has_train: 布尔值，指示是否已经训练了SVD模型，默认为True
    vocab_max_size: 整数，指定词汇表的最大大小，默认为100000
    vector_dim: 整数，指定词向量的维度，默认为100
    window_size: 整数，指定窗口大小，默认为5
    model_path: 字符串，指定SVD模型的路径，默认为'model/svd.npy'
    test_path: 字符串，指定测试数据文件的路径，默认为'data/wordsim353_agreed.txt'
    result_path: 字符串，指定结果保存的路径，默认为'data/svd_ans.txt'
    """
    if not has_train:
        svd = SVD(train_path='data/lmtraining.txt', vocab_max_size=vocab_max_size)
        svd.build_svd_vec(save_path=model_path, vector_dim=vector_dim, window_size=window_size)
    else:
        svd = SVD(train_path='data/lmtraining.txt', vocab_max_size=vocab_max_size)
        svd.load_svd_vec(model_path)

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
        sim_sgns = svd.get_cos_sim(word1, word2)
        f.write(f'{word1}\t{word2}\t{true_result}\t{(sim_sgns + 1) * 5:.2f}\n')
    f.close()


def top5(word, has_train=True, vocab_max_size=100000, vector_dim=100, window_size=5, model_path='model/svd.npy'):
    """
    word: 字符串，指定要查询相似词的词
    has_train: 布尔值，指示是否已经训练了SVD模型，默认为True
    vocab_max_size: 整数，指定词汇表的最大大小，默认为100000
    vector_dim: 整数，指定词向量的维度，默认为100
    window_size: 整数，指定窗口大小，默认为5
    model_path: 字符串，指定SVD模型的路径，默认为'model/svd.npy'
    """
    if not has_train:
        svd = SVD(train_path='data/lmtraining.txt', vocab_max_size=vocab_max_size)
        svd.build_svd_vec(save_path=model_path, vector_dim=vector_dim, window_size=window_size)
    else:
        svd = SVD(train_path='data/lmtraining.txt', vocab_max_size=vocab_max_size)
        svd.load_svd_vec(model_path)
    if word not in svd.word2id_dict:
        print(f'Sorry, "{word}" is not found in the corpus.')
        return
    q = PriorityQueue()
    for i in range(svd.vocab_size):
        word2 = svd.id2word_dict[i]
        if word2 == word:
            continue
        sim_sgns = svd.get_cos_sim(word, word2)
        q.put((-sim_sgns, word2))
    print(f'The most similar 5 words to "{word}" are:')
    for i in range(5):
        next_item = q.get()
        print(f'{-next_item[0]:.2f}\t{next_item[1]}')


def main():
    get_svd_ans(has_train=False, test_path='data/wordsim353_agreed.txt', vocab_max_size=100000)

main()