from FlagEmbedding import FlagReranker
# # 可以测试 base, large 等不同模型的效果
reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)

score = reranker.compute_score(['query', 'passage'])
print("score: ", score)

scores = reranker.compute_score([['what is panda?', 'hi'], 
                                 ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']])
print("scores: ", scores)

"""
score:  -1.529296875
scores:  [-5.6171875, 5.765625]
"""