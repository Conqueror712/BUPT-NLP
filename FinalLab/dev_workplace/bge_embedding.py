from FlagEmbedding import FlagModel

sentences_1 = ["样例数据-1", "样例数据-2"]
sentences_2 = ["样例数据-3", "样例数据-4"]
# 可以测试 small, base, large 等不同模型的效果
model = FlagModel('BAAI/bge-large-zh-v1.5', 
                  query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                  use_fp16=True) # 将 use_fp16 设置为 True 可以提高计算速度，但性能略有下降
embeddings_1 = model.encode(sentences_1)
embeddings_2 = model.encode(sentences_2)
similarity = embeddings_1 @ embeddings_2.T
print("similarity = \n", similarity)

# 对于 s2p 检索任务，建议使用 encode_queries()，它将自动向每个查询添加指令
# 检索任务中的语料库仍然可以使用 encode() 或 encode_corpus()，因为它们不需要指令
queries = ['query_1', 'query_2']
passages = ["样例文档-1", "样例文档-2"]
q_embeddings = model.encode_queries(queries)
p_embeddings = model.encode(passages)
scores = q_embeddings @ p_embeddings.T
print("scores = \n", scores)

"""
similarity =
 [[0.855  0.852 ]
 [0.874  0.8555]]
scores =
 [[0.337  0.2048]
 [0.2258 0.3848]]
"""