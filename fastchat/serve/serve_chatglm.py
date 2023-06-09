import torch
from typing import List, Tuple
import logging
import random
from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    utility
)

_COLLECTION_NAME = 'demo'
_ID_FIELD_NAME = 'sentence_id'
_VECTOR_FIELD_NAME = 'embedding'
_HOST = '127.0.0.1'
_PORT = '19530'

# Vector parameters
_DIM = 768
_INDEX_FILE_SIZE = 32  # max file size of stored index

# Index parameters
_METRIC_TYPE = 'IP'
_INDEX_TYPE = 'IVF_FLAT'
_NLIST = 1024
_NPROBE = 16
_TOPK = 5
_COLLECTION_NAME = 'demo'
_ID_FIELD_NAME = 'sentence_id'
_VECTOR_FIELD_NAME = 'embedding'

def search(collection, vector_field, id_field, search_vectors):
    search_param = {
        "data": search_vectors,
        "anns_field": vector_field,
        "param": {"metric_type": _METRIC_TYPE, "params": {"nprobe": _NPROBE}},
        "limit": _TOPK}
        #"expr": "id_field >= 0"}
    results = collection.search(**search_param)
   # print(results)
    for i, result in enumerate(results):
        print("\nSearch result for {}th vector: ".format(i))
        for j, res in enumerate(result):
            print("Top {}: {}".format(j, res))
    random_idx = random.randint(0, _TOPK-1)
    return results[0][random_idx].id

def post_process(text):
    text = text.replace("ChatGLM-6B", "情绪哥")
    text = text.replace("ChatGLM", "情绪哥")
    text = text.replace("智谱 AI 公司", "情绪哥语料团队")
    text = text.replace("清华大学 KEG 实验室", "情绪哥AI团队")
    return text

def get_mood(model, query, tokenizer):
    prompt = query + "。请判断上述语句是不是带有负面情绪"
    print("prompt:{}".format(prompt))
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(inputs.input_ids, max_new_tokens=8)
    result = tokenizer.batch_decode(outputs)
    result = result[0][len(prompt):].strip()
    logging.warning("query: {} mood result:{}".format(query, result))
    if result.find("无法判断") != -1:
        return 0
    elif result.find('没有负面') != -1 or result.find('没有') != -1:
        return 0
    elif result.startswith('是') or result.find('有负面') != -1:
        return 1
    return 0
    

    


@torch.inference_mode()
def chatglm_generate_stream(model, t2v_model, milvus_collection, sentences, tokenizer, params, device,
                            context_len=2048, stream_interval=2):
    """Generate text using model's chat api"""
    print(t2v_model)
    messages = params["prompt"]
    max_new_tokens = int(params.get("max_new_tokens", 256))
    temperature = float(params.get("temperature", 1.0))
    top_p = float(params.get("top_p", 0.7))
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "top_p": top_p,
        "temperature": temperature,
        "logits_processor": None
    }

    hist = []
    for i in range(0, len(messages) - 2, 2):
        hist.append((messages[i][1], messages[i+1][1]))
    query = messages[-2][1]
    mood_result = get_mood(model, query, tokenizer)
    logging.warning("mood result:{}".format(mood_result))
    if mood_result == 1:
        embedding = [list(t2v_model.encode(query))]
        print("embeding", embedding)
        print(f"query:{query}")
        print("sentences len:{}".format(len(sentences)))
        doc_id = search(milvus_collection, _VECTOR_FIELD_NAME, _ID_FIELD_NAME, embedding)
        print("selected doc id:{} selected sentence:{}", doc_id, sentences[doc_id])
        inner_response = sentences[doc_id] + "(内部语料)"

    for response, new_hist in model.stream_chat(tokenizer, query, hist):
        if mood_result == 1:
            response = inner_response +"\n\n" +response
        else:
            response = post_process(response)
        output = query + " " + response
        yield output
