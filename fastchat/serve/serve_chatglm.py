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
from text2vec import SentenceModel

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

def search(collection, vector_field, out_field, search_vectors):
    search_param = {
        "data": search_vectors,
        "anns_field": vector_field,
        "param": {"metric_type": _METRIC_TYPE, "params": {"nprobe": _NPROBE}},
        "limit": _TOPK,
        "output_fields": [out_field]}
        #"expr": "id_field >= 0"}
    results = collection.search(**search_param)
   # print(results)
    for i, result in enumerate(results):
        print("\nSearch result for {}th vector: ".format(i))
        for j, res in enumerate(result):
            print("Top {}: {}".format(j, res))
    random_idx = random.randint(0, len(results[0]) - 1)
    return results[0][random_idx]

def post_process(text):
    text = text.replace("ChatGLM-6B", "情绪哥")
    text = text.replace("ChatGLM", "情绪哥")
    text = text.replace("智谱 AI 公司", "情绪哥语料团队")
    text = text.replace("清华大学 KEG 实验室", "情绪哥AI团队")
    return text

def get_mood(model, query, tokenizer):
    prompt = f'''请判断下面给出的描述是否含有负面情绪，请回答“含”或者“不含”。
    ```
    {query}
    ```'''
    print("prompt:{}".format(prompt))
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(inputs.input_ids, max_new_tokens=8)
    result = tokenizer.batch_decode(outputs)
    result = result[0][len(prompt):].strip()
    logging.warning("query: {} mood result:{}".format(query, result))
    if result.find("不含") != -1:
        return 0
    else:
        return 1

def get_education(model, query, tokenizer):
    prompt = '请判断下面给出的描述是否涉及子女教育，\n请回答“涉及”或者“不涉及”\n```\n'+query+'\n```\n'
    print("prompt:{}".format(prompt))
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(inputs.input_ids, max_new_tokens=8)
    result = tokenizer.batch_decode(outputs)
    result = result[0][len(prompt):].strip()
    logging.warning("query: {} get_education result:{}".format(query, result))
    if result.find("不涉及") != -1:
        return 0
    else:
        return 1
    

    


@torch.inference_mode()
def chatglm_generate_stream(model, t2v_models, milvus_collections, sentences, tokenizer, params, device,
                            context_len=2048, stream_interval=2):
    """Generate text using model's chat api"""
    print(t2v_models)
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
        embeddings = t2v_models[0].encode([query])
        content = search(milvus_collections[0], 'vector', 'content', embeddings).entity.get('content')
        print(f"selected sentence:{content}")
        inner_response = content + '[内部语料]'
    else:
        education_result = get_education(model, query, tokenizer)
        logging.warning("education result:{}".format(education_result))
        if education_result == 1:
            embeddings = t2v_models[1].encode([query])
            content = search(milvus_collections[1], 'vector', 'content', embeddings).entity.get('content')
            query = f'''请总结下面的材料内容来回答问题，尽量简洁，50个字以内。材料以“```”开始和结束，问题以“///”开始和结束。
            材料：
            ```{content}```
            问题：
            ///{query}///'''
            logging.warning("education query is:" + query)
        else:
            query = f'''
            请回答下面的问题，尽量简洁，200个字以内。
            问题:
            ```{query}```
            '''
    logging.warning(f'full query is: {query}, negative emotion:{mood_result}, about education {education_result}')
    for response, new_hist in model.stream_chat(tokenizer, query, hist):
        if mood_result == 1:
            response = f'{inner_response}\n\n{response}'
        else:
            response = post_process(response)
            if education_result == 1:
                response += '[书籍参考]'

        output = response
        yield output
