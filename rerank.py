import re
import copy
import threading
from queue import Queue

from utils.api_utils import api_single_request, prompt_format, embedding_multi_request
from utils.evaluate_util import cosine_similarity


instruct_prompt = "I will provide you with {num} documents, each indicated by number identifier []. \n\n " \
                  "Rank the documents based on the score according to the above rules. \n\n " \
                  "The question is: {question} \n\n " \
                  "The expansion is: {expansion}"
execute_prompt = "The documents should be listed in descending order using identifiers. The output format should be " \
                 "[]>[], e.g., [1]>[3]>[2]>[4]. The documents with the highest score should be listed first. Only " \
                 "response the ranking results, do not say any word or explain. Please make sure that you have return " \
                 "the whole ranking results of the {num} documents."


def create_permutation_instruction(documents=None, question=None, expansion=None, rank_start=0, rank_end=100,
                                   prompt_engine=None):
    messages = []
    prompt_template = prompt_engine.get_prompt(1)['template']
    prompt_standard = prompt_engine.get_prompt(1)['standard']
    system_prompt = prompt_template.replace("{standard}", prompt_standard)
    messages.append({
        'role': 'system',
        'content': system_prompt
    })
    messages.append({
        'role': 'user',
        'content': prompt_format(instruct_prompt, {'num': rank_end - rank_start,
                                                   'question': question,
                                                   'expansion': expansion})
    })
    messages.append({
        'role': 'assistant',
        'content': 'OK, I have got the task instruction. Please provide the documents.'
    })
    rank = 0
    for doc in documents[rank_start:rank_end]:
        messages.append({
            'role': 'user',
            'content': f"[{rank}] {doc['text']}"
        })
        messages.append({
            'role': 'assistant',
            'content': f'Received passage [{rank}].'
        })
        rank += 1
    messages.append({
        'role': 'user',
        'content': prompt_format(execute_prompt, {'num': rank_end - rank_start})
    })
    return messages


def format_response(response):
    # only extract the [digit]>[digit]>[digit] patterns in the response
    pattern = re.compile(r'\[\d+\](?:>\[\d+\])+')
    res = pattern.findall(response)
    if len(res) == 0:
        return None
    else:
        return res[0]


def validate_permutation(permutation, rank_len):
    flag = [0 for _ in range(rank_len)]
    res = []
    for order in permutation:
        if order < len(flag) and flag[order] == 0:
            res.append(order)
            flag[order] = 1
    for i in range(len(flag)):
        if flag[i] == 0:
            res.append(i)
    assert len(res) == rank_len, "Illegal permutation {}".format(permutation)
    return res


def permutation_step(documents=None, question=None, expansion=None, rank_start=0, rank_end=100,
                     engine=None, prompt_engine=None):
    messages = create_permutation_instruction(documents=documents, question=question, expansion=expansion,
                                              rank_start=rank_start, rank_end=rank_end,
                                              prompt_engine=prompt_engine)
    permutation = format_response(api_single_request(messages, model=engine))
    if permutation is not None:
        permutation = [int(item[1:-1]) for item in permutation.split('>')]
        permutation = validate_permutation(permutation, rank_end - rank_start)
    else:
        permutation = [i for i in range(rank_end - rank_start)]
    # order the documents[rank_start:rank_end] according to the permutation
    res = copy.deepcopy(documents)
    res[rank_start:rank_end] = [documents[rank_start + order] for order in permutation]
    return res


def sliding_windows_rerank(documents=None, question=None, expansion=None,
                           rank_start=0, rank_end=100, window_size=20, step=10, rerank_n=10,
                           engine=None, prompt_engine=None):
    item = copy.deepcopy(documents)
    end_pos = rank_end
    start_pos = rank_end - window_size
    iter_n = rerank_n // (window_size - step)
    for _ in range(iter_n):
        while True:
            start_pos = max(start_pos, rank_start)
            end_pos = min(end_pos, rank_end)
            item = permutation_step(item, question, expansion, start_pos, end_pos, engine, prompt_engine)
            if start_pos == rank_start:
                break
            start_pos = start_pos - step
            end_pos = start_pos + window_size
    return item


def sliding_windows_rerank_multi(question, expansion, documents, rerank_n, engine, prompt_engine, result_queue, rank):
    reranking_results = [sliding_windows_rerank(documents=documents, question=question, expansion=expansion,
                                                rank_start=0, rank_end=len(documents), rerank_n=rerank_n,
                                                engine=engine, prompt_engine=prompt_engine)
                         for _ in range(3)]
    reranking_results = [item[:rerank_n] for item in reranking_results]
    result_queue.put({
        'rank': rank,
        'res': reranking_results
    })
    return



def rerank_documents(questions, expansions, documents, engine, prompt_engine, rerank_n):
    threads = []
    unordered_result = Queue()
    results = []
    for i in range(len(questions)):
        question = questions[i]
        expansion = expansions[i]
        document = documents[i]
        t = threading.Thread(target=sliding_windows_rerank_multi, args=(question, expansion, document, rerank_n,
                                                                        engine, prompt_engine, unordered_result, i))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    while not unordered_result.empty():
        results.append(unordered_result.get())
    results = sorted(results, key=lambda x: x['rank'])
    results = [item['res'] for item in results]
    return results


def evaluate_reranking(questions, expansions, reranking_candidates):
    question_expansion = [q + ' ' + e for q, e in zip(questions, expansions)]
    question_expansion_embeddings = embedding_multi_request(question_expansion)
    reranking_results = []
    candidate_n = len(reranking_candidates[0])
    for candidate in reranking_candidates:
        for item in candidate:
            reranking_results.append('\n'.join([doc['text'] for doc in item]))
    reranking_embeddings = embedding_multi_request(reranking_results)
    reranking_scores = []
    for i in range(len(questions)):
        qe_emb = question_expansion_embeddings[i]
        rerank_embedding = reranking_embeddings[i * candidate_n: (i + 1) * candidate_n]
        scores = [cosine_similarity(qe_emb, item) for item in rerank_embedding]
        reranking_scores.append(scores)
    return reranking_scores

