import openai
import threading
import time
from queue import Queue

openai.api_key = ''
openai.api_base = ''


def rmreturn(s):
    s = s.replace('\n\n', ' ')
    s = s.replace('\n', ' ')
    return s.strip()


def prompt_format(prompt_template, prompt_content):
    return prompt_template.format(**prompt_content)


def api_single_request(message, model="gpt-3.5-turbo-16k", max_tokens=128, temperature=0.7, candidate_n=1,
                       rank=-1, result_queue=None):
    request_cnt = 0
    while True:
        request_cnt += 1
        if request_cnt > 20:
            exit(0)
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=message,
                max_tokens=max_tokens,
                temperature=temperature,
                n=candidate_n)
            if candidate_n == 1:
                return_response = rmreturn(response.choices[0].message.content)
            else:
                return_response = [rmreturn(response.choices[i].message.content) for i in range(candidate_n)]
            # single thread request
            if rank == -1:
                return return_response
            # multi thread request
            else:
                result_queue.put({
                    'rank': rank,
                    'response': return_response
                })
                return
        except Exception as e:
            # raise e
            print("API ERROR!")
            time.sleep(1)
            continue


def api_multi_request(prompts, model="gpt-3.5-turbo-16k", max_token=128, temperature=0.7, candidate_n=1):
    threads = []
    result_queue = Queue()
    gathered_response = []
    for i in range(len(prompts)):
        message = [{'role': 'user', 'content': prompts[i]}]
        t = threading.Thread(target=api_single_request,
                             args=(message, model, max_token, temperature, candidate_n, i, result_queue))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    while not result_queue.empty():
        gathered_response.append(result_queue.get())
    assert len(gathered_response) == len(prompts)
    gathered_response.sort(key=lambda x: x['rank'])
    gathered_response = [x['response'] for x in gathered_response]
    return gathered_response


def embedding_single_request(text):
    request_cnt = 0
    while True:
        request_cnt += 1
        if request_cnt > 20:
            exit(0)
        try:
            response = openai.Embedding.create(
                model='text-embedding-ada-002',
                input=text
            )
            embedding = response['data'][0]['embedding']
            return embedding
        except Exception as e:
            # raise e
            print('API Error')
            time.sleep(1)
            continue


def embedding_multi_request(texts):
    request_cnt = 0
    while True:
        request_cnt += 1
        if request_cnt > 20:
            exit(0)
        try:
            response = openai.Embedding.create(
                model='text-embedding-ada-002',
                input=texts
            )
            embeddings = [response['data'][i]['embedding'] for i in range(len(texts))]
            return embeddings
        except Exception as e:
            print('API Error')
            time.sleep(1)
            continue
