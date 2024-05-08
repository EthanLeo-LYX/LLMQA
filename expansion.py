from utils.evaluate_util import cosine_similarity, sigmoid
from utils.api_utils import api_multi_request, embedding_multi_request


def generate_expansion(questions, engine, prompt_engine):
    prompt_standard = prompt_engine.get_prompt(0)['standard']
    prompt_template = prompt_engine.get_prompt(0)['template']
    prompt = prompt_template.replace("{standard}", prompt_standard)
    prompts = [prompt.format(question=q) for q in questions]
    responses = api_multi_request(prompts, engine, candidate_n=10)
    return responses


def evaluate_expansion(questions, expansions):
    question_embeddings = embedding_multi_request(questions)
    expansion_embeddings = [embedding_multi_request(expansion) for expansion in expansions]
    expansion_scores = []
    for q_emb, e_emb in zip(question_embeddings, expansion_embeddings):
        score = [sigmoid(cosine_similarity(q_emb, e)) for e in e_emb]
        expansion_scores.append(score)
    return expansion_scores
