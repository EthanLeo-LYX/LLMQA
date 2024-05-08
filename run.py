import os
import argparse
import json
import tqdm

from expansion import generate_expansion, evaluate_expansion
from rerank import rerank_documents, evaluate_reranking
from utils.prompt_util import PromptEngine


def get_mini_batch(inlines, batch_size, index):
    batch = []
    # generate mini batch
    for _ in range(batch_size):
        if index >= len(inlines):
            break
        batch.append(inlines[index])
        index += 1

    return batch


def expansion_generation(inlines, engine, prompt_engine, output_dir):
    output_file = open(os.path.join(output_dir, '0-expansion_generation.jsonl'), 'a')
    questions = [line['question'] for line in inlines]
    batch_expansions = generate_expansion(questions, engine, prompt_engine)
    for line, expansions in zip(inlines, batch_expansions):
        line['expansion_candidates'] = expansions
        output_file.write(json.dumps(line) + '\n')
    return inlines


def expansion_evaluation(inlines, output_dir):
    output_file = open(os.path.join(output_dir, '1-expansion_evaluation.jsonl'), 'a')
    questions = [line['question'] for line in inlines]
    expansion_candidates = [line['expansion_candidates'] for line in inlines]
    expansion_scores = evaluate_expansion(questions, expansion_candidates)
    for line, score in zip(inlines, expansion_scores):
        best_expansion = line['expansion_candidates'][score.index(max(score))]
        line['expansion_scores'] = score
        line['expansion'] = best_expansion
        output_file.write(json.dumps(line) + '\n')
    return inlines


def document_reranking(inlines, engine, prompt_engine, rerank_n, output_dir):
    output_file = open(os.path.join(output_dir, '2-document_reranking.jsonl'), 'a')
    questions = [line['question'] for line in inlines]
    documents = [line['ctxs'] for line in inlines]
    expansions = [line['expansion'] for line in inlines]
    reranking_results = rerank_documents(questions, expansions, documents, engine, prompt_engine, rerank_n)
    for line, rerank_result in zip(inlines, reranking_results):
        line['reranking_candidates'] = rerank_result
        output_file.write(json.dumps(line) + '\n')
    return inlines


def reranking_evaluation(inlines, output_dir):
    output_file = open(os.path.join(output_dir, '3-reranking_evaluation.jsonl'), 'a')
    questions = [line['question'] for line in inlines]
    expansions = [line['expansion'] for line in inlines]
    reranking_candidates = [line['reranking_candidates'] for line in inlines]
    reranking_scores = evaluate_reranking(questions, expansions, reranking_candidates)
    for line, score in zip(inlines, reranking_scores):
        best_candidate = line['reranking_candidates'][score.index(max(score))]
        line['reranking_scores'] = score
        line['rerank'] = best_candidate
        output_file.write(json.dumps(line) + '\n')
    return inlines


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--input_dir", default='input_data', type=str,
                        help="The input data dir")
    parser.add_argument("--output_dir", default='output_result', type=str,
                        help="The output result dir")
    parser.add_argument("--prompt_dir", default='prompt_templates', type=str,
                        help="The prompt dir")
    parser.add_argument("--dataset", default='nq', type=str,
                        help="dataset name: [nq, tqa, webq]")
    parser.add_argument("--split", default='test', type=str,
                        help="dataset split: [train, dev, test]")
    parser.add_argument("--engine", default='gpt-3.5-turbo-16k', type=str,
                        help="gpt-3.5-turbo-16k/gpt-4/text-davinci-003?")
    parser.add_argument("--batch_size", default=10, type=int)
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--rerank_n", default=10, type=int)

    args = parser.parse_args()

    # load dataset
    input_file = os.path.join(args.input_dir, args.dataset, args.split + '.json')
    with open(input_file, 'r', encoding='utf8') as fin:
        inlines = json.load(fin)

    # load prompts
    prompt_engine = PromptEngine(args.prompt_dir)

    # output dir
    output_dir = os.path.join(args.output_dir, args.dataset, args.split)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pbar = tqdm.tqdm(total=len(inlines))
    index = 0
    pbar.update(index)
    while index < len(inlines):
        # generate mini batch
        batch_inlines = get_mini_batch(inlines, args.batch_size, index)
        index += len(batch_inlines)

        batch_inlines = expansion_generation(batch_inlines, args.engine, prompt_engine, output_dir)
        batch_inlines = expansion_evaluation(batch_inlines, output_dir)
        batch_inlines = document_reranking(batch_inlines, args.engine, prompt_engine, args.rerank_n, output_dir)
        batch_inlines = reranking_evaluation(batch_inlines, output_dir)

        pbar.update(len(batch_inlines))


if __name__ == "__main__":
    main()
