#!/usr/bin/env python2.7
import click
import logging
import pandas as pd
import pomegranate as pg
import ujson as json
from tqdm import tqdm
from itertools import product, repeat, izip
from multiprocessing import Pool

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s : %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger()


def get_answer(sentence, model):
    if any(len(w["v"]) > 1 and len(w["a"]) == 0 for w in sentence["words"]):
        logger.debug("Sentence %d has unresolved words, skip", sentence["id"])
        return

    result = []
    word_ids = []
    parts = []
    current_part = []
    unambig_count = 0
    for w in sentence["words"]:
        result.append(w["v"][0] if len(w["v"]) == 1 else w["a"][0])
        word_ids.append(w["id"])

        if len(w["v"]) > 1:
            unambig_count += 1
            current_part.append(w)
        else:
            if len(current_part) > 0:
                parts.append(current_part)
            current_part = []
    if len(current_part) > 0:
        parts.append(current_part)

    result = pd.DataFrame(result, index=word_ids)

    answers = []
    for part in parts:
        word_ids = [w["id"] for w in part]
        temp = result.copy()

        max_prob, max_answer = None, None
        for answer in product(*[w["v"] for w in part]):
            answer = pd.DataFrame(list(answer), index=word_ids)
            temp.loc[answer.index, :] = answer
            prob = model.log_probability(temp.values)
            if max_prob is None or prob > max_prob:
                max_prob = prob
                max_answer = answer.copy()
        answers.append(max_answer)

    temp = result.copy()
    for answer in answers:
        temp.loc[answer.index,:] = answer

    return result != temp


def process(args):
    (line, model) = args
    sentence = json.loads(line)
    errors = get_answer(sentence, model)
    if errors is None:
        return

    results = dict()
    results["words"] = errors.shape[0]
    results["errors_by_tag"] = list(errors.sum(axis=0).values)
    results["errors"] = errors.any(axis=1).sum()

    sent_unambig = []
    sent_unambig_all = 0
    for word in sentence["words"]:
        if len(word["v"]) > 1:
            sent_unambig_all += 1
            variants = pd.DataFrame(word["v"])
            sent_unambig.append(variants.nunique() > 1)
    if sent_unambig_all > 0:
        results["unambig"] = sent_unambig_all
        results["unambig_by_tag"] = list((pd.concat(sent_unambig, axis=1)).sum(axis=1).values)
    results["id"] = sentence["id"]
    results["text"] = sentence["text"]

    return results


@click.command()
@click.argument("sentences", type=click.File("rt", encoding="utf8"))
@click.argument("model", type=click.File("rt", encoding="utf8"))
@click.argument("output", type=click.File("wt"))
def main(sentences, model, output):
    model = pg.HiddenMarkovModel.from_json(model.read())
    pool = Pool(processes=5)

    for result in tqdm(pool.imap_unordered(process, izip(sentences, repeat(model))), total=114883):
        if result is not None:
            output.write(json.dumps(result))
            output.write("\n")
    output.close()

if __name__ == "__main__":
    main()
