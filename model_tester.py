#!/usr/bin/env python2.7
import os
import click
import logging
import numpy as np
import pandas as pd
import pomegranate as pg
import ujson as json
from collections import defaultdict
from itertools import product, izip, imap, cycle, ifilter
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s : %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger()


def load_tests(line, texts):
    sentence = json.loads(line)
    if sentence["text"] not in texts:
        return
    if any(len(w["v"]) > 1 and len(w["a"]) == 0 for w in sentence["words"]):
        logger.debug("Sentence %d has unresolved words, skip", sentence["id"])
        return

    unambig_word_count = sum(len(w["v"]) > 1 for w in sentence["words"])
    total_word_count = len(sentence["words"])

    parts = []
    current_part = []
    for w in sentence["words"]:
        if len(w["v"]) > 1:
            current_part.append(w)
        elif len(current_part) == 0 or len(current_part[-1]["v"]) > 1:
            current_part.append(w)
        else:
            parts.append(current_part)
            current_part = [w]
    parts.append(current_part)

    assert sum(map(len, parts)) == total_word_count

    parts = filter(lambda p: len(p) >= 1, parts)

    chains = []
    for part in parts:
        result = np.array([w["v"][0] if len(w["v"]) == 1 else w["a"][0] for w in part])
        answers = imap(np.array, product(*[w["v"] for w in part]))
        chains.append((result, answers))

    return sentence["id"], total_word_count, unambig_word_count, chains


@click.command()
@click.argument("sentences", type=click.File("rt", encoding="utf8"))
@click.argument("model", type=click.File("rt", encoding="utf8"))
@click.option("--texts", nargs=2, default=[1, 10], type=int)
def main(sentences, model, texts):
    texts = set(range(texts[0], texts[1] + 1))

    logger.info("Texts: %s", sorted(list(texts)))

    model = pg.HiddenMarkovModel.from_json(model.read())
    metrics = defaultdict(int)

    with tqdm() as pbar:
        for l in sentences:
            item = load_tests(l, texts)
            if item is None:
                continue
            (sid, total_word_count, unambig_word_count, samples) = item
            pbar.update(1)

            errors = 0
            for result, sample in samples:
                max_prob, max_chain = None, None
                result_cnt = 0
                for chain in sample:
                    assert chain.shape == result.shape
                    result_cnt += int(np.all(chain == result))
                    prob = model.log_probability(chain)
                    if prob >= 0.:
                        prob = -np.inf
                    if max_prob is None or prob > max_prob:
                        max_prob, max_chain = prob, chain
                assert result_cnt == 1
                errors += np.sum(np.any(max_chain != result, axis=1))

            metrics["all_words"] += total_word_count
            metrics["all_sent"] += 1
            metrics["wrong_words"] += errors
            metrics["wrong_sent"] += int(errors > 0)
            metrics["unambig_words"] += unambig_word_count
            metrics["unambig_sent"] += int(unambig_word_count > 0)

    for metric_name, val in metrics.iteritems():
        print "{}\t{}".format(metric_name, val)
    print
    print "Unambiguous words accuracy: ", 100. * (1 - 1. * metrics["wrong_words"] / metrics["unambig_words"])
    print "All words accuracy: ", 100. * (1 - 1. * metrics["wrong_words"] / metrics["all_words"])
    print "Unambiguous sentences accuracy: ", 100. * (1 - 1. * metrics["wrong_sent"] / metrics["unambig_sent"])
    print "All sentences accuracy: ", 100. * (1 - 1. * metrics["wrong_sent"] / metrics["all_sent"])


if __name__ == "__main__":
    main()
