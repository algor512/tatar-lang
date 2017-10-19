#!/usr/bin/env python2.7
import click
import logging
import ujson as json
import numpy as np
import pomegranate as pg

from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s : %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger()


def generate_random_dist(shape, roots):
    dists = []
    if roots is not None:
        probs = np.random.rand(len(roots))
        probs = probs / np.sum(probs)
        dists.append(pg.DiscreteDistribution(dict(zip(roots, probs))))
    for i in range(shape-1):
        probs = np.random.rand(2)
        probs = probs / np.sum(probs)
        dists.append(pg.DiscreteDistribution(dict(zip([0, 1], probs))))
    return pg.IndependentComponentsDistribution(dists)


@click.command()
@click.argument("sentences", type=click.File("rt", encoding="utf8"))
@click.argument("output", type=click.File("wt", encoding="utf8"))
@click.argument("roots_file", type=click.File("rt", encoding="utf8"))
@click.option("--texts", nargs=2, default=[1, 10], type=int)
@click.option("--states", default=10, type=int)
@click.option("--init", default=None, type=click.File("rt", encoding="utf8"))
@click.option("--method", type=click.Choice(["viterbi", "baum-welch"]), default="viterbi")
def main(sentences, output, roots_file, texts, states, init, method):
    texts = set(range(texts[0], texts[1] + 1))

    logger.info("Texts: %s", texts)
    logger.info("States: %d", states)

    logger.info("Load data...")

    roots = range(len(roots_file.read().splitlines())) + [-1]
    roots_file.close()

    unsolved_words = 0
    train_dataset = []
    for l in tqdm(sentences):
        sentence = json.loads(l)

        if sentence["text"] not in texts:
            continue

        ans = []
        for w in sentence["words"]:
            if len(w["v"]) == 1:
                ans.append(w["v"][0])
            elif len(w["a"]) == 1:
                ans.append(w["a"][0])
            else:
                ans.append(None)
                unsolved_words += 1
        if None not in ans:
            train_dataset.append(np.array(ans))
    logging.info("There are %d unsolved words", unsolved_words)
    logging.info("%d sentences have been loaded", len(train_dataset))

    logger.info("Initialise distributions...")
    if init is None:
        logger.info("Use random initialization.")
        dists = [generate_random_dist(train_dataset[0].shape[1], roots) for _ in range(states)]
        trans_mat = np.ones((states, states)) / (1. * states)
        starts = np.ones(states) / (1. * states)
        ends = np.ones(states) / (1. * states)

        model = pg.HiddenMarkovModel.from_matrix(trans_mat, dists, starts, ends)
    else:
        logger.info("Load model from file %s", init.name)
        model = pg.HiddenMarkovModel.from_json(init.read())

    logger.info("Model fitting ({}):".format(method))
    model.fit(train_dataset, verbose=True, n_jobs=8, pseudocount=0.1, algorithm=method)

    logger.info("Write outputs")
    output.write(unicode(model.to_json()))
    output.close()


if __name__ == "__main__":
    main()
