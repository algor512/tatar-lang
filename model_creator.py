#!/usr/bin/env python2.7
import os
import click
import logging
import pandas as pd
import numpy as np
import pomegranate as pg


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s : %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger()


def generate_random_dist(shape, roots=None):
    dists = []
    if roots is not None:
        probs = np.random.rand(len(roots))
        probs = probs / np.sum(probs)
        dists.append(pg.DiscreteDistribution(dict(zip(roots, probs))))
        shape -= 1
    for i in range(shape):
        probs = np.random.rand(2)
        probs = probs / np.sum(probs)
        dists.append(pg.DiscreteDistribution(dict(zip([0, 1], probs))))
    return pg.IndependentComponentsDistribution(dists)


@click.command()
@click.argument("tags", type=click.File("rt", encoding="utf8"))
@click.argument("roots", type=click.File("rt", encoding="utf8"))
@click.argument("train", type=click.Path(exists=True))
@click.argument("output", type=click.File("wt", encoding="utf8"))
@click.option("--texts", nargs=2, default=[1, 10], type=int)
@click.option("--states", default=10, type=int)
@click.option("--no-roots", is_flag=True)
def main(tags, roots, train, output, texts, states, no_roots):
    logger.info("No roots: %s", no_roots)

    texts = set(range(texts[0], texts[1] + 1))
    columns = ["RootId"] + tags.read().splitlines() + ["ChainId"]
    roots = range(len(roots.read().splitlines())) + [-1]

    print len(roots), max(roots)
    sys.exit(0)

    logger.info("Texts: %s", texts)
    logger.info("States: %d", states)

    logger.info("Load data...")
    chains = []
    for f in os.listdir(train):
        textid = int(f.split(".")[0].split("_")[1])
        if textid in texts:
            data = pd.read_csv(os.path.join(train, f), sep="\t", names=columns)
            for chain_id, chain in data.groupby("ChainId"):
                if chain.shape[0] > 5:
                    chains.append(chain.drop(["ChainId", "RootId"] if no_roots else "ChainId", axis=1))
    concated_chains = pd.concat(chains).as_matrix()

    logger.info("Initialise distributions...")
    dists = [generate_random_dist(concated_chains.shape[1], None if no_roots else roots) for _ in range(states)]
    trans_mat = np.ones((states, states)) / (1. * states)
    starts = np.ones(states) / (1. * states)
    ends = np.ones(states) / (1. * states)

    logger.info("Model fitting...")
    model = pg.HiddenMarkovModel.from_matrix(trans_mat, dists, starts, ends)
    model.fit([ch.as_matrix() for ch in chains], verbose=True, n_jobs=8,
              max_iterations=5000, stop_threshold=0.0001, pseudocount=0.1)

    logger.info("Write output...")
    output.write(unicode(model.to_json()))
    output.close()


if __name__ == "__main__":
    main()
