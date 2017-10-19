#!/usr/bin/env python2.7
# coding=utf8
from bidict import frozenbidict
from itertools import imap, groupby, tee
from toolz.functoolz import excepts
from tqdm import tqdm
import click
import numpy as np
import ujson as json


class Chain(object):
    def __init__(self, id, fout):
        self.words = []
        self.textid = None
        self.id = id
        self.fout = fout

    def append_word(self, word):
        if self.textid is None:
            self.textid = word["tid"]
        assert self.textid == word["tid"]

        if len(self.words) > 0:
            last = self.words[-1]
            assert word["wid"] - last["id"] == 1

        self.words.append({"id": word["wid"], "v": word["v"], "a": word["a"]})

    def is_next(self, word):
        return len(self.words) == 0 or (word["tid"] == self.textid and word["wid"] - self.words[-1]["id"] == 1)

    def flush(self):
        res = json.dumps({"id": self.id, "text": self.textid, "words": self.words})
        self.fout.write("{}\n".format(res).decode("utf-8"))


@click.command()
@click.argument("all_tags", type=click.File("rt", encoding="utf8"))
@click.argument("vectorized", type=click.File("rt", encoding="utf8"))
@click.argument("ans_vectorized", type=click.File("rt", encoding="utf8"))
@click.argument("sentences", type=click.File("wt", encoding="utf8"))
def main(all_tags, vectorized, ans_vectorized, sentences):
    tags = frozenbidict(enumerate(all_tags.read().splitlines()))

    test_ans, ans_vectorized = tee(ans_vectorized)
    ids_ans = {np.int32(s.split()[0]) for s in tqdm(test_ans, desc="Answers (for test)")}

    stream_dataset = groupby(imap(lambda e: np.fromstring(e, sep="\t", dtype=np.int32),
                                  tqdm(vectorized, desc="Dataset")),
                             lambda v: v[0])
    stream_ans = groupby(imap(lambda e: np.fromstring(e, sep="\t", dtype=np.int32),
                              tqdm(ans_vectorized, desc="Answers")),
                             lambda v: v[0])

    ids_unsolved, ids_unambig = set(), set()
    key_ans, ans = next(stream_ans)

    safe_next = excepts(StopIteration, lambda s: next(s), lambda s: (None, None))

    chain_id = 0
    chain = Chain(chain_id, sentences)
    for key, vars in stream_dataset:
        vars = list(vars)

        text_id = {v[1] for v in vars}
        assert len(text_id) == 1
        text_id = list(text_id)[0]
        assert all(len(v) == len(tags)+3 for v in vars)

        variants = list(set(map(lambda v: tuple(int(e) for e in v[2:]), vars)))
        answers = []
        if len(variants) != 1:
            ids_unambig.add(key)
            if key_ans < key:
                while key_ans is not None and key_ans < key:
                    print "Skip answers for ", key_ans
                    key_ans, ans = safe_next(stream_ans)
            if key_ans != key:
                ids_unsolved.add(key)
                if key in ids_ans:
                    print key
                    assert False
            else:
                ans = list(ans)
                assert all(len(v) == len(tags)+3 for v in ans)
                answers = list(set(map(lambda v: tuple(int(e) for e in v[2:]), ans)))
                key_ans, ans = safe_next(stream_ans)
        current_item = {"wid": int(key), "tid": int(text_id), "v": variants, "a": answers}

        assert len(variants) > 0

        if not chain.is_next(current_item):
            chain.flush()
            chain_id += 1
            chain = Chain(chain_id, sentences)
            chain.append_word(current_item)
        elif sum(variants[0][1:]) == 1 and variants[0][tags.inv["Type1"] + 1] == 1:
            assert len(variants) == 1
            chain.append_word(current_item)
            chain.flush()
            chain_id += 1
            chain = Chain(chain_id, sentences)
        else:
            chain.append_word(current_item)
    chain.flush()

    assert ids_unambig == set(list(ids_ans) + list(ids_unsolved))
    assert ids_unambig - ids_ans - ids_unsolved == set()
    sentences.close()


if __name__ == "__main__":
    main()
