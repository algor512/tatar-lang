#!/usr/bin/env python2.7
import sys
import numpy as np
import click

@click.command()
@click.argument("tags", type=click.File("rt", encoding="utf8"))
@click.argument("roots", type=click.File("rt", encoding="utf8"))
@click.option("--use-roots/--skip-roots", default=True)
@click.option("--column", default=-1)
def main(tags, roots, use_roots, column):
    tags = dict(map(reversed, enumerate(tags.read().splitlines())))
    roots = dict(map(reversed, enumerate(roots.read().splitlines())))

    assert set(tags.values()) == set(range(len(tags)))

    for line in sys.stdin:
        parts = line.strip().decode("utf8").split("\t")
        tag_field = parts[column] 
        del parts[column]
        vec = np.zeros(len(tags), dtype=np.uint8)
        if "+" in tag_field:
            tag_list = tag_field.split("+")
            root_id = roots[tag_list[0]]
            for t in tag_list[1:]:
                vec[tags[t]] = 1
            if np.sum(vec) != len(set(tag_list))-1:
                print line 
                assert False
        else:
            root_id = -1
            vec[tags[tag_field]] = 1
        res = parts 
        if use_roots:
            res += [str(root_id)]
        res += map(str, vec.tolist())
        print "\t".join(res)


if __name__ == "__main__":
    main()
