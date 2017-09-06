#!/usr/bin/env python2.7 
import sys
import click


@click.command()
@click.option("--special-tags-output", type=click.File("wt"))
@click.option("--all-tags-output", type=click.File("wt"))
@click.option("--roots-output", type=click.File("wt"))
def parse(special_tags_output, all_tags_output, roots_output):
    input_stream = sys.stdin

    spec_tags = set()
    all_tags = set()
    roots = set()

    next(input_stream)
    for line in input_stream:
        parts = line.strip().split("\t")
        if len(parts) != 4:
            print >>sys.stderr, "Error in", line
            assert False
        if "+" not in parts[3]:
            spec_tags.add(parts[3])
        tags = set(parts[3].strip(";").split(";"))
        for t in tags:
            if "+" in t:
                tag_splitted = t.split("+")
                all_tags.update(tag_splitted[1:])
                roots.add(tag_splitted[0])
            print "\t".join(parts[:3] + [t])

    for s in spec_tags:
        if s in all_tags:
            print >>sys.stderr, "Tag ", s, " is used as both special and word tag."
            all_tags.remove(s)
    all_tags = list(all_tags)
    all_tags = all_tags + list(spec_tags)
    if special_tags_output:
        special_tags_output.write("\n".join(spec_tags))
    if all_tags_output:
        all_tags_output.write("\n".join(all_tags))
    if roots_output:
        roots_output.write("\n".join(roots))


if __name__ == "__main__":
    parse()
