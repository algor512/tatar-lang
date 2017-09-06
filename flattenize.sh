#!/bin/bash 
cat ../words.csv | /usr/bin/env python2.7 flattenize.py \
    --special-tags-output special_tags.txt \
    --all-tags-output all_tags.txt\
    --roots-output roots.txt > tmp.tsv
cat ./tmp.tsv | sort -u | sort -k 1 -g > ./flattenized.tsv
rm ./tmp.tsv
