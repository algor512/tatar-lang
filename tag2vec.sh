#!/bin/bash 

cat flattenized.tsv \
    | awk -F'\t' 'BEGIN { OFS="\t" } ; { print $1,$2,$4 }' \
    | python2.7 tag2vec.py all_tags.txt roots.txt > vectorized.tsv.tmp
cat ../disamed.csv \
    | python2.7 tag2vec.py all_tags.txt roots.txt --column 1 >> ans_vectorized.tsv.tmp
cat vectorized.tsv.tmp | sort -k1 -g > vectorized.tsv 
cat ans_vectorized.tsv.tmp | sort -k1 -g > ans_vectorized.tsv 
rm vectorized.tsv.tmp ans_vectorized.tsv.tmp
