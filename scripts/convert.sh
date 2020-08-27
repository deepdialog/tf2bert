#!/bin/bash

set -e

python -m bert.tests.convert_official \
    --input="../bert-embs/pretrained/chinese_L-12_H-768_A-12" \
    --output="../../bert/zh-bert-L12"
    
python -m bert.tests.convert_official \
    --input="../bert-embs/pretrained/chinese_wwm_ext_L-12_H-768_A-12" \
    --output="../../bert/zh-bert-wwm-L12"
    
python -m bert.tests.convert_official \
    --input="../bert-embs/pretrained/chinese_roberta_wwm_ext_L-12_H-768_A-12" \
    --output="../../bert/zh-roberta-wwm-L12"
    
# python -m bert.tests.convert_official \
#     --input="../bert-embs/pretrained/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16" \
#     --output="../../bert/zh-roberta-wwm-large-L24"

# for i in {1,3,6,9,12}; do
#     echo $i
#     python -m bert.tests.convert_official \
#         --input="../bert-embs/pretrained/chinese_L-12_H-768_A-12" \
#         --output="../../bert/zh-bert-L"$i \
#         --num_hidden_layers=$i
# done

# for i in {1,3,6,9,12}; do
#     echo $i
#     python -m bert.tests.convert_official \
#         --input="../bert-embs/pretrained/chinese_wwm_ext_L-12_H-768_A-12" \
#         --output="../../bert/zh-bert-wwm-L"$i \
#         --num_hidden_layers=$i
# done

# for i in {1,3,6,9}; do
#     echo $i
#     python -m bert.tests.convert_official \
#         --input="../bert-embs/pretrained/chinese_roberta_wwm_ext_L-12_H-768_A-12" \
#         --output="../../bert/zh-roberta-wwm-L"$i \
#         --num_hidden_layers=$i
# done

# for i in {1,6,12,18,24}; do
#     echo $i
#     python -m bert.tests.convert_official \
#         --input="../bert-embs/pretrained/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16" \
#         --output="../../bert/zh-roberta-wwm-large-L"$i \
#         --num_hidden_layers=$i
# done
