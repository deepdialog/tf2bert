#!/bin/bash

# https://github.com/google-research/albert
# https://storage.googleapis.com/albert_models/albert_base_zh.tar.gz
# https://storage.googleapis.com/albert_models/albert_large_zh.tar.gz
# https://storage.googleapis.com/albert_models/albert_xlarge_zh.tar.gz
# https://storage.googleapis.com/albert_models/albert_xxlarge_zh.tar.gz

set -e

python -m bert.tests.convert_albert \
        --input="../bert-embs/pretrained/albert_base/" \
        --output="../../bert/albert_base/"

python -m bert.tests.convert_albert \
        --input="../bert-embs/pretrained/albert_large/" \
        --output="../../bert/albert_large/"

python -m bert.tests.convert_albert \
        --input="../bert-embs/pretrained/albert_xlarge/" \
        --output="../../bert/albert_xlarge/"

python -m bert.tests.convert_albert \
        --input="../bert-embs/pretrained/albert_xxlarge/" \
        --output="../../bert/albert_xxlarge/"
