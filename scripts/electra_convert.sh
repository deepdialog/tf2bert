#!/bin/bash

# https://github.com/ymcui/Chinese-ELECTRA

set -e

python3 -m bert.tests.convert_electra \
    --input "../pretrained/chinese_electra_base_L-12_H-768_A-12" \
    --output "../../bert/electra_base"

python3 -m bert.tests.convert_electra \
    --input "../pretrained/chinese_electra_small_L-12_H-256_A-4" \
    --output "../../bert/electra_small"

python3 -m bert.tests.convert_electra \
    --input "../pretrained/chinese_electra_small_ex_L-24_H-256_A-4" \
    --output "../../bert/electra_smallex"

python3 -m bert.tests.convert_electra \
    --input "../pretrained/chinese_electra_large_L-24_H-1024_A-16" \
    --output "../../bert/electra_large"
