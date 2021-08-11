#!/bin/bash

python test_NN_restarts.py \
    --rank 32 \
    --n_samp 50 \
    --tau 10.5 \
    --transient 50 \
    --restart_dataset test

python test_NN_restarts.py \
    --rank 8 \
    --n_samp 50 \
    --tau 10.5 \
    --transient 50 \
    --restart_dataset test

python test_NN_restarts.py \
    --rank 16 \
    --n_samp 50 \
    --tau 10.5 \
    --transient 50 \
    --restart_dataset test

python test_NN_restarts.py \
    --rank 64 \
    --n_samp 50 \
    --tau 10.5 \
    --transient 50 \
    --restart_dataset test
