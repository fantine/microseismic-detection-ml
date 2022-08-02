#! /usr/bin/env bash

export MODULE_ARGS=" \
  --model=CNN2DModular \
  --height=512 \
  --width=128 \
  --channels=1 \
  --tfrecord_height=712 \
  --tfrecord_width=196 \
  --num_epochs=100 \
  --learning_rate=0.001 \
  --batch_size=32 \
  --network_depth=5 \
  --num_filters=16 \
  --filter_increase_mode=2 \
  --filter_multiplier=8 \
  --activation=0 \
  --downsampling=1 \
  --batchnorm=1 \
  --conv_dropout=0.2 \
  --dense_dropout=0.4 \
  --regularizer=0 \
"
