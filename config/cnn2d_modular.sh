#! /usr/bin/env bash

export MODULE_ARGS=" \
  --model=CNN2DModular \
  --height=512 \
  --width=128 \
  --tfrecord_height=712 \
  --tfrecord_width=196 \
  --num_epochs=100 \
  --overlap=0.9375 \
"
