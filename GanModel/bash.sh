#!/usr/bin/env bash

python main.py --im_path "15.png" --target_mask "15.png" --sign realistic --smooth 5 --learning_rate 0.01 --W_steps 1000 --FS_steps 200 \
--align_steps1 50 --nose_shape "1" --nose_align_steps 50 --blend_steps 200
