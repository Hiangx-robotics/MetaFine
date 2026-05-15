#!/usr/bin/env bash
set -euo pipefail

cd /export/xuhy/EAI/git

python utils/convert_to_rlds.py \
  -i /nat/demos/PegInsertionSide-v1/ \
  -o /nat/demos/openvla/data/peg_insertion_side \
  --dataset-name peg_insertion_side_rlds_100 \
  --image-size 256 \
  --max-episodes 100

python utils/convert_to_rlds.py \
  -i /nat/demos/press_part/press_switch_7/ \
  -o /nat/demos/openvla/data/press_switch_7 \
  --dataset-name press_switch_7_rlds_100 \
  --image-size 256 \
  --max-episodes 100

python utils/convert_to_rlds.py \
  -i /nat/demos/toggle_part/toggle_switch_2/ \
  -o /nat/demos/openvla/data/toggle_switch_2 \
  --dataset-name toggle_switch_2_rlds_100 \
  --image-size 256 \
  --max-episodes 100

python utils/convert_to_rlds.py \
  -i /nat/demos/rotate_along/rotate_along_2/ \
  -o /nat/demos/openvla/data/rotate_along_2 \
  --dataset-name rotate_along_2_rlds_100 \
  --image-size 256 \
  --max-episodes 100

python utils/convert_to_rlds.py \
  -i /nat/demos/PlugCharger-v1/ \
  -o /nat/demos/openvla/data/plug_charger \
  --dataset-name plug_charger_rlds_100 \
  --image-size 256 \
  --max-episodes 100

python utils/convert_to_rlds.py \
  -i /nat/demos/StackPyramid-v1/ \
  -o /nat/demos/openvla/data/stack_pyramid \
  --dataset-name stack_pyramid_rlds_100 \
  --image-size 256 \
  --max-episodes 100
