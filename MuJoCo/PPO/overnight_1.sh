#!/bin/bash
python train.py HalfCheetah -n 3000 -b 5 -ti 12
python train.py HalfCheetah -n 3000 -b 5 -ti 15
python train.py HalfCheetah -n 3000 -b 5 -ti 17
python train.py HalfCheetah -n 3000 -b 5 -ti 18
python train.py HalfCheetah -n 3000 -b 5 -ti 19
python test_expert_models.py HalfCheetah