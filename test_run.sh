#!/bin/bash

for i in {1..5}
do
   echo "Test Run Iteration: $i"
   R=$RANDOM
   echo "Random Seed: $R"
   python3 train.py --warp False --seed $R --verbose True
   python3 train.py --seed $R --verbose True

done

