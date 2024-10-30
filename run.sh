#!/bin/bash

for i in {1..10}
do
   #echo $i
   python3 train.py --warp False --seed $i
   python3 train.py --seed $i

done

