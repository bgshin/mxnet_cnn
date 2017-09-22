#!/bin/bash
echo "train cnn att..."

# "pretrained"
for t in `seq 0 9`
do
    echo "nohup python train_s17_att.py -g 2 -v 1 -t $t > ./log_att/att1_ex$t.txt"
    nohup nohup python train_s17_att.py -g 2 -v 1 -t $t > ./log_att/att1_ex$t.txt
done


