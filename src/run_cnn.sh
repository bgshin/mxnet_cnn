#!/bin/bash
echo "train cnn att..."

# "pretrained"
for t in `seq 0 9`
do
    echo "nohup python train_s17_att.py -g 1 -v 0 -t $t > ./log_att/cnn_ex$t.txt"
    nohup nohup python train_s17_att.py -g 1 -v 0 -t $t > ./log_att/cnn_ex$t.txt
done

