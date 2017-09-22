#!/bin/bash
echo "train cnn att..."

# "pretrained"
for t in `seq 0 9`
do
    echo "nohup python train_s17_att.py -g 3 -v 2 -t $t > ./log_att/att2_ex$t.txt"
    nohup nohup python train_s17_att.py -g 3 -v 2 -t $t > ./log_att/att2_ex$t.txt
done


