#!/bin/bash


echo "cnn"
for t in `seq 0 9`
do
    tail -n 1 ./log_att/cnn_ex$t.txt
done

echo "att1"
for t in `seq 0 9`
do
    tail -n 1 ./log_att/att1_ex$t.txt
done

echo "att2"
for t in `seq 0 9`
do
    tail -n 1 ./log_att/att2_ex$t.txt
done
