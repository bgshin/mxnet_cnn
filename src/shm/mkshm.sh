#!/bin/bash
export PYTHONPATH='../../'
nohup /home/bgshin/.virtualenvs/mxnet/bin/python -u /home/bgshin/works/mxnet_cnn/src/shm/w2v_shm_loader.py > mkshm.txt &
while [ ! -f /dev/shm/s17_y_tst_400 ]
do
    sleep 2
done
ls /dev/shm/
echo 'done'
