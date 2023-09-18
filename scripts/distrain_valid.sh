#!/bin/bash
THIS_DIR="$( cd "$( dirname "$0" )" && pwd )"
cd $THIS_DIR
cd ..

NGPUS=2
CFGFILEPATH=./configs/mscc/mscc_resnet_ff_all.py
PORT=${PORT:-8667}
NNODES=${NNODES:-1}
NODERANK=${NODERANK:-0}
MASTERADDR=${MASTERADDR:-"127.0.0.1"}

CUDA_VISIBLE_DEVICES=6,7 \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODERANK \
    --master_addr=$MASTERADDR \
    --nproc_per_node=$NGPUS \
    --master_port=$PORT \
    main/train_segmentor.py --nproc_per_node $NGPUS \
                   --cfgfilepath $CFGFILEPATH ${@:3} 
                   

CUDA_VISIBLE_DEVICES=6,7 \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODERANK \
    --master_addr=$MASTERADDR \
    --nproc_per_node=$NGPUS \
    --master_port=$PORT \
    main/test_segmentor.py --nproc_per_node $NGPUS \
                   --cfgfilepath $CFGFILEPATH ${@:3}