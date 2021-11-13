#!/bin/bash

ENV=LunarLander-v2
#ENV=CartPole-v0
# ENV=Acrobot-v1

echo "REINFORCE"
export CUDA_VISIBLE_DEVICES=1
nohup python train.py\
         --env_name $ENV\
         --algo REINFORCE > out_REINFORCE2.txt &
sleep 1

echo "PPO"
export CUDA_VISIBLE_DEVICES=1
nohup python train.py\
         --env_name $ENV\
         --algo PPO > out_PPO2.txt &
sleep 1

echo "TRPO"
export CUDA_VISIBLE_DEVICES=1
nohup python train.py\
         --env_name $ENV\
         --algo TRPO > out_TRPO2.txt &
sleep 1
