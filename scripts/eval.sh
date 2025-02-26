save_snapshot=False
use_wandb=False
env='robosuite'
model_dir='/home/tienpham/Desktop/DeGuV/eval'
if [ "$env" = "robosuite" ]; then
task_name='Lift'
test_agent='test_deguv'
mode='eval-hard'
action_repeat=2
CUDA_VISIBLE_DEVICES=0  python eval.py \
              env=${env} \
              task=${task_name} \
              model_dir=${model_dir}/${task_name} \
              seed=5 \
              action_repeat=${action_repeat} \
              use_wandb=${use_wandb} \
              use_tb=False \
              save_snapshot=${save_snapshot} \
              save_video=True \
              mode=${mode}\
              wandb_group=${test_agent} \

fi