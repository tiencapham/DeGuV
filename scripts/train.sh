
task='Lift'
frames=1001000
feature_dim=50
action_repeat=2
env=robosuite

CUDA_VISIBLE_DEVICES=0  python train.py \
							env=${env} \
							task=${task} \
							seed=5 \
							action_repeat=${action_repeat} \
							use_wandb=False \
							use_tb=False \
							num_train_frames=${frames} \
							save_snapshot=True \
							save_video=True \
							feature_dim=${feature_dim} \
							
