taskset -c 35-39 \
python ../train_pendulum.py \
  --saveas=train \
  --log=log \
  --save-freq 10 \
  --num_train_cycles=100 \
  --buffer_size=512 \
  --num_trajectories=32 \
  --episode_batch_size=32 \
  --num_learning_iterations=40 \
  --use-gpu \
  --gpu-device 7