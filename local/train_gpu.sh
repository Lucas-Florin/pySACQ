taskset -c 35-39 \
python ../train_pendulum.py \
  --saveas=train \
  --log=train_log \
  --num_train_cycles=100 \
  --buffer_size=10000 \
  --num_trajectories=20 \
  --episode_batch_size=32 \
  --num_learning_iterations=40 \
  --use-gpu \
  --gpu-device 7