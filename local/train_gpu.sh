cd ..
taskset -c 5-9 \
python train_pendulum.py \
  --saveas=test \
  --log=train_log \
  --num_train_cycles=100 \
  --buffer_size=10000 \
  --num_trajectories=20 \
  --episode_batch_size=20 \
  --num_learning_iterations=40 \
  --use-gpu \
  --gpu-device 1