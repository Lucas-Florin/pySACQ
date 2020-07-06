cd ..
#taskset -c 5-9 \
python train_pendulum.py \
  --saveas=test \
  --log=train_log \
  --num_train_cycles=100 \
  --buffer_size=1000 \
  --num_trajectories=50 \
  --episode_batch_size=32 \
  --num_learning_iterations=40
  --use-gpu \
  --gpu-device 1