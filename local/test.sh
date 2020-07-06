cd ..
# taskset -c 35-39 \
python train_pendulum.py \
  --saveas=test \
  --log=test_log \
  --num_train_cycles=2 \
  --buffer_size=100 \
  --num_trajectories=5 \
  --num_learning_iterations=20 \
  --episode_batch_size=1 \
  --render
  --use-gpu \
  --gpu-device 7