cd ..
python train_pendulum.py \
  --saveas=test \
  --log=train_log \
  --num_train_cycles=30 \
  --buffer_size=1000 \
  --num_trajectories=50 \
  --episode_batch_size=32 \
  --num_learning_iterations=40 \
  --render
