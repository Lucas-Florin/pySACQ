python ../train_fetch.py \
  --saveas=train \
  --log=log \
  --save-freq 10 \
  --num_train_cycles=100 \
  --buffer_size=1024 \
  --num_trajectories=32 \
  --episode_batch_size=32 \
  --num_learning_iterations=40
