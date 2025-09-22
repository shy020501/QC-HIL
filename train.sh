export XLA_FLAGS=--xla_gpu_autotune_level=0
export JAX_DEFAULT_matmul_precision='float32'
export XLA_PYTHON_CLIENT_PREALLOCATE=false 

MUJOCO_GL=egl python main.py \
  --run_group=MY_ROBOT \
  --env_name=AGI_TF \
  --sparse=True \
  --horizon_length=10 \
  --eval_interval 0 \
  --custom_dataset_path ./final_robotics_dataset.h5 \
  --save_interval 10000 \
  --online_steps 100000 \
  --offline_steps 100000 \
