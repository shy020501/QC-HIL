export XLA_FLAGS=--xla_gpu_autotune_level=0
export JAX_DEFAULT_matmul_precision='float32'
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export ROBOSUITE_LOG_PATH=$HOME/robosuite_logs/robosuite.log 

MUJOCO_GL=egl python main_hil.py \
  --run_group=MY_ROBOT \
  --env_name=AGI_TF \
  --sparse=True \
  --horizon_length=10 \
  --eval_interval 1000000 \
  --custom_dataset_path ./final_robotics_dataset.h5 \
  --ckpt_path ./ckpt/params_99000.pkl \
  --save_interval 10000 \
  --online_steps 1000000 \