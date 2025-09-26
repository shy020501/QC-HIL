import glob, tqdm, wandb, os, json, random, time, jax, threading, pickle, flax
from absl import app, flags
from ml_collections import config_flags
from log_utils import setup_wandb, get_exp_name, get_flag_dict, CsvLogger

# from envs.env_utils import make_env_and_datasets
# from envs.ogbench_utils import make_ogbench_env_and_datasets
# from envs.robomimic_utils import is_robomimic_env

from utils.flax_utils import save_agent
from utils.datasets import Dataset, ReplayBuffer
from utils.async_saver import AsyncSaver
import agentlace.inference as ali

from evaluation import evaluate
from agents import agents
import numpy as np

import h5py

PROCESS_KEYS = {
    "left/head_cam": "image_head",
    "left/wrist_cam": "image_wrist_left",
    "right/wrist_cam": "image_wrist_right",
    "state": "state",
}

if 'CUDA_VISIBLE_DEVICES' in os.environ:
    os.environ['EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']
    os.environ['MUJOCO_EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-triple-play-singletask-task2-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')

flags.DEFINE_integer('offline_steps', 1000000, 'Number of offline steps.')
flags.DEFINE_integer('online_steps', 1000000, 'Number of online steps.')
flags.DEFINE_integer('buffer_size', 2000, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 1000000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 1000000, 'Save interval.')
flags.DEFINE_integer('start_training', 5000, 'when does training start')

flags.DEFINE_integer('utd_ratio', 1, "update to data ratio")

flags.DEFINE_float('discount', 0.99, 'discount factor')

flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

config_flags.DEFINE_config_file('agent', 'agents/acfql.py', lock_config=False)

flags.DEFINE_float('dataset_proportion', 1.0, "Proportion of the dataset to use")
flags.DEFINE_integer('dataset_replace_interval', 1000, 'Dataset replace interval, used for large datasets because of memory constraints')
flags.DEFINE_string('ogbench_dataset_dir', None, 'OGBench dataset directory')


flags.DEFINE_string('custom_dataset_path', None, 'Path to a custom HDF5 dataset file.')
flags.DEFINE_string('ckpt_path', None, 'Path to pretrained checkpoint.')

flags.DEFINE_integer('horizon_length', 10, 'action chunking length.')
flags.DEFINE_bool('sparse', False, "make the task sparse reward")

flags.DEFINE_bool('save_all_online_states', False, "save all trajectories to npy")

flags.DEFINE_bool('override_aborts_chunk', True, 'Flush action chunk when intervening.')
flags.DEFINE_integer('min_offline_intervention_len', -1, 'min length to push into demo buffer (default = horizon_length)')
flags.DEFINE_integer('param_pull_interval', 500, 'Actor pulls latest params every N env steps.') # N step 마다 가장 최신 모델 load 해와서 환경에서 작동하도록 #[TODO] 적당히 해당 값 변경

flags.DEFINE_integer('trainer_rpc_port', 45587, 'Remote env server port.')


class LoggingHelper:
    def __init__(self, csv_loggers, wandb_logger):
        self.csv_loggers = csv_loggers
        self.wandb_logger = wandb_logger
        self.first_time = time.time()
        self.last_time = time.time()

    def log(self, data, prefix, step):
        assert prefix in self.csv_loggers, prefix
        self.csv_loggers[prefix].log(data, step=step)
        self.wandb_logger.log({f'{prefix}/{k}': v for k, v in data.items()}, step=step)

def main(_):
    exp_name = get_exp_name(FLAGS.seed)
    run = setup_wandb(project='qc', group=FLAGS.run_group, name=exp_name)

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, FLAGS.env_name, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()

    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    config = FLAGS.agent
    
    # =================================================================================
    # Data Loading Section (CHANGED)
    # =================================================================================
    if FLAGS.custom_dataset_path is not None:
        print(f"Loading custom dataset from: {FLAGS.custom_dataset_path}")
        
        # 1. HDF5 파일에서 데이터 로드
        with h5py.File(FLAGS.custom_dataset_path, 'r') as f:
            train_dataset = {
                'observations': {
                    'image_head': f['observations/image_head'][()],
                    'image_wrist_left': f['observations/image_wrist_left'][()],
                    'image_wrist_right': f['observations/image_wrist_right'][()],
                    'state': f['observations/state'][()],
                },
                'actions': f['actions'][()],
                'rewards': f['rewards'][()],
                'terminals': f['terminals'][()],
                'next_observations': {
                    'image_head': f['next_observations/image_head'][()],
                    'image_wrist_left': f['next_observations/image_wrist_left'][()],
                    'image_wrist_right': f['next_observations/image_wrist_right'][()],
                    'state': f['next_observations/state'][()],
                },
            }
        
        # D4RL 데이터셋 형식에 맞게 'timeouts'와 'masks' 추가
        train_dataset['timeouts'] = train_dataset['terminals']
        train_dataset['masks'] = 1.0 - train_dataset['terminals']

    # house keeping
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    online_rng, rng = jax.random.split(jax.random.PRNGKey(FLAGS.seed), 2)
    log_step = 0
    
    discount = FLAGS.discount
    config["horizon_length"] = FLAGS.horizon_length

    # handle dataset
    def process_train_dataset(ds):
        """
        Process the train dataset to 
            - handle dataset proportion
            - handle sparse reward
            - convert to action chunked dataset
        """

        ds = Dataset.create(**ds)
        if FLAGS.dataset_proportion < 1.0:
            new_size = int(len(ds['masks']) * FLAGS.dataset_proportion)
            ds = Dataset.create(
                **{k: v[:new_size] for k, v in ds.items()}
            )
        
        # if is_robomimic_env(FLAGS.env_name):
        #     penalty_rewards = ds["rewards"] - 1.0
        #     ds_dict = {k: v for k, v in ds.items()}
        #     ds_dict["rewards"] = penalty_rewards
        #     ds = Dataset.create(**ds_dict)
        
        if FLAGS.sparse:
            # Create a new dataset with modified rewards instead of trying to modify the frozen one
            sparse_rewards = (ds["rewards"] != 0.0) * -1.0
            ds_dict = {k: v for k, v in ds.items()}
            ds_dict["rewards"] = sparse_rewards
            ds = Dataset.create(**ds_dict)

        return ds
    
    train_dataset = process_train_dataset(train_dataset)
    example_batch = train_dataset.sample(())
    action_dim = example_batch['actions'].shape[-1]
    
    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    if FLAGS.ckpt_path is not None:
        ckpt_path = FLAGS.ckpt_path
        if os.path.exists(ckpt_path):
            print(f"[INFO] Loading checkpoint from {ckpt_path}")
            with open(ckpt_path, "rb") as f:
                saved_dict = pickle.load(f)
            agent = flax.serialization.from_state_dict(agent, saved_dict["agent"])
        else:
            print(f"[WARNING] Checkpoint path {ckpt_path} does not exist. Skipping load.")

    # Setup logging.
    prefixes = ["eval", "env"]
    if FLAGS.offline_steps > 0:
        prefixes.append("offline_agent")
    if FLAGS.online_steps > 0:
        prefixes.append("online_agent")

    logger = LoggingHelper(
        csv_loggers={prefix: CsvLogger(os.path.join(FLAGS.save_dir, f"{prefix}.csv")) 
                    for prefix in prefixes},
        wandb_logger=wandb,
    )

    rl_buffer = ReplayBuffer.create(example_batch, size=FLAGS.buffer_size)
    demo_buffer = ReplayBuffer.create_from_initial_dataset(
        dict(train_dataset), size=max(FLAGS.buffer_size, train_dataset.size + 1)
    )
        
    rl_buffer_lock = threading.Lock()
    demo_buffer_lock = threading.Lock()

    log_lock = threading.Lock()
    rpc_lock = threading.Lock()
    intervene_lock = threading.Lock()
    action_queue_lock = threading.Lock()

    prev_step_intervened = False  # 직전 스텝에 유저 개입 있었는지
    
    # 파라미터 공유
    params_ref = {'params': agent.network.params}
    version_id = {'val': 0}
    params_lock = threading.Lock()
    stop_event = threading.Event()

    from collections import defaultdict
    data = defaultdict(list)
    online_init_time = time.time()

    # Learner Thread
    def learner_loop():
        nonlocal agent, log_step
        H = FLAGS.horizon_length
        gamma = FLAGS.discount

        while not stop_event.is_set():
            if log_step < FLAGS.start_training:
                time.sleep(0.001)
                continue

            B = config['batch_size'] * FLAGS.utd_ratio
            B_demo = B // 2
            B_rl   = B - B_demo

            with demo_buffer_lock:
                demo_sample = demo_buffer.sample_sequence(B_demo, sequence_length=H, discount=gamma)
            with rl_buffer_lock:
                rl_sample = rl_buffer.sample_sequence(B_rl, sequence_length=H, discount=gamma)

            batch = {k: np.concatenate([demo_sample[k], rl_sample[k]], axis=0) for k in demo_sample}

            agent, update_info = agent.batch_update(batch)

            # 최신 파라미터 업데이트
            with params_lock:
                params_ref['params'] = agent.network.params
                version_id['val'] += 1

            logger.log(update_info, "online_agent", step=log_step)

    learner_thread = threading.Thread(target=learner_loop, name="learner", daemon=True)
    learner_thread.start()

    # 비동기 저장용
    async_saver = AsyncSaver(FLAGS.save_dir, params_lock)

    # Actor server 부분
    server = ali.InferenceServer(port_num=FLAGS.trainer_rpc_port)

    action_queue = []
    local_version = -1
    rpc_call_count = 0

    def _process_obs_keys(ob_dict):
        return {PROCESS_KEYS[k]: v for k, v in ob_dict.items() if k in PROCESS_KEYS}

    def on_reset(payload):
        nonlocal action_queue, prev_step_intervened
        with action_queue_lock:
            action_queue.clear()
        with intervene_lock:
            prev_step_intervened = False
        return {"ok": True}
    
    def act(payload):
        nonlocal online_rng, agent, action_queue, prev_step_intervened

        ob = payload["ob"]
        processed_ob = _process_obs_keys(ob)

        with intervene_lock:
            must_skip_chunk = prev_step_intervened
            if prev_step_intervened:
                prev_step_intervened = False

        with action_queue_lock:
            need_refill = (len(action_queue) == 0)

        # 비어있고 intervene 직후면 dummy 동작 반환
        if need_refill and must_skip_chunk:
            action = np.zeros((action_dim,), dtype=np.float32)
            return {"action": action, "dummy": True}

        refill_chunk = None
        if need_refill:
            online_rng, key = jax.random.split(online_rng)
            a_chunk = agent.sample_actions(observations=processed_ob, rng=key)
            refill_chunk = np.array(a_chunk).reshape(-1, action_dim).tolist()

        # 다시 락을 잡고, queue가 비어있으면 refill한 뒤 pop
        with action_queue_lock:
            if len(action_queue) == 0 and refill_chunk is not None:
                action_queue.extend(refill_chunk)
            if len(action_queue) == 0:
                action = np.zeros((action_dim,), dtype=np.float32)
                return {"action": action, "dummy": True}
            action = np.asarray(action_queue.pop(0), dtype=np.float32)

        return {"action": action, "dummy": False}

    # Transition을 replay buffer에 추가
    def push_transition(payload):
        nonlocal logger, prev_step_intervened, log_step, rpc_call_count, local_version
        ob         = payload["ob"]
        next_ob    = payload["next_ob"]
        reward     = float(payload["reward"])
        terminated = bool(payload["terminated"])
        truncated  = bool(payload["truncated"])
        info       = payload.get("info", {})
        action     = np.asarray(payload["action"], dtype=np.float32)
        state      = payload.get("state")

        done = terminated or truncated
        processed_ob = _process_obs_keys(ob)
        processed_next_ob = _process_obs_keys(next_ob)

        with rpc_lock:
            rpc_call_count += 1

        with log_lock:
            log_step += 1

        # 파라미터 저장
        if FLAGS.save_interval > 0 and (rpc_call_count % FLAGS.save_interval == 0):
            async_saver.request(agent, log_step, params_ref=params_ref)   

        # 파라미터 pull
        if FLAGS.param_pull_interval > 0 and (rpc_call_count % FLAGS.param_pull_interval == 0) and (local_version != version_id['val']):
            with params_lock:
                agent = agent.replace(network=agent.network.replace(params=params_ref['params']))
                local_version = version_id['val']
            with action_queue_lock:
                action_queue.clear()

        if "intervene_action" in info: 
            used_action = np.asarray(info.pop("intervene_action"), dtype=np.float32)
            intervened = True 
        else: 
            used_action = action.astype(np.float32, copy=False)
            intervened = False 
            
        if intervened and FLAGS.override_aborts_chunk:    
            with action_queue_lock:
                action_queue.clear()

        with intervene_lock:
            prev_step_intervened = intervened

        transition = dict(
            observations=processed_ob,
            actions=used_action,
            rewards=reward,
            terminals=float(done),
            masks=1.0 - float(terminated),
            next_observations=processed_next_ob,
            timeouts=float(done),
        )
        with rl_buffer_lock:
            rl_buffer.add_transition(transition)

        env_info = {k: v for k, v in info.items() if str(k).startswith("distance")}
        if env_info:
            logger.log(env_info, "env", step=log_step)

        if FLAGS.save_all_online_states and state is not None:
            i = payload.get("step_idx", None)  # 있으면 같이 저장
            if i is not None: data["steps"].append(int(i))
            if "qpos" in state: data["qpos"].append(np.asarray(state["qpos"]))
            if "qvel" in state: data["qvel"].append(np.asarray(state["qvel"]))
            if "button_states" in state: data["button_states"].append(np.asarray(state["button_states"]))
            data["obs"].append(processed_next_ob["state"] if "state" in processed_next_ob else np.nan)

        return {"ok": True}

    server.register_interface("on_reset", on_reset)
    server.register_interface("act", act)
    server.register_interface("push_transition", push_transition)

    print(f"[Trainer RPC] listening on :{FLAGS.trainer_rpc_port}")
    server.start()

    try:
        server.wait()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        learner_thread.join(timeout=2.0)

    for key, csv_logger in logger.csv_loggers.items():
        csv_logger.close()

    end_time = time.time()

    if FLAGS.save_all_online_states and len(data["obs"]) > 0:
        c_data = {"online_time": end_time - online_init_time}
        if len(data["steps"]) > 0: c_data["steps"] = np.array(data["steps"])
        if len(data["qpos"])  > 0: c_data["qpos"]  = np.stack(data["qpos"], axis=0)
        if len(data["qvel"])  > 0: c_data["qvel"]  = np.stack(data["qvel"], axis=0)
        if len(data["obs"])   > 0 and isinstance(data["obs"][0], np.ndarray):
            c_data["obs"] = np.stack(data["obs"], axis=0)
        if len(data["button_states"]) > 0:
            c_data["button_states"] = np.stack(data["button_states"], axis=0)
        np.savez(os.path.join(FLAGS.save_dir, "data.npz"), **c_data)

    with open(os.path.join(FLAGS.save_dir, 'token.tk'), 'w') as f:
        f.write(run.url)

    async_saver.flush_and_stop(agent, log_step, params_ref=params_ref, timeout=3.0)

if __name__ == '__main__':
    app.run(main)