import pickle
import numpy as np
import h5py
import os
import time
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# --- 설정 (사용자 수정 필요) ---

# 1. 원본 대용량 데이터셋 파일 경로
# 여러 파일인 경우 리스트로 작성: ['file1.pkl', 'file2.pkl']
# 단일 파일인 경우: 'YOUR_LARGE_FILE.pkl'
input_file_path = '../data/cleanup_table_two_waste/2025-08-19/cleanup_table_3_demos_2025-08-19_15-51-19.pkl'
input_file_path = './robo.hdf5'

# 2. 전처리 후 저장될 파일 경로
output_file_path = 'final_robotics_dataset.h5'

# --- 스크립트 ---

def split_into_episodes(transitions):
    """'infos'의 'succeed' 키를 기준으로 전체 데이터를 에피소드 리스트로 분할합니다."""
    print("Splitting data into episodes...")
    episodes = []
    current_episode = []
    for transition in tqdm(transitions, desc="Splitting Episodes"):
        current_episode.append(transition)
        # 에피소드 종료 조건 (필요에 따라 'dones' 키 등으로 변경 가능)
        if transition.get("infos", {}).get("succeed", False):
            episodes.append(current_episode)
            current_episode = []
    
    # 마지막 에피소드가 종료 조건으로 끝나지 않은 경우에도 추가
    if current_episode:
        episodes.append(current_episode)
        
    print(f"Found {len(episodes)} episodes.")
    return episodes

def process_episode(episode):
    """
    하나의 에피소드를 처리하는 작업자(worker) 함수.
    3개의 이미지와 상태 벡터를 모두 추출합니다.
    """
    # 모든 관측 데이터를 저장할 리스트들을 초기화
    obs_head, obs_wrist_l, obs_wrist_r, obs_states = [], [], [], []
    next_obs_head, next_obs_wrist_l, next_obs_wrist_r, next_obs_states = [], [], [], []
    actions, rewards, dones = [], [], []

    for step_data in episode:
        # observations 딕셔너리에서 각 데이터를 추출
        obs = step_data['observations']
        obs_head.append(np.squeeze(obs['left/head_cam'], axis=0))
        obs_wrist_l.append(np.squeeze(obs['left/wrist_cam'], axis=0))
        obs_wrist_r.append(np.squeeze(obs['right/wrist_cam'], axis=0))
        obs_states.append(obs['state'].flatten())
        
        # next_observations 딕셔너리에서 각 데이터를 추출
        next_obs = step_data['next_observations']
        next_obs_head.append(np.squeeze(next_obs['left/head_cam'], axis=0))
        next_obs_wrist_l.append(np.squeeze(next_obs['left/wrist_cam'], axis=0))
        next_obs_wrist_r.append(np.squeeze(next_obs['right/wrist_cam'], axis=0))
        next_obs_states.append(next_obs['state'].flatten())
        
        actions.append(step_data['actions'].flatten())
        rewards.append(step_data['rewards'])
        dones.append(step_data['dones'])

    # HDF5 그룹에 저장할 수 있도록 키 이름을 '/'로 구분하여 딕셔너리를 반환
    return {
        'observations/image_head': np.array(obs_head, dtype=np.uint8),
        'observations/image_wrist_left': np.array(obs_wrist_l, dtype=np.uint8),
        'observations/image_wrist_right': np.array(obs_wrist_r, dtype=np.uint8),
        'observations/state': np.array(obs_states, dtype=np.float32),
        'actions': np.array(actions, dtype=np.float32),
        'rewards': np.array(rewards, dtype=np.float32).reshape(-1, 1),
        'next_observations/image_head': np.array(next_obs_head, dtype=np.uint8),
        'next_observations/image_wrist_left': np.array(next_obs_wrist_l, dtype=np.uint8),
        'next_observations/image_wrist_right': np.array(next_obs_wrist_r, dtype=np.uint8),
        'next_observations/state': np.array(next_obs_states, dtype=np.float32),
        'terminals': np.array(dones, dtype=np.bool_).reshape(-1, 1),
    }

def main():
    start_time = time.time()

    # 단일 파일과 여러 파일 목록 모두 처리 가능하도록 수정
    if isinstance(input_file_path, str):
        paths = [input_file_path]
    else:
        paths = input_file_path

    all_transitions = []
    print(f"Loading {len(paths)} dataset file(s)...")
    for path in paths:
        with open(path, 'rb') as f:
            all_transitions.extend(pickle.load(f))
    print("All dataset files loaded.")

    episodes = split_into_episodes(all_transitions)
    
    # CPU 코어 수만큼 프로세스 생성
    num_processes = cpu_count()
    print(f"Starting parallel processing with {num_processes} cores...")
    
    with Pool(processes=num_processes) as pool:
        # pool.imap을 사용하여 메모리를 더 효율적으로 처리하고 진행 상황을 표시
        processed_episodes = list(tqdm(pool.imap(process_episode, episodes), total=len(episodes), desc="Processing Episodes"))
    
    print("Parallel processing finished. Concatenating results...")
    
    # 모든 에피소드의 결과를 하나의 큰 NumPy 배열로 합치기
    final_dataset = {}
    if not processed_episodes:
        print("No episodes were processed. Exiting.")
        return
        
    keys = processed_episodes[0].keys()
    for key in tqdm(keys, desc="Concatenating Data"):
        arrays_to_concat = [ep[key] for ep in processed_episodes]
        final_dataset[key] = np.concatenate(arrays_to_concat, axis=0)

    print("\n--- Final Combined Data Shapes ---")
    total_timesteps = len(next(iter(final_dataset.values())))
    print(f"Total timesteps: {total_timesteps}")
    for key, data in final_dataset.items():
        print(f"'{key}': {data.shape}")
    print("----------------------------------\n")

    print(f"Saving final dataset to: {output_file_path}")
    with h5py.File(output_file_path, 'w') as f:
        for key, data in final_dataset.items():
            f.create_dataset(key, data=data, compression='gzip')

    end_time = time.time()
    print(f"Preprocessing complete! Total time taken: {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    main()