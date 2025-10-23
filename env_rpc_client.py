import argparse
import numpy as np
import agentlace.inference as ali
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from experiments.config import load_and_apply_config


def npify_obs(obs):
    if isinstance(obs, dict):
        return {k: np.asarray(v) for k, v in obs.items()}
    return np.asarray(obs)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trainer_ip", type=str, default="115.145.172.97")
    p.add_argument("--trainer_port", type=int, default=45587) 
    p.add_argument("--env_name", type=str, default="cleanup_table")
    p.add_argument("--arm_type", type=str, default="right")
    p.add_argument("--fake_env", action="store_true", default=False)
    p.add_argument("--save_video", action="store_true", default=False)
    p.add_argument("--classifier", action="store_true", default=False)
    p.add_argument("--episodes", type=int, default=0, help="0=loop forever")
    args = p.parse_args()

    train_config_instance = load_and_apply_config(
        env_name=args.env_name,
        arm_type=args.arm_type,
        overrides=None,
    )
    
    env = train_config_instance.get_environment(fake_env=args.fake_env)
    env = RecordEpisodeStatistics(env)

    client = ali.InferenceClient(server_ip=args.trainer_ip, port_num=args.trainer_port)
    print(f"[Env Client] connected to trainer {args.trainer_ip}:{args.trainer_port}")

    ep = 0
    while True:
        ob, info = env.reset()
        client.call("on_reset", {"ob": npify_obs(ob), "info": info})
        done = False

        while not done:
            resp = client.call("act", {"ob": npify_obs(ob)})
            action = np.asarray(resp["action"], dtype=np.float32)

            next_ob, reward, terminated, truncated, step_info = env.step(action)
            done = bool(terminated or truncated)

            payload = {
                "ob": npify_obs(ob),
                "action": action,
                "reward": float(reward),
                "next_ob": npify_obs(next_ob),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "info": step_info,
            }
            try:
                payload["state"] = env.get_state()
            except Exception:
                pass

            # 받은게 dummy면서, 이번 step에 intervention이 없었으면 해당 transaction에 intervention이 없었으면, transition에 넣으면 안됨
            if not resp['dummy'] or 'intervene_action' in step_info:
                client.call("push_transition", payload)
            ob = next_ob

        ep += 1
        if args.episodes > 0 and ep >= args.episodes:
            break

    env.close()
    print("[Env Client] finished.")

if __name__ == "__main__":
    main()
