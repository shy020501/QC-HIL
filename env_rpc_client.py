import argparse
import numpy as np
import agentlace.inference as ali
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from experiments.mappings import CONFIG_MAPPING

def npify_obs(obs):
    if isinstance(obs, dict):
        return {k: np.asarray(v) for k, v in obs.items()}
    return np.asarray(obs)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trainer_ip", type=str, default="115.145.172.97")
    p.add_argument("--trainer_port", type=int, default=45587) 
    p.add_argument("--env_name", type=str, default="cleanup_table")
    p.add_argument("--fake_env", action="store_true", default=False)
    p.add_argument("--save_video", action="store_true", default=False)
    p.add_argument("--classifier", action="store_true", default=False)
    p.add_argument("--episodes", type=int, default=0, help="0=loop forever")
    args = p.parse_args()

    exp_config = CONFIG_MAPPING[args.env_name]()
    env = exp_config.get_environment(
        fake_env=args.fake_env,
        save_video=args.save_video,
        classifier=args.classifier,
    )
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

            client.call("push_transition", payload)
            ob = next_ob

        ep += 1
        if args.episodes > 0 and ep >= args.episodes:
            break

    env.close()
    print("[Env Client] finished.")

if __name__ == "__main__":
    main()
