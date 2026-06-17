import random

import numpy as np


def make_async(
    id,
    num_envs=1,
    asynchronous=True,
    wrappers=None,
    render=False,
    obs_dim=23,
    action_dim=7,
    env_type=None,
    max_episode_steps=None,
    env_kwargs=None,
    **kwargs,
):
    if int(num_envs) != 1:
        raise ValueError("PC-HMD supports only num_envs=1")
    import gym
    env = gym.make(id, **(env_kwargs or {}))
    return _SingleEnvAdapter(env)


class _SingleEnvAdapter:
    def __init__(self, env):
        self.env = env
        self.num_envs = 1
        self.is_vector_env = True
        self.observation_space = getattr(env, "observation_space", None)
        self.action_space = getattr(env, "action_space", None)
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space
        self.metadata = getattr(env, "metadata", {})

    def reset_arg(self, options_list=None, **kwargs):
        obs = self.reset()
        return np.expand_dims(obs, 0) if isinstance(obs, np.ndarray) else [obs]

    def reset_one_arg(self, env_ind, options=None):
        assert env_ind == 0
        return self.reset()

    def reset(self, **kwargs):
        return self.env.reset()

    def step(self, actions):
        if isinstance(actions, (list, tuple)) or getattr(actions, "ndim", 0) > 0:
            action = actions[0]
        else:
            action = actions
        obs, reward, done, info = self.env.step(action)
        obs_b = np.expand_dims(obs, 0) if isinstance(obs, np.ndarray) else [obs]
        rew_b = np.asarray([reward], dtype=np.float32)
        done_b = np.asarray([done], dtype=bool)
        info_b = [info]
        return obs_b, rew_b, done_b, info_b

    def seed(self, seeds=None):
        if isinstance(seeds, int) or seeds is None:
            seed = seeds if isinstance(seeds, int) else None
        elif isinstance(seeds, (list, tuple)) and len(seeds) > 0:
            seed = seeds[0]
        else:
            seed = None
        try:
            self.env.reset(seed=seed)
            return [seed]
        except TypeError:
            pass
        except Exception:
            pass
        if hasattr(self.env, "seed") and callable(getattr(self.env, "seed")):
            try:
                return self.env.seed(seed)
            except Exception:
                pass
        random.seed(seed)
        np.random.seed(seed if seed is not None else 0)
        return [seed]

    def render(self, *args, **kwargs):
        if hasattr(self.env, "render"):
            return self.env.render(*args, **kwargs)

    def close(self, **kwargs):
        if hasattr(self.env, "close"):
            self.env.close()
