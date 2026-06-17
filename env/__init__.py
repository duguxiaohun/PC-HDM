import gym

gym.register(
    id="CarlaTown10Cross-v0",
    entry_point="env.carla_env_town10:InterSection",
)

gym.register(
    id="CarlaTown05Cross-v0",
    entry_point="env.carla_env_town05:InterSection",
)

gym.register(
    id="CarlaTown03Cross-v0",
    entry_point="env.carla_env_town03:InterSection",
)
