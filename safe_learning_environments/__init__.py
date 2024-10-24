from gymnasium.envs.registration import register

register(
     id="safe_learning_environments/TargetHazardWorld-v0",
     entry_point="safe_mpc_environments.envs:TargetHazardWorld",
     # max_episode_steps=300,
)
