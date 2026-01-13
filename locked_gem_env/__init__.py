from gymnasium.envs.registration import register

register(
    id="locked_gem_env/LockedGem-v0",
    entry_point="locked_gem_env.envs:LockedGemEnv",
)
