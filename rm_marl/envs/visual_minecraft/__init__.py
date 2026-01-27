from gymnasium.envs.registration import register

register(
    id="VisualMinecraft-v0",
    entry_point="rm_marl.envs.visual_minecraft.env:GridWorldEnv",
)


register(
    id="DebugVisualMinecraft-v0",
    entry_point="rm_marl.envs.visual_minecraft.debug_env:DebugGridWorldEnv",
)