from gymnasium.envs.registration import register

register(
    id="VisualMinecraft-v0",
    entry_point="rm_marl.new_stack.env.visual_minecraft.env:GridWorldEnv",
)