from rm_marl.reward_machine import RewardMachine
from rm_marl.envs.buttons import *
from rm_marl.envs.wrappers import *
from rm_marl.agent import RewardMachineAgent, RewardMachineLearningAgent
from rm_marl.trainer import Trainer
import gym
import os


def create_env1():
    env = gym.make(
        "rm-marl/Buttons-v0",
        render_mode="rgb_array",
        file="/Users/leo/dev/phd/rm-marl/data/buttons/env.txt",
    )
    # env = LabelingFunctionWrapper(env)
    env = SingleAgentEnvWrapper(env, "A1")
    env = ButtonsLabelingFunctionWrapper(env)
    env = RandomLabelingFunctionWrapper(env, {"br": (0.3, ButtonsEnv.open_walls_R)})
    env = RewardMachineWrapper(
        env,
        RewardMachine.load_from_file(
            "/Users/leo/dev/phd/rm-marl/data/buttons/rm_agent_1.txt"
        ),
        label_mode=RewardMachineWrapper.LabelMode.STATE,
    )
    env = RecordEpisodeStatistics(env)
    return env

def create_env2():
    env = gym.make(
        "rm-marl/Buttons-v0",
        render_mode="rgb_array",
        file="/Users/leo/dev/phd/rm-marl/data/buttons/env.txt",
    )
    # env = LabelingFunctionWrapper(env)
    env = SingleAgentEnvWrapper(env, "A2")
    env = ButtonsLabelingFunctionWrapper(env)
    env = RandomLabelingFunctionWrapper(env, {"by": (0.3, ButtonsEnv.open_walls_Y), "br": (0.3, ButtonsEnv.open_walls_R)})
    env = RewardMachineWrapper(
        env,
        RewardMachine.load_from_file(
            "/Users/leo/dev/phd/rm-marl/data/buttons/rm_agent_2.txt"
        ),
        label_mode=RewardMachineWrapper.LabelMode.STATE,
    )
    env = RecordEpisodeStatistics(env)
    return env

def create_env3():
    env = gym.make(
        "rm-marl/Buttons-v0",
        render_mode="rgb_array",
        file="/Users/leo/dev/phd/rm-marl/data/buttons/env.txt",
    )
    # env = LabelingFunctionWrapper(env)
    env = SingleAgentEnvWrapper(env, "A3")
    env = ButtonsLabelingFunctionWrapper(env)
    env = RandomLabelingFunctionWrapper(env, {"bg": (0.3, ButtonsEnv.open_walls_G), "br": (0.3, ButtonsEnv.open_walls_R)})
    env = RewardMachineWrapper(
        env,
        RewardMachine.load_from_file(
            "/Users/leo/dev/phd/rm-marl/data/buttons/rm_agent_3.txt"
        ),
        label_mode=RewardMachineWrapper.LabelMode.STATE,
    )
    env = RecordEpisodeStatistics(env)
    return env

def create_shared_env():
    env = gym.make(
        "rm-marl/Buttons-v0",
        render_mode="rgb_array",
        file="/Users/leo/dev/phd/rm-marl/data/buttons/env.txt",
    )
    env = ButtonsLabelingFunctionWrapper(env)
    env = RewardMachineWrapper(
        env,
        RewardMachine.load_from_file(
            "/Users/leo/dev/phd/rm-marl/data/buttons/rm_team.txt"
        ),
        label_mode=RewardMachineWrapper.LabelMode.STATE,
    )
    env = RecordEpisodeStatistics(env)
    return {"G": env}

def create_envs():
    return {
        "E1": create_env1(),
        "E2": create_env2(),
        # "E3": create_env3(),
    }

def create_agents(envs):
    return {
        "A1": RewardMachineAgent(
            RewardMachine.load_from_file(
                "/Users/leo/dev/phd/rm-marl/data/buttons/rm_agent_1.txt"
            ),
            algo_kws={
                "action_space": envs["E1"].action_space
            }
        ),
        "A2": RewardMachineAgent(
            RewardMachine.load_from_file(
                "/Users/leo/dev/phd/rm-marl/data/buttons/rm_agent_2.txt"
            ),
            algo_kws={
                "action_space": envs["E2"].action_space
            }
        ),
        "A3": RewardMachineAgent(
            RewardMachine.load_from_file(
                "/Users/leo/dev/phd/rm-marl/data/buttons/rm_agent_3.txt"
            ),
            algo_kws={
                "action_space": envs["E3"].action_space
            }
        )
    }

def create_learning_agents(envs):
    return {
        "A1": RewardMachineLearningAgent(
            algo_kws={
                "action_space": envs["E1"].action_space
            }
        ),
        "A2": RewardMachineLearningAgent(
            algo_kws={
                "action_space": envs["E2"].action_space
            }
        ),
        "A3": RewardMachineLearningAgent(
            algo_kws={
                "action_space": envs["E3"].action_space
            }
        )
    }



if __name__ == "__main__":

    # envs = create_envs()
    # # agents = create_agents(envs)
    # agents = create_learning_agents(envs)

    # trainer = Trainer(envs, agents)

    # trainer.run({
    #     "experiment": "buttons_single_agent",
    #     "training": True,
    #     "total_episodes": 10000,
    #     "log_dir": os.path.join(os.path.dirname(__file__), "logs"),
    #     "log_freq": 1,
    #     "recording_freq": 500,
    #     "seed": 123
    # })

########################################################################################

    ## MANUAL RUN
    
    env = gym.make(
        "rm-marl/Buttons-v0",
        render_mode="rgb_array",
        file="/Users/leo/dev/phd/rm-marl/data/buttons/env.txt",
    )
    env.unwrapped.render_mode="human"
    
    done = False
    obs, info = env.reset()

    while not done:
        env.render()
        a = input("action: ")
        obs, reward, terminated, truncated, info = env.step({"A1": int(a)})
        done = terminated or truncated
        print(info)

############################################
    # path = "/Users/leo/dev/phd/rm-marl/logs/buttons_single_agent/2023-01-01_16-26-25/trainer.pkl"
    # trainer_load = Trainer.load(path)
    # envs = create_shared_env()
    # # envs = {"E3": create_env3()}
    # agents = trainer_load.agents

    # trainer = Trainer(envs, agents)

    # trainer.run({
    #     "experiment": "buttons_single_agent_test",
    #     "training": False,
    #     "greedy": False,
    #     "total_episodes": 1,
    #     "log_dir": os.path.join(os.path.dirname(__file__), "logs"),
    #     "log_freq": 1,
    #     "recording_freq": 1,
    #     "seed": 10000 + 123
    # })
