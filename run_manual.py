from rm_marl.reward_machine import RewardMachine
from rm_marl.envs.buttons import *
from rm_marl.envs.wrappers import *
from rm_marl.agent import RewardMachineAgent, RewardMachineLearningAgent
from rm_marl.trainer import Trainer
import gym
import os

AGENT_RANDOM_LABEL_CONFIG = {
    1: {"br": (0.3, ButtonsEnv.open_walls_R)},
    2: {"by": (0.3, ButtonsEnv.open_walls_Y), "br": (0.3, ButtonsEnv.open_walls_R)},
    3: {"bg": (0.3, ButtonsEnv.open_walls_G), "br": (0.3, ButtonsEnv.open_walls_R)}
}


def _create_env(agent_id):
    env = gym.make(
        "rm-marl/Buttons-v0",
        render_mode="rgb_array",
        file="/Users/leo/dev/phd/rm-marl/data/buttons/env.txt",
    )
    env = SingleAgentEnvWrapper(env, f"A{agent_id}")
    env = ButtonsLabelingFunctionWrapper(env)
    env = RandomLabelingFunctionWrapper(env, AGENT_RANDOM_LABEL_CONFIG[agent_id])
    env = RewardMachineWrapper(
        env,
        RewardMachine.load_from_file(
            f"data/buttons/rm_agent_{agent_id}.txt"
        ),
        label_mode=RewardMachineWrapper.LabelMode.STATE,
    )
    env = RecordEpisodeStatistics(env)
    return env

def create_local_envs():
    return {f"E{aid}": _create_env(aid) for aid in AGENT_RANDOM_LABEL_CONFIG.keys()}

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

def create_rm_agents(envs):
    return {
        "A1": RewardMachineAgent(
            "A1",
            RewardMachine.load_from_file(
                "/Users/leo/dev/phd/rm-marl/data/buttons/rm_agent_1.txt"
            ),
            algo_kws={
                "action_space": envs["E1"].action_space
            }
        ),
        "A2": RewardMachineAgent(
            "A2",
            RewardMachine.load_from_file(
                "/Users/leo/dev/phd/rm-marl/data/buttons/rm_agent_2.txt"
            ),
            algo_kws={
                "action_space": envs["E2"].action_space
            }
        ),
        "A3": RewardMachineAgent(
            "A3",
            RewardMachine.load_from_file(
                "/Users/leo/dev/phd/rm-marl/data/buttons/rm_agent_3.txt"
            ),
            algo_kws={
                "action_space": envs["E3"].action_space
            }
        )
    }

def create_rm_learning_agents(envs):
    return {
        "A1": RewardMachineLearningAgent(
            "A1",
            algo_kws={
                "action_space": envs["E1"].action_space
            }
        ),
        "A2": RewardMachineLearningAgent(
            "A2",
            algo_kws={
                "action_space": envs["E2"].action_space
            }
        ),
        "A3": RewardMachineLearningAgent(
            "A3",
            algo_kws={
                "action_space": envs["E3"].action_space
            }
        )
    }



if __name__ == "__main__":

    # Training
    # envs = create_local_envs()
    
    # # agents = create_rm_agents(envs)
    # agents = create_rm_learning_agents(envs)

    # trainer = Trainer(envs, agents)

    # trainer.run({
    #     "experiment": "buttons",
    #     "training": True,
    #     "total_episodes": 10000,
    #     "log_dir": os.path.join(os.path.dirname(__file__), "logs"),
    #     "log_freq": 1,
    #     "recording_freq": 500,
    #     "seed": 123
    # })

##########################################################

    # Evaluation
    # path = "logs/buttons_single_agent/2023-01-01_16-26-25/trainer.pkl"
    # trainer_load = Trainer.load(path)
    # agents = trainer_load.agents
    
    # envs = create_shared_env()

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
        a = input("action for A1: ")
        obs, reward, terminated, truncated, info = env.step({"A1": int(a)})
        done = terminated or truncated
        print(info)

############################################
    
