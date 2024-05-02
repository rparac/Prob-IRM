import sys
from uuid import uuid1

import gym
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import RandomSampler
from torch.optim import Adadelta

from rm_marl.agent import NoRMAgent
from rm_marl.algo.dqrn.definitions import EpsilonAnnealingTimescale
from rm_marl.algo.dqrn.model import DQRN
from rm_marl.envs.gym_subgoal_automata_wrapper import OfficeWorldOfficeLabelingFunctionWrapper, \
    OfficeWorldCoffeeLabelingFunctionWrapper
from rm_marl.envs.wrappers import NoisyLabelingFunctionComposer
from rm_marl.trainer import Trainer

run_config = {
    'training': True, 'total_episodes': 2000, 'log_freq': 1, 'log_dir': 'logs', 'testing_freq': 400,
    'greedy': True, 'synchronize': False, 'counterfactual_update': False, 'recording_freq': 400,
    'no_display': False, 'seed': 123, 'name': 'test_dqrn', 'extra_debug_information': True,
    'num_envs': 10, 'checkpoint_freq': 1000, 'restart_from_checkpoint': False, 'use_rs': True,
    'rm_learner_kws': {'edge_cost': 2, 'n_phi_cost': 1, 'ex_penalty_multiplier': 2, 'min_penalty': 2,
                       'cross_entropy_threshold': 0.8, 'use_cross_entropy': True}, 'edge_cost': 2,
    'n_phi_cost': 2, 'ex_penalty_multiplier': 1
}
env_config = {}


# important - counterfactual_update needs to be false

def _get_base_env():
    seed = 123
    max_episode_length = 100
    use_restricted_observables = True
    env = gym.make("gym_subgoal_automata:OfficeWorldDeliverCoffee-v0",
                   params={"generation": "custom", "environment_seed": seed, "hide_state_variables": True})
    env = GymSubgoalAutomataAdapter(env, agent_id="A1", render_mode="rgb_array",  # type: ignore
                                    max_episode_length=max_episode_length,
                                    use_restricted_observables=use_restricted_observables)
    office_l = OfficeWorldOfficeLabelingFunctionWrapper(env, sensor_true_confidence=1, sensor_false_confidence=1)
    coffee_l = OfficeWorldCoffeeLabelingFunctionWrapper(env, sensor_true_confidence=1, sensor_false_confidence=1)
    env = NoisyLabelingFunctionComposer([coffee_l, office_l])

    env = RecordEpisodeStatistics(env)  # type: ignore

    return env


def objective(trial):
    num_observables = 2
    seed = 123
    optimizer_cls = Adadelta
    exploration_rate_annealing_timescale = EpsilonAnnealingTimescale.EPISODES

    lr = trial.suggest_int("lr")
    rho = trial.suggest_int("rho")

    optimizer_kws = {"lr": lr, "rho": rho}
    buffer_size = trial.suggest_int("buffer_size", 500, 10000)
    policy_train_freq = trial.suggest_int("policy_train_freq", 1, 32, log=True)
    target_update_freq = trial.suggest_int("target_update_freq", 500, 10000, log=True)
    er_start_size = trial.suggest_int("er_start_size", 1, 1000, log=True)
    er_sequence_length = trial.suggest_int("er_sequence_length", 128, 128)
    er_batch_size = trial.suggest_int("er_batch_size", 1, 32, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.999)
    lstm_hidden_state = trial.suggest_int("lstm_hidden_state", 4, 32)
    embedding_num_layers = trial.suggest_int("embedding_num_layers", 1, 4)
    embedding_layer_size = trial.suggest_int("embedding_layer_size", 4, 32, log=True)
    embedding_output_size = trial.suggest_int("embedding_output_size", 2, 32, log=True)
    use_double_dqn = trial.suggest_categorical("use_double_dqn", [True, False])
    use_gradient_clipping = trial.suggest_categorical("use_gradient_clipping", [True, False])
    exploration_rate_init = 1  # trial.suggest_int("exploration_rate_init", 1)
    exploration_rate_final = 0.1  # trial.suggest_int("exploration_rate_final", 0.1, 0.3)
    exploration_rate_annealing_duration = trial.suggest_int("exploration_rate_annealing_duration", 5000, 500000,
                                                            log=True)

    env = _get_base_env()

    algo = DQRN(env.observation_space, env.action_space, num_observables, seed, buffer_size, policy_train_freq,
                target_update_freq, er_start_size, er_sequence_length, er_batch_size, gamma, optimizer_cls,
                optimizer_kws, lstm_hidden_state, embedding_num_layers, embedding_layer_size, embedding_output_size,
                use_double_dqn, use_gradient_clipping, exploration_rate_annealing_timescale,
                exploration_rate_init, exploration_rate_final, exploration_rate_annealing_duration)

    agent = NoRMAgent(agent_id="A1", algo=algo)
    agent_dict = {agent.agent_id: agent}
    env_dict = {"E1": env}

    run_config["name"] = f"{run_config['name']}/{uuid1()}"

    trainer = Trainer(env_dict, env_dict, agent_dict)
    return trainer.run(run_config, trial)


# Load study
def run(study_name: str, storage_location: str):
    study = optuna.load_study(
        study_name=study_name, storage=storage_location,
        sampler=RandomSampler(), pruner=MedianPruner(),
    )
    study.optimize(objective, n_trials=1)


if __name__ == "__main__":
    args = sys.argv
    study_name = args[1]
    storage_location = args[2]
    run(study_name, storage_location)

# Run one trial
