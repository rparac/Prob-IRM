"""
Wrapper around https://github.com/ertsiger/gym-subgoal-automata/tree/master
which provides enviornments such as coffee world and water world.

Sample usage:
# See the link above for other options
env = gym.make("gym_subgoal_automata:OfficeWorldDeliverCoffee-v0",
               params={"hide_state_variables": true, ...})
env = DanielGymAdapter(env)
"""
import gym

from gym_subgoal_automata.envs.base.base_env import BaseEnv
from rm_marl.envs.wrappers import LabelingFunctionWrapper
from rm_marl.reward_machine import RewardMachine


class DanielGymAdapter(gym.Wrapper):
    def __init__(self, env: BaseEnv, render_mode=None, max_episode_length=None):
        # Explicitly returns observables as a part of the observation.
        # We regenerate them in this adapter using the info output.
        env.hide_state_variables = True
        super().__init__(env)

        assert render_mode in ["human", "rgb_array"]
        self._render_mode = render_mode
        self.env = env
        self.observables = self.env.get_restricted_observables()
        self.max_episode_length = max_episode_length
        self.current_step = 0
        self.agent_id = "A1"

        observables_obs_space = {
            observable: gym.spaces.Discrete(2)
            for observable in self.observables
        }
        self.unflatten_obs_space = gym.spaces.Dict({
            "underlying_obs_space": env.observation_space,
            **observables_obs_space,
        })
        self.observation_space = gym.spaces.Dict(
            {self.agent_id: gym.spaces.utils.flatten_space(self.unflatten_obs_space)})

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_step = 0
        if self._render_mode == "human":
            self.env.render(self._render_mode)

        info["is_positive_trace"] = False
        return self._to_new_obs(obs, info), {}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action[self.agent_id])
        self.current_step += 1
        if self._render_mode == "human":
            self.env.render(self._render_mode)

        if self.max_episode_length and self.current_step >= self.max_episode_length:
            truncated = True

        info["is_positive_trace"] = reward > 0
        return self._to_new_obs(obs, info), reward, terminated, truncated, info

    def _to_new_obs(self, old_obs, old_info):
        seen_observables = old_info.get("observations", set())
        new_obs = {
            observable: observable in seen_observables
            for observable in self.observables
        }
        new_obs["underlying_obs_space"] = old_obs

        new_obs = {self.agent_id: gym.spaces.utils.flatten(self.unflatten_obs_space, new_obs)}
        return new_obs

    def render(self, **kwargs):
        if self._render_mode == "rgb_array":
            return self.env.render(self._render_mode)

    # Converts subgoal automaton to Reward Machine
    def get_perfect_rm(self):
        subgoal_automaton = self.env.get_automaton()

        rm = RewardMachine()
        rm.add_states(subgoal_automaton.states)
        rm.set_u0(subgoal_automaton.initial_state)
        rm.set_uacc(subgoal_automaton.accept_state)
        rm.set_urej(subgoal_automaton.reject_state)

        for from_state, l_cond_to_state in subgoal_automaton.edges.items():
            for conditions, to_state in l_cond_to_state:
                for condition in conditions:
                    rm.add_transition(from_state, to_state, condition)
        return rm

    @staticmethod
    def _split_conditions(conditions: str):
        return conditions.split('&')


class OfficeWorldDeliverCoffeeLabelingFunctionWrapper(LabelingFunctionWrapper):
    """
    Looked at https://github.com/ertsiger/gym-subgoal-automata/blob/1879a6512441cdf0758c937cc659931d49260d38/gym_subgoal_automata/envs/officeworld/officeworld_env.py#L9-L18
    to find object ids
    """

    def __init__(self, env: DanielGymAdapter):
        super().__init__(env)
        self.agent_id = "A1"
        self.env = env

    def get_labels(self, obs: dict = None, prev_obs: dict = None):
        """Returns a modified observation."""
        labels = []

        unwrapped_obs = gym.spaces.unflatten(self.env.unflatten_obs_space, obs[self.agent_id])
        if unwrapped_obs["f"]:
            labels.append('f')
        if unwrapped_obs["g"]:
            labels.append('g')
        if unwrapped_obs["n"]:
            labels.append('n')

        return labels

    def get_all_labels(self):
        return [
            'f',  # coffee
            'g',  # office
            'n',  # plant
        ]
