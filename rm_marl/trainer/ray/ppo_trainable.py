from ray import tune
from ray.rllib.algorithms import PPOConfig
from ray.rllib.algorithms.ppo import PPO


class PPOTrainable(tune.Trainable):
    def setup(self, config):
        # Use AlgorithmConfig to create the PPO algorithm
        self.algo_config = (
            PPOConfig()
            .environment("CartPole-v1")
            .rollouts(num_rollout_workers=1)
            .framework("torch")
            .training(lr=config["lr"])
        )
        self.algo = self.algo_config.build()

    def step(self):
        result = self.algo.train()
        return {"episode_reward_mean": result["episode_reward_mean"], "custom_test": "hahha"}

    def save_checkpoint(self, checkpoint_dir):
        checkpoint_path = self.algo.save(checkpoint_dir)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path):
        self.algo.restore(checkpoint_path)

    def cleanup(self):
        self.algo.stop()
