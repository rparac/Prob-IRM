import ray
from ray import tune
import os

from ray.rllib.algorithms import PPO

# Initialize Ray
ray.init(ignore_reinit_error=True)

# Define the configuration for the PPO algorithm
config = {
    "env": "CartPole-v1",
    "framework": "torch",  # Use PyTorch
    "num_gpus": 1,  # Number of GPUs to use
    "num_workers": 10,  # Number of parallel workers
    "model": {
        "fcnet_hiddens": [256, 256],  # Define the neural network architecture
        "fcnet_activation": "relu",
    },
    "train_batch_size": 4000,  # Training batch size
    "sgd_minibatch_size": 128,  # Size of minibatches for stochastic gradient descent
    "num_sgd_iter": 30,  # Number of SGD iterations
}

# Train the PPO agent
tune.run(
    PPO,
    config=config,
)

# Shut down Ray
ray.shutdown()
