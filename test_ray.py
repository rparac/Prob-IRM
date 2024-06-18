from rm_marl.trainer.new_trainer import NewTrainer

config = {
    "num_workers": 1,

    "ray_init_config": {
        "runtime_env": {
            "env_vars": {
                "RAY_DEBUG": "1",
            }
        },
    },

    # Config for cartpole
    "env_config": {

    },

    # Config for PPO
    "training_config": {
        "entropy_coeff": 0.02,
        "lr": 2.5e-4,
        "gamma": 0.99,
        "vf_loss_coeff": 0.5,
        "lambda_": 0.95,
        "clip_param": 0.2,
        "vf_clip_param": 0.2,
        "grad_clip": 0.5,
        "grad_clip_by": "value",
        "kl_target": 0.0,
        "num_sgd_iter": 10,
        "use_critic": True,
        "use_gae": True,
    },

    "evaluation": {
        # "evaluation_num_env_runners": 1,
        # "evaluation_interval": 50,
        # "evaluation_duration": 1,
        # "evaluation_duration_unit": "episodes",
        # "evaluation_sample_timeout_s": 600,
        # "evaluation_parallel_to_training": True,
    },


}


def main():
    print("hello world")

    NewTrainer().run(config)


if __name__ == "__main__":
    main()
