import os
import pickle
import ray
from ray.tune.callback import Callback
from ray.util import PublicAPI

# TODO: unify with algo
save_name = "ilasp_learner.pkl"

@PublicAPI(stability="alpha")
class SaveActorCallback(Callback):

    def on_trial_save(self, iteration, trials, trial, **info):
        if not hasattr(trial.temporary_state.ray_actor, 'get_rm_learner'):
            return super().on_trial_save(iteration, trials, trial, **info)

        checkpoint_path = trial.checkpoint.path
        rm_learner_actor = ray.get(trial.temporary_state.ray_actor.get_rm_learner.remote())        
        state = ray.get(rm_learner_actor.get_state_dict.remote())
        # print(state)
        path = os.path.join(checkpoint_path, save_name)
        with open(path, "wb") as f:
            pickle.dump(state, f)

        return super().on_trial_save(iteration, trials, trial, **info)