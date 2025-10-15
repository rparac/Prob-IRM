import copy
import itertools
import os
import random
import datetime as dt
from typing import Iterator, List, Dict

import numpy as np
import ray
from sklearn.metrics import log_loss
from pympler import asizeof

from rm_marl.new_stack.learner.util import generate_previous_incomplete_examples
from rm_marl.reward_machine import RewardMachine
from rm_marl.rm_learning.ilasp.ilasp_example_representation import ISAILASPExample, MultiISAExampleContainer, \
    ISAExampleContainer, LastPredicate, ObservablePredicate, lift_dend_example, lift_goal_example, lift_inc_example, remove_duplicates
from rm_marl.rm_learning.ilasp.task_generator import generate_ilasp_task
from rm_marl.rm_learning.ilasp.task_improvement_validator import get_ilasp_solution_penalty
from rm_marl.rm_learning.ilasp.task_parser import parse_ilasp_solutions
from rm_marl.rm_learning.ilasp.task_solver import solve_ilasp_task
from rm_marl.rm_learning.trace_tracker import TraceTracker
from rm_marl.rm_transition.deterministic_rm_transitioner import DeterministicRMTransitioner
from rm_marl.rm_transition.prob_rm_transitioner import ProbRMTransitioner
from rm_marl.rm_transition.rm_transitioner import RMTransitioner
from rm_marl.utils.logging import getLogger

LOGGER = getLogger(__name__)



@ray.remote
class NonNoisyRMLearner:
    """
    """

    def __init__(self, starting_rm, actor_name, base_dir, **kwargs):
        self.actor_name = actor_name

        # Minimum is 3 states (accepting, rejecting, u0)
        self.rm_num_states = 8

        # Minimum number of new traces before we validate if the reward machine is the correct one
        self._initial_min_rm_num_episodes = 100
        self.min_rm_num_episodes = self._initial_min_rm_num_episodes

        # the number of traces when the automata was relearned
        self.last_relearning_trace_num = 0


        # Filename of the currently used ILASP solution
        self._curr_ilasp_solution_filename = None

        # Number of ILASP examples
        # ILASP example counter
        self.ex_counter = 0

        self.rm_learning_counter = 0

        self.curr_rm = starting_rm

        self._seen_positive_counter_examples = set()
        self._seen_negative_counter_examples = set()
        self._seen_incomplete_counter_examples = set()
        self._task_unsolvable = False
        # If the task is already too complex, we will not try to solve it
        self._task_too_complex = False
        self._last_num_counterexamples = 0


        self._base_dir = base_dir 
        self._log_folder = None
        random.seed(0)

    def _get_num_counterexamples(self):
        return len(self._seen_positive_counter_examples) + len(self._seen_negative_counter_examples) + len(self._seen_incomplete_counter_examples)

    def get_curr_rm(self):
        return self.curr_rm

    def _create_dir(self):
        log_id = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self._log_folder = f'{os.getcwd()}/logs/{self._base_dir}/{log_id}-{self.actor_name}'
        os.makedirs(self._log_folder, exist_ok=True)

    def log_folder(self):
        return self._log_folder

    def get_rm(self):
        return self.curr_rm

    def relearn_rm(self):
        if self._get_num_counterexamples() == self._last_num_counterexamples:
            # We have not seen a new counter-example
            return None

        if self._log_folder is None:
            self._create_dir()

        candidate_rm = self._update_reward_machine()
        if candidate_rm:
            self.curr_rm = candidate_rm
        return candidate_rm

    # TODO: remove duplication; this is computed by the connector
    def _compute_curr_state_from_trace(self, trace):
        rm_transitioner = ProbRMTransitioner(self.curr_rm)
        curr_state = rm_transitioner.get_initial_state()
        for labels in trace.labels_sequence:
            curr_state = rm_transitioner.get_next_state(curr_state, labels)
        return curr_state

    def batch_update_examples(self, traces):
        for trace in traces:
            self.update_examples(trace)

    
    def _update_counterexamples(self, trace):
        if self._task_unsolvable or self._task_too_complex:
            # No need to update counterexamples if the task is already unsolvable
            return False

        rm_transitioner = DeterministicRMTransitioner(self.curr_rm)

        curr_state = rm_transitioner.get_initial_state()
        for labels in trace.labels_sequence:
            curr_state = rm_transitioner.get_next_state(curr_state, labels)

        if not trace.is_complete:
            if self.curr_rm.is_state_terminal(curr_state):
                # A counter-example occurs if the state is terminal but the trace is not
                self._seen_incomplete_counter_examples.add(copy.deepcopy(trace))
                return True
            return False


        if trace.is_positive:
            # A counter-example occurs if the trace is goal, but the RM state is not accepting 
            if not self.curr_rm.is_accepting_state(curr_state):
                self._seen_positive_counter_examples.add(copy.deepcopy(trace))
                return True
            return False

        if not self.curr_rm.is_rejecting_state(curr_state):
            # A counter-example occurs if the trace is deadend, but the RM state is not rejecting
            self._seen_negative_counter_examples.add(copy.deepcopy(trace))
            return True
        return False


    def update_examples(self, trace):
        if not trace:
            return

        self._update_counterexamples(trace)


    def _update_reward_machine(self):
        if self._task_unsolvable or self._task_too_complex:
            # If task is already unsolvable, adding more examples will not help
            return None

        self.rm_learning_counter += 1

        ilasp_task_filename = os.path.join(
            self._log_folder, f"task_{self.rm_learning_counter}"
        )
        ilasp_solution_filename = os.path.join(
            self._log_folder, f"solution_{self.rm_learning_counter}"
        )

        self._generate_ilasp_task(ilasp_task_filename)
        solver_completed = self._solve_ilasp_task(ilasp_task_filename, ilasp_solution_filename)
        if solver_completed:
            candidate_rm = parse_ilasp_solutions(ilasp_solution_filename)

            # TODO: remove duplication here with ILASPLearner
            if candidate_rm.states:
                candidate_rm.set_u0("u0")
                if "u_acc" in candidate_rm.states:
                    candidate_rm.set_uacc("u_acc")
                if "u_rej" in candidate_rm.states:
                    candidate_rm.set_urej("u_rej")

                rm_plot_filename = os.path.join(
                    self._log_folder, f"plot_{self.rm_learning_counter}"
                )
                candidate_rm.plot(rm_plot_filename)
                self._curr_ilasp_solution_filename = ilasp_solution_filename
                self._last_num_counterexamples = self._get_num_counterexamples()
                return candidate_rm
            else:
                # Can't solve the task
                self._task_unsolvable = True
                # Can't solve with the current set of examples nor adding more will help. 
                LOGGER.debug(f"ILASP task unsolvable")
                # We set the RM to the default RM to get results without the RM
                candidate_rm = RewardMachine.default_rm()
                return candidate_rm
        else:
            LOGGER.debug(f"ILASP task timeout")
            self._task_too_complex = True
            candidate_rm = RewardMachine.default_rm()
            return candidate_rm

    def _solve_ilasp_task(self, ilasp_task_filename, ilasp_solution_filename):
        return solve_ilasp_task(
            ilasp_task_filename,
            ilasp_solution_filename,
            timeout=60 * 60, # 60 minutes * 60 seconds
            version="2",
            max_body_literals=1,
            binary_folder_name=None,
            compute_minimal=True,
        )


    @property
    def observables(self):
        union = set((l for e in self._seen_incomplete_counter_examples for ls in e.labels_sequence for l in ls)).union(
            set((l for e in self._seen_positive_counter_examples for ls in e.labels_sequence for l in ls))).union(
            set((l for e in self._seen_negative_counter_examples for ls in e.labels_sequence for l in ls))
        )
        return union

    def _generate_ilasp_task(self, ilasp_task_filename):
        positive_examples = remove_duplicates([lift_goal_example(ex.labels_sequence, f"ex_goal_{i}") for i, ex in enumerate(self._seen_positive_counter_examples)])
        negative_examples = remove_duplicates([lift_dend_example(ex.labels_sequence, f"ex_dend_{i}") for i, ex in enumerate(self._seen_negative_counter_examples)])
        incomplete_examples = remove_duplicates([lift_inc_example(ex.labels_sequence, f"ex_inc_{i}") for i, ex in enumerate(self._seen_incomplete_counter_examples)])

        # the sets of examples are sorted to make sure that ILASP produces the same solution for the same sets (ILASP
        # can produce different hypothesis for the same set of examples but given in different order)
        generate_ilasp_task(
            self.rm_num_states,
            "u_acc",
            "u_rej",
            self.observables,
            positive_examples,
            negative_examples,
            incomplete_examples,
            self._log_folder,
            ilasp_task_filename,
            symmetry_breaking_method="bfs-alternative",
            max_disj_size=1,
            learn_acyclic=True,
            use_compressed_traces=True,
            avoid_learning_only_negative=True,
            prioritize_optimal_solutions=False,
            use_state_id_restrictions=False,  # True,  # states used need to be used in order
            binary_folder_name=None,
        )



    def get_statistics(self):
        return {}


    # Used for serialization
    def get_state_dict(self):
        return self.__dict__

    def set_state_dict(self, d):
        self.__dict__ = d
        print(f"Log folder is {self.log_folder()}")

