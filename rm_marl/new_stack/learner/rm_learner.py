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

from rm_marl.rm_learning.ilasp.ilasp_example_representation import ISAILASPExample, MultiISAExampleContainer, \
    ISAExampleContainer, LastPredicate, ObservablePredicate
from rm_marl.rm_learning.ilasp.task_generator import generate_ilasp_task
from rm_marl.rm_learning.ilasp.task_improvement_validator import get_ilasp_solution_penalty
from rm_marl.rm_learning.ilasp.task_parser import parse_ilasp_solutions
from rm_marl.rm_learning.ilasp.task_solver import solve_ilasp_task
from rm_marl.rm_learning.trace_tracker import TraceTracker
from rm_marl.rm_transition.prob_rm_transitioner import ProbRMTransitioner
from rm_marl.utils.logging import getLogger

LOGGER = getLogger(__name__)



# TODO: implement new API
# TODO: more cpus for this actor. The problem is the placement group allocation at the moment
@ray.remote
class RMLearner:
    """
    # edge_cost - ILASP penalty for using the ed predicate
    # n_phi_cost - ILASP penalty for using the n_phi_predicate
    # ex_penalty_multiplier - multipler for the ILASP penalties. If =2 this means that all ILASP example penalties will
    #                         sum to 200
    # min_penalty - the penalty threshold for discarding an ILASP example - makes the ILASP task simpler
    # base_dir - relative directory path where results will be stored
    """

    def __init__(self, starting_rm, actor_name, edge_cost, n_phi_cost, ex_penalty_multiplier, min_penalty,
                 cross_entropy_threshold, replay_experience, rebalance_classes, max_container_size, new_inc_examples, base_dir):
        self._new_inc_examples = new_inc_examples
        self.examples = MultiISAExampleContainer(min_penalty, rebalance_classes, new_inc_examples, max_container_size)

        self.actor_name = actor_name

        # Minimum is 3 states (accepting, rejecting, u0)
        self.rm_num_states = 8

        # Minimum number of new traces before we validate if the reward machine is the correct one
        self._initial_min_rm_num_episodes = 100
        self.min_rm_num_episodes = self._initial_min_rm_num_episodes

        # the number of traces when the automata was relearned
        self.last_relearning_trace_num = 0

        # The percentage of traces that need to conform to the reward
        # machine to avoid relearning
        self.rm_recognize_threshold = 0.4
        self.cross_entropy_threshold = cross_entropy_threshold

        # Debug tracking
        self.num_pos_ex = 0
        self.num_neg_ex = 0
        self.overriden_with_debugger = False

        self._rm_cross_entropy_sum = 0
        # variable to track if infinity cross entropy is recorded
        self._inf_cross_entropy_recorded = False

        self.replay_experience = replay_experience
        if replay_experience:
            self._seen_positive_traces: List[TraceTracker] = []
            self._seen_negative_traces: List[TraceTracker] = []
            self._seen_incomplete_traces: List[TraceTracker] = []
        self._num_pos_traces = 0
        self._num_neg_traces = 0
        self._num_inc_traces = 0

        # Filename of the currently used ILASP solution
        self._curr_ilasp_solution_filename = None

        self.edge_cost = edge_cost
        self.n_phi_cost = n_phi_cost
        self.ex_penalty_multipler = ex_penalty_multiplier

        # Number of ILASP examples
        # self.I = 100
        self.I = 10
        # ILASP example counter
        self.ex_counter = 0

        self.rm_learning_counter = 0

        self.curr_rm = starting_rm

        self._base_dir = base_dir
        self._log_folder = None
        random.seed(0)

        self._record_memory_every = 100
        self._curr_memory_step = 0

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

    def _print_memory_report(self):
        # for attr_name, attr_value in self.__dict__.items():
        #     attr_size = asizeof.asizeof(attr_value) / (1024 ** 3)
        #     print(f"Memory usage of '{attr_name}': {attr_size} GB")

        print(f"The size of examples is {asizeof.asizeof(self.examples) / (1024 ** 3)} Gb")
        print(f"The size of the whole object is {asizeof.asizeof(self)/ (1024 ** 3)} Gb")
        self._curr_memory_step = 0

    def relearn_rm(self):
        curr_rm = self.curr_rm

        self._curr_memory_step += 1
        if self._curr_memory_step >= self._record_memory_every:
            self._print_memory_report()

        if not self._should_relearn_rm() and not self.overriden_with_debugger:
            return None

        if self._log_folder is None:
            self._create_dir()

        candidate_rm = self._update_reward_machine(curr_rm)
        if candidate_rm:
            if self.replay_experience:
                self._initialize_trace_counters(candidate_rm)
            else:
                self._rm_cross_entropy_sum = 0
                self._inf_cross_entropy_recorded = False
                self._num_pos_traces = 0
                self._num_neg_traces = 0
                self._num_inc_traces = 0
            self.curr_rm = candidate_rm
        return candidate_rm

    # TODO: remove duplication; this is computed by the connector
    def _compute_curr_state_from_trace(self, trace):
        rm_transitioner = ProbRMTransitioner(self.curr_rm)
        curr_state = rm_transitioner.get_initial_state()
        for labels in trace.trace:
            curr_state = rm_transitioner.get_next_state(curr_state, labels)
        return curr_state

    def batch_update_examples(self, traces):
        for trace in traces:
            self.update_examples(trace)

    def update_examples(self, trace):
        if not trace:
            return False

        # raise RuntimeError(trace.trace, trace.is_positive, trace.is_complete)
        self._store_trace(trace)
        self._update_trace_counters(self.curr_rm, self._compute_curr_state_from_trace(trace), trace)

        if trace.is_positive and trace.is_complete:
            self.num_pos_ex += 1
        if trace.is_complete and not trace.is_positive:
            self.num_neg_ex += 1

        examples, ex_type = self.create_examples_from(trace)
        # breakpoint()
        # curr_ex = list(examples.storage.keys())[0]
        # if len(curr_ex.observable_context) == 1 and curr_ex.observable_context[-1] == ObservablePredicate("g",
        #                                                                                                  0) and curr_ex.example_type == ISAILASPExample.ExType.GOAL:
        #     breakpoint()
        self.examples.merge(examples, ex_type)

    def _store_trace(self, trace):
        if trace.is_complete:
            if trace.is_positive:
                self._num_pos_traces += 1
                if self.replay_experience:
                    self._seen_positive_traces.append(copy.deepcopy(trace))
            else:
                self._num_neg_traces += 1
                if self.replay_experience:
                    self._seen_negative_traces.append(copy.deepcopy(trace))
        else:
            self._num_inc_traces += 1
            if self.replay_experience:
                self._seen_incomplete_traces.append(copy.deepcopy(trace))

    def _update_reward_machine(self, curr_rm):
        self.rm_learning_counter += 1
        if self.replay_experience:
            self.last_relearning_trace_num = len(self._seen_positive_traces) + len(self._seen_negative_traces) + len(self._seen_incomplete_traces) 
        else:
            self.last_relearning_trace_num += self._num_pos_traces + self._num_neg_traces + self._num_inc_traces

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

                # TODO: abstract example file name away
                new_sol_penalty = get_ilasp_solution_penalty(self._log_folder, ilasp_solution_filename,
                                                             f"{ilasp_task_filename}_examples")
                old_sol_penalty = get_ilasp_solution_penalty(self._log_folder, self._curr_ilasp_solution_filename,
                                                             f"{ilasp_task_filename}_examples")

                # If the RMs are equal or they are equally good for the current task
                if candidate_rm == curr_rm or new_sol_penalty >= old_sol_penalty:
                    self.min_rm_num_episodes *= 2
                    return None

                # self.min_rm_num_episodes = self._initial_min_rm_num_episodes
                self.min_rm_num_episodes *= 2

                rm_plot_filename = os.path.join(
                    self._log_folder, f"plot_{self.rm_learning_counter}"
                )
                candidate_rm.plot(rm_plot_filename)
                self._curr_ilasp_solution_filename = ilasp_solution_filename
                return candidate_rm
            else:
                # Can't solve with the current set of examples. Wait for more traces
                LOGGER.debug(f"ILASP task unsolvable")
                self.min_rm_num_episodes *= 2
                return None
        else:
            # Can't solve in specified time
            LOGGER.debug(f"ILASP task timeout")
            self.min_rm_num_episodes *= 2
            return None

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

    def _generate_ilasp_task(self, ilasp_task_filename):
        total_ex_sum = int(100 * self.ex_penalty_multipler)
        goal_ex, dend_ex, inc_ex = self.examples.generate_goal_dend_inc(total_ex_sum)
        # breakpoint()

        generate_ilasp_task(
            self.rm_num_states,
            "u_acc",
            "u_rej",
            self.examples.get_observables(),  # self._observables,
            goal_ex,  # self.goal_examples.generate_goal_dend_inc(total_ex_sum),
            dend_ex,  # self.dend_examples.generate_goal_dend_inc(total_ex_sum),
            inc_ex,  # self.inc_examples.generate_goal_dend_inc(total_ex_sum),
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
            n_phi_cost=self.n_phi_cost,
            edge_cost=self.edge_cost,
        )

    def _should_relearn_rm(self) -> bool:
        # Num seen traces only contains new traces when we are not replaying experience
        if not self.replay_experience and self._num_seen_traces < self.min_rm_num_episodes:
            return False
        if self.replay_experience and self._num_seen_traces < self.last_relearning_trace_num + self.min_rm_num_episodes:
            return False

        condition_satisfied = (
                self._inf_cross_entropy_recorded or self._rm_cross_entropy_sum / self._num_seen_traces > self.cross_entropy_threshold)
        if condition_satisfied:
            print(self._inf_cross_entropy_recorded, self._rm_cross_entropy_sum / self._num_seen_traces)
        return condition_satisfied

    def _update_trace_counters(self, curr_rm, curr_state, trace):
        # Set the expected belief based on the trace outcome
        # accepting, rejecting, incomplete
        true_vec = [0, 0, 0]
        accepting_idx, rejecting_idx, incomplete_idx = 0, 1, 2
        if trace.is_complete:
            if trace.is_positive:
                true_vec[accepting_idx] = 1
            else:
                true_vec[rejecting_idx] = 1
        else:
            true_vec[incomplete_idx] = 1

        pred_vec = [0, 0, 0]
        pred_vec[accepting_idx] = curr_rm.accepting_state_prob(curr_state)
        pred_vec[rejecting_idx] = curr_rm.rejecting_state_prob(curr_state)
        pred_vec[incomplete_idx] = 1 - curr_rm.accepting_state_prob(curr_state) - curr_rm.rejecting_state_prob(
            curr_state)

        # Check if cross entropy should be infinity.
        # We make the loss extremely large to always trigger relearning
        if np.isclose(pred_vec[np.argmax(true_vec)], 0):
            # print(f"Inf cross entropy recorded: pred vec is {pred_vec}, while {true_vec}, {curr_state}, {trace.trace}")
            self._inf_cross_entropy_recorded = True
        loss_val = log_loss(true_vec, pred_vec)
        self._rm_cross_entropy_sum += loss_val

    # Replays old traces to the success rate
    def _initialize_trace_counters(self, candidate_rm):
        self._rm_cross_entropy_sum = 0
        self._inf_cross_entropy_recorded = False

        transitioner = ProbRMTransitioner(candidate_rm)
        for trace in itertools.chain(self._seen_positive_traces, self._seen_negative_traces,
                                     self._seen_incomplete_traces):
            curr_state = transitioner.get_initial_state()
            for event in trace.trace:
                curr_state = transitioner.get_next_state(curr_state, event)

            self._update_trace_counters(candidate_rm, curr_state, trace)

    @property
    def _num_seen_traces(self):
        if self.replay_experience:        
            return len(self._seen_positive_traces) + len(self._seen_incomplete_traces) + len(self._seen_negative_traces)
        return self._num_pos_traces + self._num_neg_traces + self._num_inc_traces

    def get_statistics(self):
        avg_cross_entropy = self._rm_cross_entropy_sum / self._num_seen_traces if self._num_seen_traces > 0 else 0
        return {
            "cross_entropy": avg_cross_entropy,
            "last_relearning_trace_num": self.last_relearning_trace_num,
            "num_pos_traces": len(self._seen_positive_traces) if self.replay_experience else self._num_pos_traces,
            "num_neg_traces": len(self._seen_negative_traces) if self.replay_experience else self._num_neg_traces,
            "num_incomplete_traces": len(self._seen_incomplete_traces) if self.replay_experience else self._num_inc_traces,
            "warmup_wait": self.min_rm_num_episodes,
        }

    # TODO: remove duplicatios
    def create_examples_from(self, trace: TraceTracker) -> tuple[ISAExampleContainer, ISAILASPExample.ExType]:
        if trace.is_complete:
            if trace.is_positive:
                ex_type = ISAILASPExample.ExType.GOAL
            else:
                ex_type = ISAILASPExample.ExType.DEND
        else:
            ex_type = ISAILASPExample.ExType.INCOMPLETE

        ctxs = self.create_multiple_example_contexts(trace, self.I)

        sol = ISAExampleContainer()
        for i in range(self.I):
            ex_id = f"ex_{self.ex_counter}"
            # context = self.create_example_context(trace)
            context = ctxs[i]
            penalty = 1
            last_predicate = LastPredicate(len(trace.trace) - 1)
            ex = ISAILASPExample(ex_id, penalty, ex_type, context, last_predicate, new_inc_example=self._new_inc_examples)
            # penalty_threshold=self.ilasp_penalty_threshold)
            ex.compact_observations()
            sol.add(ex)
            self.ex_counter += 1
        return sol, ex_type
        # return sol.as_list()

    # This method has the same functionality as create_example_context but done in bulk. 
    #   This choice is made for optimization purposes.
    def create_multiple_example_contexts(self, trace: TraceTracker, n) -> List[List[ObservablePredicate]]:
        # Create context
        all_labels = list(trace.trace[0].keys())
        num_labels = len(all_labels)

        sol = np.zeros((len(trace.trace), num_labels, n))

        for time_step, labels in enumerate(trace.trace):
            for i, (label, prob) in enumerate(labels.items()):
                if prob > 0:
                    sol[time_step,i,:] = np.random.rand(n) <= prob

        ret = [[] for _ in range(n)]

        for time_step in range(len(trace.trace)):
            for j in range(num_labels):
                for k in range(n):
                    if sol[time_step,j,k] != 0:
                        label = all_labels[j] 
                        ret[k].append(ObservablePredicate(label, time_step))

        return ret

    def create_example_context(self, trace: TraceTracker) -> List[ObservablePredicate]:
        # Create context
        sol = []

        for time_step, labels in enumerate(trace.trace):
            true_labels = self._sample_dict(labels)
            predicates = [ObservablePredicate(label, time_step) for label in true_labels]
            sol.extend(predicates)
        return sol

    # This method is inlined in create example context for speed up
    # labels - dictionary of labels paired with their probability
    # returns: keys which are considered as true
    def _sample_dict(self, labels: Dict[str, float]) -> Iterator[str]:
        true_elems = []
        for label, prob in labels.items():
            if prob > 0 and random.random() <= prob:
                true_elems.append(label)
        return true_elems

    # Used for serialization
    def get_state_dict(self):
        return self.__dict__

    def set_state_dict(self, d):
        self.__dict__ = d
        print(f"Log folder is {self.log_folder()}")

