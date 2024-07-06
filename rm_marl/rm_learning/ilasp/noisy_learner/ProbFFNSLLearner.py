import copy
import itertools
import os
import random
from typing import List, Dict

import numpy as np
from sklearn.metrics import log_loss

from rm_marl.reward_machine import RewardMachine
from rm_marl.rm_learning import RMLearner
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


class ProbFFNSLLearner(RMLearner):
    """
    # edge_cost - ILASP penalty for using the ed predicate
    # n_phi_cost - ILASP penalty for using the n_phi_predicate
    # ex_penalty_multiplier - multipler for the ILASP penalties
    # min_penalty - the penalty threshold for discarding an ILASP example - makes the ILASP task simpler
    """

    def __init__(self, agent_id, edge_cost=2, n_phi_cost=2, ex_penalty_multiplier=1, min_penalty=1,
                 cross_entropy_threshold=0.5):
        super().__init__(agent_id)

        self.examples = MultiISAExampleContainer(min_penalty)

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

        self._seen_positive_traces: List[TraceTracker] = []
        self._seen_negative_traces: List[TraceTracker] = []
        self._seen_incomplete_traces: List[TraceTracker] = []

        # Filename of the currently used ILASP solution
        self._curr_ilasp_solution_filename = None

        self.edge_cost = edge_cost
        self.n_phi_cost = n_phi_cost
        self.ex_penalty_multipler = ex_penalty_multiplier

        # Number of ILASP examples
        self.I = 100
        # ILASP example counter
        self.ex_counter = 0

        random.seed(0)

    # We assume this function be called when a trace is fully generated
    def update_rm(self, curr_rm: RewardMachine, curr_state, trace: TraceTracker, terminated, truncated,
                  is_positive_trace):
        assert isinstance(curr_state, np.ndarray)

        self._store_trace(trace)
        self._update_trace_counters(curr_rm, curr_state, trace)

        if terminated or truncated:
            self._update_examples(trace)
        elif curr_rm.is_state_terminal(curr_state):
            self._update_examples(trace)

        if not self._should_relearn_rm() and not self.overriden_with_debugger:
            return None

        candidate_rm = self._update_reward_machine(curr_rm)
        if candidate_rm:
            self._initialize_trace_counters(candidate_rm)
        return candidate_rm

    def _store_trace(self, trace):
        if trace.is_complete:
            if trace.is_positive:
                self._seen_positive_traces.append(copy.deepcopy(trace))
            else:
                self._seen_negative_traces.append(copy.deepcopy(trace))
        else:
            self._seen_incomplete_traces.append(copy.deepcopy(trace))

    def _update_examples(self, trace: TraceTracker):
        if not trace:
            return False

        if trace.is_positive and trace.is_complete:
            self.num_pos_ex += 1
        if trace.is_complete and not trace.is_positive:
            self.num_neg_ex += 1

        examples, ex_type = self.create_examples_from(trace)
        self.examples.merge(examples, ex_type)

    def _update_reward_machine(self, curr_rm):
        self.rm_learning_counter += 1
        self.last_relearning_trace_num = len(self._seen_positive_traces) + len(self._seen_negative_traces) + len(
            self._seen_incomplete_traces)

        ilasp_task_filename = os.path.join(
            self.log_folder, f"task_{self.rm_learning_counter}"
        )
        ilasp_solution_filename = os.path.join(
            self.log_folder, f"solution_{self.rm_learning_counter}"
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
                new_sol_penalty = get_ilasp_solution_penalty(self.log_folder, ilasp_solution_filename,
                                                             f"{ilasp_task_filename}_examples")
                old_sol_penalty = get_ilasp_solution_penalty(self.log_folder, self._curr_ilasp_solution_filename,
                                                             f"{ilasp_task_filename}_examples")

                # If the RMs are equal or they are equally good for the current task
                if candidate_rm == curr_rm or new_sol_penalty >= old_sol_penalty:
                    self.min_rm_num_episodes *= 2
                    return None

                # self.min_rm_num_episodes = self._initial_min_rm_num_episodes
                self.min_rm_num_episodes *= 2

                rm_plot_filename = os.path.join(
                    self.log_folder, f"plot_{self.rm_learning_counter}"
                )
                candidate_rm.plot(rm_plot_filename)
                self._curr_ilasp_solution_filename = ilasp_solution_filename
                return candidate_rm
            else:
                # Can't solve with the current set of examples. Wait for more traces
                LOGGER.debug(f"[{self.agent_id}] ILASP task unsolvable")
                self.min_rm_num_episodes *= 2
                return None
        else:
            raise RuntimeError(
                "Error: Couldn't find an automaton within the specified timeout!"
            )

    def _solve_ilasp_task(self, ilasp_task_filename, ilasp_solution_filename):
        return solve_ilasp_task(
            ilasp_task_filename,
            ilasp_solution_filename,
            timeout=60 * 10,
            version="2",
            max_body_literals=1,
            binary_folder_name=None,
            compute_minimal=True,
        )

    def _generate_ilasp_task(self, ilasp_task_filename):
        total_ex_sum = int(100 * self.ex_penalty_multipler)
        goal_ex, dend_ex, inc_ex = self.examples.generate_goal_dend_inc(total_ex_sum)

        generate_ilasp_task(
            self.rm_num_states,
            "u_acc",
            "u_rej",
            self.examples.get_observables(),  # self._observables,
            goal_ex,  # self.goal_examples.generate_goal_dend_inc(total_ex_sum),
            dend_ex,  # self.dend_examples.generate_goal_dend_inc(total_ex_sum),
            inc_ex,  # self.inc_examples.generate_goal_dend_inc(total_ex_sum),
            self.log_folder,
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
        if self._num_seen_traces < self.last_relearning_trace_num + self.min_rm_num_episodes:
            return False

        return (self._inf_cross_entropy_recorded or
                self._rm_cross_entropy_sum / self._num_seen_traces > self.cross_entropy_threshold)

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
        return len(self._seen_positive_traces) + len(self._seen_incomplete_traces) + len(self._seen_negative_traces)

    def get_statistics(self):
        avg_cross_entropy = self._rm_cross_entropy_sum / self._num_seen_traces if self._num_seen_traces > 0 else 0
        return {
            "ProbFFNSL/cross_entropy": avg_cross_entropy,
            "ProbFFNSL/last_relearning_trace_num": self.last_relearning_trace_num,
        }

    def create_examples_from(self, trace: TraceTracker) -> (ISAExampleContainer, ISAILASPExample.ExType):
        if trace.is_complete:
            if trace.is_positive:
                ex_type = ISAILASPExample.ExType.GOAL
            else:
                ex_type = ISAILASPExample.ExType.DEND
        else:
            ex_type = ISAILASPExample.ExType.INCOMPLETE

        sol = ISAExampleContainer()
        for i in range(self.I):
            ex_id = f"ex_{self.ex_counter}"
            context = self.create_example_context(trace)
            penalty = 1
            last_predicate = LastPredicate(len(trace.trace) - 1)
            ex = ISAILASPExample(ex_id, penalty, ex_type, context, last_predicate)
            # penalty_threshold=self.ilasp_penalty_threshold)
            ex.compact_observations()
            sol.add(ex)
            self.ex_counter += 1
        return sol, ex_type
        # return sol.as_list()

    # TODO: this method is called often so it might need to be sped up
    def create_example_context(self, trace: TraceTracker) -> List[ObservablePredicate]:
        # Create context
        sol = []
        for time_step, labels in enumerate(trace.trace):
            true_labels = self._sample_dict(labels)
            predicates = [ObservablePredicate(label, time_step) for label in true_labels]
            sol.extend(predicates)
        return sol

    # TODO: this method is called often so it might need to be sped up
    # labels - dictionary of labels paired with their probability
    # returns: keys which are considered as true
    def _sample_dict(self, labels: Dict[str, float]) -> List[str]:
        true_elems = []
        for label, prob in labels.items():
            if random.random() <= prob:
                true_elems.append(label)
        return true_elems
