import copy
import itertools
import os
from typing import List

import numpy as np
from sklearn.metrics import log_loss

from rm_marl.reward_machine import RewardMachine
from rm_marl.rm_learning import RMLearner
from rm_marl.rm_learning.ilasp.ilasp_example_representation import ISAILASPExample, MultiISAExampleContainer
from rm_marl.rm_learning.ilasp.noisy_learner.example_generator import NoisyILASPExampleGenerator
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
                 cross_entropy_threshold=0.5, use_cross_entropy=True):
        super().__init__(agent_id)

        # self.goal_examples = ISAExampleContainer()
        # self.dend_examples = ISAExampleContainer()
        # self.inc_examples = ISAExampleContainer()
        self.examples = MultiISAExampleContainer(min_penalty)

        self.ex_generator = NoisyILASPExampleGenerator()

        # Minimum is 3 states (accepting, rejecting, u0)
        # self.rm_num_states = 4
        self.rm_num_states = 8  # 5

        # Minimum number of new traces before we validate if the
        # reward machine is the correct one
        # TODO: current choice is to double this number every time the same RM is learned. Need to think this through
        self._initial_min_rm_num_episodes = 100  # 10
        self.min_rm_num_episodes = self._initial_min_rm_num_episodes

        # the number of traces when the automata was relearned
        self.last_relearning_trace_num = 0

        # The percentage of traces that need to conform to the reward
        # machine to avoid relearning
        self.rm_recognize_threshold = 0.4  # 0.6  # 0.35
        self.cross_entropy_threshold = cross_entropy_threshold
        # TODO: remove after experiment with cross entropy
        self.use_cross_entropy = use_cross_entropy

        # Debug tracking
        self.num_pos_ex = 0
        self.num_neg_ex = 0
        self.overriden_with_debugger = False

        self._rm_cnt_since_restart = 0

        # TODO: delete after debugging is finished
        self._debug_ratio = []

        # TODO: delete if cross entropy performs better
        self._rm_goal_trace_success = 0
        self._rm_incomplete_trace_success = 0
        self._rm_dend_trace_success = 0

        self._rm_cross_entropy_sum = 0

        # TODO: We might want to make this more efficient (if there are repeated traces)
        # self._seen_traces: List[TraceTracker] = []

        self._seen_positive_traces: List[TraceTracker] = []
        self._seen_negative_traces: List[TraceTracker] = []
        self._seen_incomplete_traces: List[TraceTracker] = []

        self._curr_ilasp_solution_filename = None

        self.edge_cost = edge_cost
        self.n_phi_cost = n_phi_cost
        self.ex_penalty_multipler = ex_penalty_multiplier
        # self.min_penalty = min_penalty

    # We assume this function be called when a trace is fully generated
    def learn(self, curr_rm: RewardMachine, curr_state, trace: TraceTracker, terminated, truncated,
              is_positive_trace):
        assert isinstance(curr_state, np.ndarray)

        if trace.is_complete:
            if trace.is_positive:
                self._seen_positive_traces.append(copy.deepcopy(trace))
            else:
                self._seen_negative_traces.append(copy.deepcopy(trace))
        else:
            self._seen_incomplete_traces.append(copy.deepcopy(trace))

        if not self.use_cross_entropy:
            self._update_trace_counters(curr_rm, curr_state, trace)
        else:
            self._new_update_trace_counters(curr_rm, curr_state, trace)

        if terminated or truncated:
            self._update_examples(trace)
        elif curr_rm.is_state_terminal(curr_state):
            self._update_examples(trace)


        # if not self._should_relearn_rm() and not self.overriden_with_debugger:
        should_relearn = self._new_should_relearn_rm() if self.use_cross_entropy else self._should_relearn_rm()
        if not should_relearn and not self.overriden_with_debugger:
            return None

        candidate_rm = self._update_reward_machine(curr_rm)
        if candidate_rm:
            self._initialize_trace_counters(candidate_rm)
        return candidate_rm

    def _update_examples(self, trace: TraceTracker):
        if not trace:
            return False

        if trace.is_positive and trace.is_complete:
            self.num_pos_ex += 1
        if trace.is_complete and not trace.is_positive:
            self.num_neg_ex += 1

        examples, ex_type = self.ex_generator.create_examples_from(trace)
        self.examples.merge(examples, ex_type)

        # for ex in examples:
        #     # ex.compact_observations()
        #     # Imporant to do before the .add function as it may change the example penalty
        #     #  Using deepcopy in the .add function results in a significant
        #     #  increase in runtime (+50%)
        #     incomplete_examples = ex.generate_incomplete_examples()
        #     if ex.example_type == ISAILASPExample.ExType.GOAL:
        #         self.goal_examples.add(ex)
        #     elif ex.example_type == ISAILASPExample.ExType.DEND:
        #         self.dend_examples.add(ex)
        #     else:
        #         self.inc_examples.add(ex)
        #     for inc_ex in incomplete_examples:
        #         self.inc_examples.add(inc_ex)

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
                # TODO now: delete these log lines if all is well
                # if len(self.goal_examples) > 0 and "u_acc" in candidate_rm.states:
                if "u_acc" in candidate_rm.states:
                    candidate_rm.set_uacc("u_acc")
                # if len(self.dend_examples) > 0 and "u_rej" in candidate_rm.states:
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
            use_state_id_restrictions=False, # True,  # states used need to be used in order
            binary_folder_name=None,
            n_phi_cost=self.n_phi_cost,
            edge_cost=self.edge_cost,
        )

    # TODO: move logic inside a container if this is too slow
    # @property
    # def _observables(self):
    #     # Using dict for reproducibility - converting set into a list does not preserve the ordering
    #     # (only changing the number of episodes changed the list)
    #     ret = dict()
    #     # ret = set()
    #     for ex in itertools.chain(self.goal_examples.as_list(), self.dend_examples.as_list(),
    #                               self.inc_examples.as_list()):
    #         for obs in ex.observable_context:
    #             ret[obs.label] = None
    #     return list(ret.keys())

    def _new_should_relearn_rm(self) -> bool:
        num_seen_traces = len(self._seen_positive_traces) + len(self._seen_incomplete_traces) + len(
            self._seen_negative_traces)
        if num_seen_traces < self.last_relearning_trace_num + self.min_rm_num_episodes:
            return False

        return self._rm_cross_entropy_sum / num_seen_traces > self.cross_entropy_threshold

    def _should_relearn_rm(self) -> bool:
        num_seen_traces = len(self._seen_positive_traces) + len(self._seen_incomplete_traces) + len(
            self._seen_negative_traces)
        if num_seen_traces < self.last_relearning_trace_num + self.min_rm_num_episodes:
            return False

        # positive
        if (len(self._seen_positive_traces) >= 1 and
                self._rm_goal_trace_success / len(self._seen_positive_traces) < self.rm_recognize_threshold):
            return True

        # negative
        if (len(self._seen_negative_traces) >= 1 and
                self._rm_dend_trace_success / len(self._seen_negative_traces) < self.rm_recognize_threshold):
            return True

        # incomplete
        if (len(self._seen_incomplete_traces) >= 1 and
                self._rm_incomplete_trace_success / len(self._seen_incomplete_traces) < self.rm_recognize_threshold):
            return True

        return False

    def _new_update_trace_counters(self, curr_rm, curr_state, trace):
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

        self._rm_cross_entropy_sum += log_loss(pred_vec, true_vec)

    def _update_trace_counters(self, curr_rm, curr_state, trace):
        self._rm_cnt_since_restart += 1

        # TODO: extract method; remove duplication with example generator
        if trace.is_complete:
            if trace.is_positive:
                trace_outcome = ISAILASPExample.ExType.GOAL
            else:
                trace_outcome = ISAILASPExample.ExType.DEND
        else:
            trace_outcome = ISAILASPExample.ExType.INCOMPLETE

        if trace_outcome == ISAILASPExample.ExType.INCOMPLETE:
            self._rm_incomplete_trace_success += 1 - curr_rm.accepting_state_prob(
                curr_state) - curr_rm.rejecting_state_prob(curr_state)
        elif trace_outcome == ISAILASPExample.ExType.GOAL:
            self._rm_goal_trace_success += curr_rm.accepting_state_prob(curr_state)
        else:
            self._rm_dend_trace_success += curr_rm.rejecting_state_prob(curr_state)

    # Replays old traces to the success rate
    def _initialize_trace_counters(self, candidate_rm):
        self._rm_goal_trace_success = 0
        self._rm_incomplete_trace_success = 0
        self._rm_dend_trace_success = 0

        self._rm_goal_trace_success = 0

        transitioner = ProbRMTransitioner(candidate_rm)
        for trace in itertools.chain(self._seen_positive_traces, self._seen_negative_traces,
                                     self._seen_incomplete_traces):
            curr_state = transitioner.get_initial_state()
            for event in trace.trace:
                curr_state = transitioner.get_next_state(curr_state, event)

            if not self.use_cross_entropy:
                self._update_trace_counters(candidate_rm, curr_state, trace)
            else:
                self._new_update_trace_counters(candidate_rm, curr_state, trace)

        # TODO: return after hacky test
        # if self._should_relearn_rm():
        #     raise ValueError("The relarned RM would be immediately be relearned."
        #                      "Check if the threshold is too large or there is a bigger issue.")
