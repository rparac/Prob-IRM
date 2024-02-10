import copy
import itertools
import os
from typing import List

import numpy as np

from rm_marl.reward_machine import RewardMachine
from rm_marl.rm_learning import RMLearner
from rm_marl.rm_learning.ilasp.ilasp_example_representation import ISAILASPExample, ISAExampleContainer
from rm_marl.rm_learning.ilasp.noisy_learner.example_generator import NoisyILASPExampleGenerator
from rm_marl.rm_learning.ilasp.task_generator import generate_ilasp_task
from rm_marl.rm_learning.ilasp.task_parser import parse_ilasp_solutions
from rm_marl.rm_learning.ilasp.task_solver import solve_ilasp_task
from rm_marl.rm_learning.trace_tracker import TraceTracker
from rm_marl.rm_transition.prob_rm_transitioner import ProbRMTransitioner
from rm_marl.utils.logging import getLogger

LOGGER = getLogger(__name__)


class ProbFFNSLLearner(RMLearner):
    def __init__(self, agent_id):
        super().__init__(agent_id)

        self.goal_examples = ISAExampleContainer()
        self.dend_examples = ISAExampleContainer()
        self.inc_examples = ISAExampleContainer()
        self.ex_generator = NoisyILASPExampleGenerator()

        # Minimum is 3 states (accepting, rejecting, u0)
        # self.rm_num_states = 4
        self.rm_num_states = 5

        # Minimum number of new traces before we validate if the
        # reward machine is the correct one
        self.min_rm_num_episodes = 10

        # The percentage of traces that need to conform to the reward
        # machine to avoid relearning
        self.rm_recognize_threshold = 0.35

        # Debug tracking
        self.num_pos_ex = 0
        self.num_neg_ex = 0
        self.overriden_with_debugger = False

        self._rm_success_trace_cnt = 0
        # TODO: unused at the moment, but could be useful for debugging
        self._rm_cnt_since_restart = 0

        # TODO: We might want to make this more efficient (if there are repeated traces)
        self._seen_traces: List[TraceTracker] = []

    # We assume this function be called when a trace is fully generated
    def learn(self, curr_rm: RewardMachine, curr_state, trace: TraceTracker, terminated, truncated,
              is_positive_trace):
        assert isinstance(curr_state, np.ndarray)

        self._seen_traces.append(copy.deepcopy(trace))
        self._update_trace_counters(curr_rm, curr_state, trace)
        if terminated or truncated:
            self._update_examples(trace)
        elif curr_rm.is_state_terminal(curr_state):
            self._update_examples(trace)

        if not self._should_relearn_rm() and not self.overriden_with_debugger:
            return None

        candidate_rm = self._update_reward_machine(curr_rm)
        self._initialize_trace_counters(candidate_rm)
        return candidate_rm

    def _update_examples(self, trace: TraceTracker):
        if not trace:
            return False

        if trace.is_positive and trace.is_complete:
            self.num_pos_ex += 1
        if trace.is_complete and not trace.is_positive:
            self.num_neg_ex += 1

        examples = self.ex_generator.create_examples_from(trace)
        for ex in examples:
            ex.compact_observations()
            if ex.example_type == ISAILASPExample.ExType.GOAL:
                self.goal_examples.add(ex)
            elif ex.example_type == ISAILASPExample.ExType.DEND:
                self.dend_examples.add(ex)
            else:
                self.inc_examples.add(ex)
            for inc_ex in ex.generate_incomplete_examples():
                self.inc_examples.add(inc_ex)

    def _update_reward_machine(self, curr_rm):
        self.rm_learning_counter += 1

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
                if len(self.goal_examples) > 0 and "u_acc" in candidate_rm.states:
                    candidate_rm.set_uacc("u_acc")
                if len(self.dend_examples) > 0 and "u_rej" in candidate_rm.states:
                    candidate_rm.set_urej("u_rej")

                if candidate_rm == curr_rm:
                    return None
                rm_plot_filename = os.path.join(
                    self.log_folder, f"plot_{self.rm_learning_counter}"
                )
                candidate_rm.plot(rm_plot_filename)
                return candidate_rm
            else:
                LOGGER.debug(f"[{self.agent_id}] ILASP task unsolvable")
                self.rm_num_states += 1
                return self._update_reward_machine(curr_rm)

        else:
            raise RuntimeError(
                "Error: Couldn't find an automaton within the specified timeout!"
            )

    def _solve_ilasp_task(self, ilasp_task_filename, ilasp_solution_filename):
        return solve_ilasp_task(
            ilasp_task_filename,
            ilasp_solution_filename,
            timeout=3600,
            version="2",
            max_body_literals=1,
            binary_folder_name=None,
            compute_minimal=True,
        )

    def _generate_ilasp_task(self, ilasp_task_filename):
        generate_ilasp_task(
            self.rm_num_states,
            "u_acc",
            "u_rej",
            self._observables,
            self.goal_examples.as_list_reweighted(100),
            self.dend_examples.as_list_reweighted(100),
            self.inc_examples.as_list_reweighted(100),
            self.log_folder,
            ilasp_task_filename,
            symmetry_breaking_method="bfs-alternative",
            max_disj_size=1,
            learn_acyclic=True,
            use_compressed_traces=True,
            avoid_learning_only_negative=True,
            prioritize_optimal_solutions=False,
            binary_folder_name=None,
        )

    # TODO: move logic inside a container if this is too slow
    @property
    def _observables(self):
        ret = set()
        for ex in itertools.chain(self.goal_examples.as_list(), self.dend_examples.as_list(),
                                  self.inc_examples.as_list()):
            for obs in ex.observable_context:
                ret.add(obs.label)
        return ret

    def _should_relearn_rm(self) -> bool:
        if len(self._seen_traces) < self.min_rm_num_episodes:
            return False

        correct_threshold = self._rm_success_trace_cnt / len(self._seen_traces)
        return correct_threshold < self.rm_recognize_threshold

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

        # TODO: remove this if not needed
        # state_outcome = ISAILASPExample.ExType.INCOMPLETE
        # if curr_rm.is_accepting_state(curr_state):
        #     state_outcome = ISAILASPExample.ExType.GOAL
        # if curr_rm.is_rejecting_state(curr_state):
        #     state_outcome = ISAILASPExample.ExType.DEND
        #
        # if trace_outcome == state_outcome:
        #     self._rm_success_trace_cnt += 1
        if trace_outcome == ISAILASPExample.ExType.GOAL:
            self._rm_success_trace_cnt += curr_rm.accepting_state_prob(curr_state)
        if trace_outcome == ISAILASPExample.ExType.DEND:
            self._rm_success_trace_cnt += curr_rm.rejecting_state_prob(curr_state)
        if trace_outcome == ISAILASPExample.ExType.INCOMPLETE:
            self._rm_success_trace_cnt \
                += 1 - curr_rm.accepting_state_prob(curr_state) - curr_rm.rejecting_state_prob(curr_state)

    # Replays old traces to the success rate
    def _initialize_trace_counters(self, candidate_rm):
        self._rm_success_trace_cnt = 0

        transitioner = ProbRMTransitioner(candidate_rm)
        for trace in self._seen_traces:
            curr_state = transitioner.get_initial_state()
            for event in trace.trace:
                curr_state = transitioner.get_next_state(curr_state, event)
            self._update_trace_counters(candidate_rm, curr_state, trace)

        if self._should_relearn_rm():
            raise ValueError("The relarned RM would be immediately be relearned."
                             "Check if the threshold is too large or there is a bigger issue.")
