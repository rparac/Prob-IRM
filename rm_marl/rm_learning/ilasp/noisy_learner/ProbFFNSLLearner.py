import itertools
import os
from typing import List

from rm_marl.reward_machine import RewardMachine
from rm_marl.rm_learning import RMLearner
from rm_marl.rm_learning.ilasp.noisy_learner.example_generator import NoisyILASPExampleGenerator
from rm_marl.rm_learning.ilasp.ilasp_example_representation import ISAILASPExample
from rm_marl.rm_learning.ilasp.task_generator import generate_ilasp_task
from rm_marl.rm_learning.ilasp.task_parser import parse_ilasp_solutions
from rm_marl.rm_learning.ilasp.task_solver import solve_ilasp_task
from rm_marl.rm_learning.trace_tracker import NoisyTraceTracker, TraceTracker
from rm_marl.utils.logging import getLogger

LOGGER = getLogger(__name__)


class ProbFFNSLLearner(RMLearner):
    def __init__(self, agent_id):
        super().__init__(agent_id)

        self.rm_num_states = 1

        self.goal_examples = []
        self.dend_examples = []
        self.inc_examples = []
        self.ex_generator = NoisyILASPExampleGenerator()

        self.rm_num_states = 1

    def learn(self, curr_rm: RewardMachine, curr_state, trace: TraceTracker, terminated, truncated,
              is_positive_trace):
        # We assume this function be called when a trace is fully generated
        # TODO: check if this is reasonable
        if terminated or truncated:
            self._update_examples(trace)
        elif curr_rm.is_state_terminal(curr_state):
            self._update_examples(trace)

        # TODO: implement condition for checking
        if True:
            self._update_reward_machine()

    def _update_examples(self, trace: TraceTracker):
        if not trace:
            return False

        examples = self.ex_generator.create_examples_from(trace)
        for ex in examples:
            ex.compact_observations()
            if ex.ExType == ISAILASPExample.ExType.GOAL:
                self._add_example(self.goal_examples, ex)
            elif ex.ExType == ISAILASPExample.ExType.DEND:
                self._add_example(self.dend_examples, ex)
            else:
                self._add_example(self.inc_examples, ex)
            for inc_ex in ex.generate_incomplete_examples():
                self._add_example(self.inc_examples, inc_ex)

    def _add_example(self, curr_examples: List[ISAILASPExample], new_example: ISAILASPExample) -> List[ISAILASPExample]:
        ex_updated = False
        for ex in curr_examples:
            if ex == new_example:
                ex_updated = True
                ex.penalty += new_example.penalty
        if not ex_updated:
            curr_examples.append(new_example)
        return curr_examples

    def _update_reward_machine(self):
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
            self.goal_examples,
            self.dend_examples,
            self.inc_examples,
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

    @property
    def _observables(self):
        ret = set()
        for ex in itertools.chain(self.goal_examples, self.dend_examples, self.inc_examples):
            for obs in ex.observable_context:
                ret.add(obs.label)
        return ret
