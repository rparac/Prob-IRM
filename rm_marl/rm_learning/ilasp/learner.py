import itertools
import os

from rm_marl.rm_learning.ilasp.ilasp_example_representation import lift_goal_example, lift_inc_example, lift_dend_example
from ..trace_tracker import TraceTracker
from ...reward_machine import RewardMachine
from ...utils.logging import getLogger
from ..learner import RMLearner
from .task_generator import generate_ilasp_task
from .task_parser import parse_ilasp_solutions
from .task_solver import solve_ilasp_task

LOGGER = getLogger(__name__)


class ILASPLearner(RMLearner):
    def __init__(self, agent_id, init_rm_num_states=None, wait_for_pos_only=True):
        super().__init__(agent_id)

        self.init_rm_num_states = init_rm_num_states
        self.rm_num_states = init_rm_num_states
        self.wait_for_pos_only = wait_for_pos_only

        self._previous_positive_examples = None
        self._previous_dend_examples = None
        self._previous_incomplete_examples = None
        self._previous_rm_num_states = None

        self.positive_examples = []
        self.dend_examples = []
        self.incomplete_examples = []

    # Called either when rm or the environment is in terminal state
    # TODO: move is_positive_trace to trace tracker
    def update_rm(self, curr_rm: RewardMachine, curr_state, trace: TraceTracker, terminated, truncated,
              is_positive_trace):
        examples_updated = False
        seq = trace.no_dups_labels_sequence
        if terminated or truncated:
            examples_updated = self._update_examples(
                seq, terminated, is_positive_trace
            )
        elif curr_rm.is_state_terminal(curr_state):
            LOGGER.debug(
                f"State belief is wrong when trying to learn a reward machine")
            examples_updated = self._update_examples(seq, complete=False, positive=False)

        if not examples_updated:
            return None

        return self._update_reward_machine(
            self.observables,
            curr_rm,
            self.process_examples(self.positive_examples),
            self.process_examples(self.dend_examples),
            self.process_examples(self.incomplete_examples),
        )

    def _update_examples(self, trace: tuple, complete: bool, positive: bool):
        if not trace:
            return False

        if complete:
            if positive:
                self.positive_examples.append(trace)
            else:
                self.dend_examples.append(trace)
            for i in range(1, len(trace)):
                self.incomplete_examples.append(trace[:i])
        else:
            self.incomplete_examples.append(trace)

        self.incomplete_examples = [
            e for e in self.incomplete_examples
            if e not in self.positive_examples
        ]

        return True

    def process_examples(self, examples):
        return sorted(set(examples), key=len)

    @property
    def observables(self):
        union = set((l for e in self.incomplete_examples for ls in e for l in ls)).union(
            set((l for e in self.positive_examples for ls in e for l in ls))).union(
            set((l for e in self.dend_examples for ls in e for l in ls))
        )
        return union

    # We assume that the set of examples is strictly increasing. So, length checking is sufficient to check for
    # equality.
    def _have_changed(self, positive_examples, dend_examples, incomplete_examples):
        # if self._previous_positive_examples is None or set(positive_examples) != self._previous_positive_examples:
        if self._previous_positive_examples is None or len(positive_examples) != len(self._previous_positive_examples):
            return True
        if self._previous_dend_examples is None or len(dend_examples) != len(self._previous_dend_examples):
            return True
        if self._previous_incomplete_examples is None or len(incomplete_examples) != len(
                self._previous_incomplete_examples):
            return True
        return False

    def _update_reward_machine(
            self,
            observables,
            rm,
            positive_examples,
            dend_examples,
            incomplete_examples,
            rm_num_states=None,
    ):
        LOGGER.debug(f"[{self.agent_id}]`_update_reward_machine`")

        if self.wait_for_pos_only:
            if not positive_examples:
                LOGGER.debug(f"[{self.agent_id}] No positive examples")
                return
        else:
            if not positive_examples and not dend_examples:
                LOGGER.debug(f"[{self.agent_id}] No positive and no deadend examples")
                return

        rm_num_states = (rm_num_states or self.rm_num_states or min(
            len(t) for t in itertools.chain(positive_examples, dend_examples)) + 2)
        # Keep track of the number of states used.
        # Otherwise, iterative deepening would be rerun for every state
        self.rm_num_states = rm_num_states

        if (not self._have_changed(positive_examples, dend_examples, incomplete_examples)
                and rm_num_states == self._previous_rm_num_states):
            LOGGER.debug(f"[{self.agent_id}] Examples haven't changed")
            return
        else:
            self._previous_positive_examples = set(positive_examples)
            self._previous_dend_examples = set(dend_examples)
            self._previous_incomplete_examples = set(incomplete_examples)
            self._previous_rm_num_states = rm_num_states

        LOGGER.debug(f"[{self.agent_id}] num_state: {rm_num_states}")

        return self._run_and_parse_ilasp(observables, rm, positive_examples, dend_examples, incomplete_examples, rm_num_states)

    def _generate_ilasp_task(
            self,
            observables,
            positive_examples,
            dend_examples,
            incomplete_examples,
            rm_num_states,
    ):
        ilasp_task_filename = f"task_{self.rm_learning_counter}"

        # the sets of examples are sorted to make sure that ILASP produces the same solution for the same sets (ILASP
        # can produce different hypothesis for the same set of examples but given in different order)
        generate_ilasp_task(
            rm_num_states,
            "u_acc",
            "u_rej",
            observables,
            [lift_goal_example(ex, f"ex_goal_{i}") for i, ex in enumerate(sorted(positive_examples))],
            [lift_dend_example(ex, f"ex_dend_{i}") for i, ex in enumerate(sorted(dend_examples))],
            [lift_inc_example(ex, f"ex_inc_{i}") for i, ex in enumerate(sorted(incomplete_examples))],
            self.log_folder,
            ilasp_task_filename,
            "bfs-alternative",  # symmetry_breaking_method
            1,  # max_disjunction_size
            True,  # learn_acyclic_graph
            True,  # use_compressed_traces
            True,  # avoid_learning_only_negative
            False,  # prioritize_optimal_solutions
            False,  # use_state_id_restrictions - not needed in an iterative procedure
            None,  # bin directory (ILASP is on PATH)
        )

    def _solve_ilasp_task(self):
        automaton_task_folder = self.log_folder
        ilasp_task_filename = os.path.join(
            automaton_task_folder, f"task_{self.rm_learning_counter}"
        )

        ilasp_solution_filename = os.path.join(
            automaton_task_folder, f"solution_{self.rm_learning_counter}"
        )

        return solve_ilasp_task(
            ilasp_task_filename,
            ilasp_solution_filename,
            timeout=3600,
            version="2",
            max_body_literals=1,
            binary_folder_name=None,
            compute_minimal=True,
        )

    def _run_and_parse_ilasp(self, observables, rm, positive_examples, dend_examples, incomplete_examples,
                             rm_num_states):
        self.rm_learning_counter += 1

        LOGGER.debug(
            f"[{self.agent_id}] generating task {self.rm_learning_counter}: start"
        )
        self._generate_ilasp_task(
            observables,
            positive_examples,
            dend_examples,
            incomplete_examples,
            rm_num_states,
        )
        LOGGER.debug(
            f"[{self.agent_id}] generating task {self.rm_learning_counter}: done"
        )
        LOGGER.debug(
            f"[{self.agent_id}] solving task {self.rm_learning_counter}: start"
        )
        solver_success = self._solve_ilasp_task()
        LOGGER.debug(f"[{self.agent_id}] solving task {self.rm_learning_counter}: done")
        if solver_success:
            ilasp_solution_filename = os.path.join(
                self.log_folder, f"solution_{self.rm_learning_counter}"
            )
            candidate_rm = parse_ilasp_solutions(ilasp_solution_filename)

            if candidate_rm.states:
                candidate_rm.set_u0("u0")
                if positive_examples:
                    candidate_rm.set_uacc("u_acc")
                if dend_examples:
                    candidate_rm.set_urej("u_rej")

                if candidate_rm != rm:
                    rm_plot_filename = os.path.join(
                        self.log_folder, f"plot_{self.rm_learning_counter}"
                    )
                    candidate_rm.plot(rm_plot_filename)
                    return candidate_rm
            else:
                LOGGER.debug(f"[{self.agent_id}] ILASP task unsolvable")
                if self.init_rm_num_states:
                    self.rm_num_states += 1
                return self._update_reward_machine(
                    observables,
                    rm,
                    positive_examples,
                    dend_examples,
                    incomplete_examples,
                    rm_num_states=(rm_num_states + 1) or self.rm_num_states,
                )
        else:
            raise RuntimeError(
                "Error: Couldn't find an automaton within the specified timeout!"
            )
