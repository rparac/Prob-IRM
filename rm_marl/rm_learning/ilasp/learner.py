import os

from ...utils.logging import getLogger
from ..learner import RMLearner
from .task_generator import generate_ilasp_task
from .task_parser import parse_ilasp_solutions
from .task_solver import solve_ilasp_task

LOGGER = getLogger(__name__)


class ILASPLearner(RMLearner):
    def __init__(self, agent_id, init_rm_num_states = None):
        super().__init__(agent_id)

        self.init_rm_num_states = init_rm_num_states
        self.rm_num_states = init_rm_num_states
        
        self._previous_positive_examples = None
        self._previous_negative_examples = None
        self._previous_incomplete_examples = None

    def learn(
        self, observables, rm, positive_examples, negative_examples, incomplete_examples
    ):
        return self._update_reward_machine(
            observables,
            rm,
            self.process_examples(positive_examples),
            self.process_examples(negative_examples),
            self.process_examples(incomplete_examples),
        )

    def process_examples(self, examples):
        return sorted(set(examples), key=len)

    def _have_changed(self, positive_examples, negative_examples, incomplete_examples):
        if self._previous_positive_examples is None or set(positive_examples) != self._previous_positive_examples:
            return True
        if self._previous_negative_examples is None or set(negative_examples) != self._previous_negative_examples:
            return True
        if self._previous_incomplete_examples is None or set(incomplete_examples) != self._previous_incomplete_examples:
            return True
        return False

    def _update_reward_machine(
        self,
        observables,
        rm,
        positive_examples,
        negative_examples,
        incomplete_examples,
        rm_num_states=None,
    ):
        LOGGER.debug(f"[{self.agent_id}]`_update_reward_machine`")

        if not positive_examples:
            LOGGER.debug(f"[{self.agent_id}] No positive examples")
            return
        
        if not self._have_changed(positive_examples, negative_examples, incomplete_examples):
            LOGGER.debug(f"[{self.agent_id}] Examples haven't changed")
            return
        else:
            self._previous_positive_examples = set(positive_examples)
            self._previous_negative_examples = set(negative_examples)
            self._previous_incomplete_examples = set(incomplete_examples)

        rm_num_states = rm_num_states or self.rm_num_states or min(len(t) for t in positive_examples) + 2
        LOGGER.debug(f"[{self.agent_id}] num_state: {rm_num_states}")

        self.rm_learning_counter += 1

        LOGGER.debug(
            f"[{self.agent_id}] generating task {self.rm_learning_counter}: start"
        )
        self._generate_ilasp_task(
            observables,
            positive_examples,
            negative_examples,
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
                candidate_rm.set_uacc("u_acc")

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
                self._update_reward_machine(
                    observables,
                    rm,
                    positive_examples,
                    negative_examples,
                    incomplete_examples,
                    rm_num_states=self.rm_num_states or (rm_num_states + 1)
                )
        else:
            raise RuntimeError(
                "Error: Couldn't find an automaton within the specified timeout!"
            )

    def _generate_ilasp_task(
        self,
        observables,
        positive_examples,
        negative_examples,
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
            sorted(positive_examples),
            sorted(negative_examples),
            sorted(incomplete_examples),
            self.log_folder,
            ilasp_task_filename,
            "bfs-alternative",  # symmetry_breaking_method
            1,  # max_disjunction_size
            True,  # learn_acyclic_graph
            True,  # use_compressed_traces
            True,  # avoid_learning_only_negative
            False,  # prioritize_optimal_solutions
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
