import os
from typing import TYPE_CHECKING, Optional

from ..algo import QRM
from ..reward_machine import RewardMachine
from itertools import groupby
from .rm_agent import RewardMachineAgent
from ..ilasp import generate_ilasp_task, parse_ilasp_solutions, solve_ilasp_task

if TYPE_CHECKING:
    from ..algo import Algo
    from ..reward_machine import RewardMachine

class TraceTracker:

    def __init__(self) -> None:
        self.trace = []

    def reset(self):
        self.trace.clear()

    def update(self, labels):
        self.trace.append(labels or [])

    @property
    def flatten_trace(self):
        return tuple(e for es in self.trace for e in es)

    @property
    def nodups_trace(self):
        return tuple(i[0] for i in groupby(self.flatten_trace or []))


class RewardMachineLearningAgent(RewardMachineAgent):

    def __init__(
        self, algo_cls: "Algo" = QRM, algo_kws: dict = None
    ):
        self.trace = TraceTracker()
        self.incomplete_examples = set()
        self.positive_examples = set()
        self.negative_examples = set()
        
        self.rm_learning_counter = 0

        rm = self._default_rm()

        super().__init__(rm, algo_cls, algo_kws)

    @staticmethod
    def _default_rm():
        rm = RewardMachine()
        rm.add_states(["u0"])
        rm.set_u0("u0")
        return rm

    @property
    def observables(self):
        union = self.incomplete_examples.union(self.positive_examples).union(self.negative_examples)
        return {l for t in union for l in t}

    def reset(self, seed: Optional[int] = None):
        self.trace.reset()
        return super().reset(seed)

    def update_agent(self, state, action, reward, terminated, truncated, next_state, labels, learning=True):

        ret = super().update_agent(state, action, reward, terminated, truncated, next_state, labels, learning)

        if learning:
            self.trace.update(labels)
            if terminated or truncated:
                examples_updated = self._update_examples(self.trace.nodups_trace, terminated)
                if examples_updated:
                    self._update_reward_machine()

        return ret

    def project_labels(self, labels):
        return tuple(labels)

    def _update_examples(self, trace: tuple, complete: bool):
        updated = False

        if complete:
            if trace and trace not in self.positive_examples:
                self.positive_examples.add(trace)
                updated = True
                for i in range(len(trace) - 1):
                    pre = trace[: i + 1]
                    if pre not in self.positive_examples:
                        self.incomplete_examples.add(pre)
                    post = trace[-i - 1 :]
                    if post not in self.positive_examples:
                        self.incomplete_examples.add(post)
        else:
            if trace and trace not in self.incomplete_examples:
                self.incomplete_examples.add(trace)
                updated = True

        _ = [self.incomplete_examples.discard(e) for e in self.positive_examples]

        return updated

    def _update_reward_machine(self):
        self.rm_num_states = min(len(t) for t in self.positive_examples) + 2

        self.rm_learning_counter += 1

        self._generate_ilasp_task()
        solver_success = self._solve_ilasp_task()
        if solver_success:
            ilasp_solution_filename = os.path.join(
                self.log_folder, f"solution_{self.rm_learning_counter}"
            )
            candidate_rm = parse_ilasp_solutions(ilasp_solution_filename)

            if candidate_rm.states:
                candidate_rm.set_u0("u0")
                candidate_rm.set_uacc("u_acc")

                if candidate_rm != self.rm:
                    rm_plot_filename = os.path.join(
                        self.log_folder, f"plot_{self.rm_learning_counter}"
                    )
                    candidate_rm.plot(rm_plot_filename)
                    self.rm = candidate_rm
                    self.algo.reset()
            else:
                # unsatisfiable
                self.rm_num_states += 1
                self._update_reward_machine()
        else:
            raise RuntimeError(
                "Error: Couldn't find an automaton within the specified timeout!"
            )

    def _generate_ilasp_task(self):
        ilasp_task_filename = f"task_{self.rm_learning_counter}"

        # the sets of examples are sorted to make sure that ILASP produces the same solution for the same sets (ILASP
        # can produce different hypothesis for the same set of examples but given in different order)
        generate_ilasp_task(
            self.rm_num_states,
            "u_acc",
            "u_rej",
            self.observables,
            sorted(self.positive_examples),
            sorted(self.negative_examples),
            sorted(self.incomplete_examples),
            self.log_folder,
            ilasp_task_filename,
            "bfs-alternative",  # symmetry_breaking_method
            1,  # max_disjunction_size
            False,  # learn_acyclic_graph
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
            compute_minimal=False,
        )
