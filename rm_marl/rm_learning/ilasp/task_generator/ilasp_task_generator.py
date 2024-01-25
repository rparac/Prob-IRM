import os
from typing import Optional, List

from rm_marl.rm_learning.ilasp.noisy_learner.ilasp_example_representation import ISAILASPExample
from .utils.ilasp_task_generator_example import generate_examples
from .utils.ilasp_task_generator_hypothesis import get_hypothesis_space
from .utils.ilasp_task_generator_state import generate_state_statements
from .utils.ilasp_task_generator_symmetry_breaking import generate_symmetry_breaking_statements
from .utils.ilasp_task_generator_transition import generate_timestep_statements, generate_state_at_timestep_statements, \
    generate_transition_statements


class ILASPTaskBuilder:
    def __init__(self, output_folder: str, output_filename: str, binary_folder_name: str,
                 symmetry_breaking_method: Optional[str], max_disjunction_size: int, learn_acyclic: bool,
                 use_compressed_traces: bool, avoid_learning_only_negative: bool,
                 prioritize_optimal_solutions: bool):
        self.output_folder = output_folder
        self.output_filename = output_filename
        self.binary_folder_name = binary_folder_name
        self.symmetry_breaking_method = symmetry_breaking_method
        self.max_disjunction_size = max_disjunction_size
        self.learn_acyclic = learn_acyclic
        self.use_compressed_traces = use_compressed_traces
        self.avoid_learning_only_negative = avoid_learning_only_negative
        self.prioritize_optimal_solutions = prioritize_optimal_solutions

    def generate_ilasp_task(self, num_states, accepting_state, rejecting_state, observables, goal_examples,
                            dend_examples, inc_examples):
        # statements will not be generated for the rejecting state if there are not deadend examples
        if len(dend_examples) == 0:
            rejecting_state = None
        # it is possible to have only negative examples. there should not be an accepting state in that case
        if len(goal_examples) == 0:
            accepting_state = None

        with open(os.path.join(self.output_folder, self.output_filename), 'w') as f:
            task = self._generate_ilasp_task_str(num_states, accepting_state, rejecting_state, observables,
                                                 goal_examples,
                                                 dend_examples, inc_examples)
            f.write(task)

    def _generate_ilasp_task_str(self, num_states, accepting_state, rejecting_state, observables,
                                 goal_examples: List[ISAILASPExample],
                                 dend_examples: List[ISAILASPExample], inc_examples: List[ISAILASPExample]):

        task = generate_state_statements(num_states, accepting_state, rejecting_state)
        task += generate_timestep_statements(goal_examples, dend_examples, inc_examples)
        task += self._generate_edge_indices_facts(self.max_disjunction_size)
        task += generate_state_at_timestep_statements(num_states, accepting_state, rejecting_state)
        task += generate_transition_statements(self.learn_acyclic, self.use_compressed_traces,
                                               self.avoid_learning_only_negative,
                                               self.prioritize_optimal_solutions)
        task += get_hypothesis_space(num_states, accepting_state, rejecting_state, observables, self.output_folder,
                                     self.symmetry_breaking_method, self.max_disjunction_size, self.learn_acyclic,
                                     self.binary_folder_name)

        if self.symmetry_breaking_method is not None:
            task += generate_symmetry_breaking_statements(num_states, accepting_state, rejecting_state, observables,
                                                          self.symmetry_breaking_method, self.max_disjunction_size,
                                                          self.learn_acyclic)

        task += generate_examples(goal_examples, dend_examples, inc_examples)
        return task

    def _generate_edge_indices_facts(self, max_disj_size):
        return "edge_id(1..%d).\n\n" % max_disj_size

