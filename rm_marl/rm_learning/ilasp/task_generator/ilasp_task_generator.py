import os
from typing import List

from rm_marl.rm_learning.ilasp.ilasp_example_representation import ISAILASPExample
from .utils.ilasp_task_generator_example import generate_examples
from .utils.ilasp_task_generator_hypothesis import get_hypothesis_space
from .utils.ilasp_task_generator_state import generate_state_statements, generate_state_id_statements, \
    generate_state_id_restrictions
from .utils.ilasp_task_generator_symmetry_breaking import generate_symmetry_breaking_statements
from .utils.ilasp_task_generator_transition import generate_timestep_statements, generate_state_at_timestep_statements, \
    generate_transition_statements


def generate_ilasp_task(num_states, accepting_state, rejecting_state, observables, goal_examples: List[ISAILASPExample],
                        dend_examples: List[ISAILASPExample],
                        inc_examples: List[ISAILASPExample], output_folder, output_filename, symmetry_breaking_method,
                        max_disj_size,
                        learn_acyclic, use_compressed_traces, avoid_learning_only_negative,
                        prioritize_optimal_solutions, use_state_id_restrictions, binary_folder_name,
                        n_phi_cost=2, edge_cost=2):
    # statements will not be generated for the rejecting state if there are not deadend examples
    if len(dend_examples) == 0:
        rejecting_state = None
    # it is possible to have only negative examples. there should not be an accepting state in that case
    if len(goal_examples) == 0:
        accepting_state = None

    # with open(os.path.join(output_folder, output_filename), 'w') as f:
    with open(output_filename, 'w') as f:
        background = _generate_ilasp_task_background(accepting_state,
                                                     avoid_learning_only_negative, dend_examples,
                                                     goal_examples, inc_examples, learn_acyclic, max_disj_size,
                                                     num_states,
                                                     prioritize_optimal_solutions, rejecting_state,
                                                     use_compressed_traces, use_state_id_restrictions)

        hyp = _generate_ilasp_hypothesis_space(num_states, accepting_state, rejecting_state, observables, goal_examples,
                                               dend_examples, inc_examples, output_folder, symmetry_breaking_method,
                                               max_disj_size, learn_acyclic, use_compressed_traces,
                                               avoid_learning_only_negative,
                                               prioritize_optimal_solutions, binary_folder_name,
                                               n_phi_cost, edge_cost)

        examples = generate_examples(goal_examples, dend_examples, inc_examples)
        f.write(background)
        f.write('\n' + hyp)
        f.write('\n' + examples)
    with open(f"{output_filename}_examples", 'w') as f:
        f.write(background)
        f.write('\n' + examples)


def _generate_ilasp_hypothesis_space(num_states, accepting_state, rejecting_state, observables,
                                     goal_examples: List[ISAILASPExample], dend_examples: List[ISAILASPExample],
                                     inc_examples: List[ISAILASPExample], output_folder, symmetry_breaking_method,
                                     max_disj_size, learn_acyclic,
                                     use_compressed_traces, avoid_learning_only_negative, prioritize_optimal_solutions,
                                     binary_folder_name,
                                     n_phi_cost, edge_cost):
    task = get_hypothesis_space(num_states, accepting_state, rejecting_state, observables, output_folder,
                                symmetry_breaking_method, max_disj_size, learn_acyclic, binary_folder_name,
                                n_phi_cost, edge_cost)

    if symmetry_breaking_method is not None:
        task += generate_symmetry_breaking_statements(num_states, accepting_state, rejecting_state, observables,
                                                      symmetry_breaking_method, max_disj_size, learn_acyclic)

    return task


def _generate_ilasp_task_background(accepting_state, avoid_learning_only_negative, dend_examples, goal_examples,
                                    inc_examples, learn_acyclic, max_disj_size, num_states,
                                    prioritize_optimal_solutions, rejecting_state, use_compressed_traces,
                                    use_state_id_restrictions):
    task = generate_state_statements(num_states, accepting_state, rejecting_state)
    if use_state_id_restrictions:
        task += generate_state_id_statements(num_states)
        task += generate_state_id_restrictions()
    task += generate_timestep_statements(goal_examples, dend_examples, inc_examples)
    task += _generate_edge_indices_facts(max_disj_size)
    task += generate_state_at_timestep_statements(num_states, accepting_state, rejecting_state)
    task += generate_transition_statements(learn_acyclic, use_compressed_traces, avoid_learning_only_negative,
                                           prioritize_optimal_solutions)
    return task


def _generate_edge_indices_facts(max_disj_size):
    return "edge_id(1..%d).\n\n" % max_disj_size
