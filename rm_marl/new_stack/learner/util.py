import itertools
from typing import List, Set, Tuple
from rm_marl.reward_machine import RewardMachine
from rm_marl.rm_learning.ilasp.ilasp_example_representation import ISAILASPExample, LastPredicate, ObservablePredicate


def generate_previous_incomplete_examples(curr_rm: RewardMachine) -> List[ISAILASPExample]:
    """
    Generate non-terminal incomplete ISAILASP examples by taking all non-empty proper subsets of RM paths and merging duplicates.
    Returns merged examples with unit penalties derived from the provided reward machine.
    """


    rm_paths = find_all_paths(curr_rm, curr_rm.u0, set())

    sol = []
    for rm_path in rm_paths:
        subsets = find_all_subsets(rm_path)
        non_terminal_subsets = [subset for subset in subsets if not is_terminal_path(subset, curr_rm)]
        for subset in non_terminal_subsets:
            obs_predicate_subset = to_obs_predicate_list(subset)
            penalty = 1

            sol.append(
                ISAILASPExample(
                    f"previous_rm_inc_ex_{len(sol)}",
                    penalty,
                    ISAILASPExample.ExType.INCOMPLETE, obs_predicate_subset, 
                    LastPredicate(len(subset) - 1), new_inc_example=True
                )
            )
    sol = merge_identical_examples(sol)
    return sol

def merge_identical_examples(examples: List[ISAILASPExample]) -> List[ISAILASPExample]:
    """
    Merge identical ISAILASP examples (per __eq__) by summing their penalties.
    Returns a new list with duplicates collapsed.
    """
    result = []
    for ex in examples:
        try:
            found_ex_idx = result.index(ex)
            result[found_ex_idx].penalty += ex.penalty
        except ValueError:
            result.append(ex)
    return result


def to_obs_predicate_list(rm_path: List[Tuple[str]]) -> List[ObservablePredicate]:
    """
    Convert an RM path (tuples of labels per time step) into a flat list of ObservablePredicate with time steps.
    Multi-label steps are flattened into multiple predicates sharing the same time step.
    """
    result = []
    for time_step, label in enumerate(rm_path):
        for l in label:
            result.append(ObservablePredicate(l, time_step))
    return result



def is_terminal_path(rm_path: List[Tuple[str]], rm: RewardMachine) -> bool:
    """
    Return True if following rm_path from rm.u0 ends in an accepting or rejecting state; otherwise False.
    """
    curr_state = rm.u0
    for transition in rm_path:
        curr_state = rm.get_next_state(curr_state, transition)

    return rm.is_accepting_state(curr_state) or rm.is_rejecting_state(curr_state)

def find_all_subsets(rm_path: List[Tuple[str]]) -> List[List[Tuple[str]]]:
    """
    Return all non-empty proper subsets of rm_path (combinations of lengths 1..N-1), preserving order.
    """
    sol = []
    comb_length_max = len(rm_path)
    for i in range(1, comb_length_max):
        combs = itertools.combinations(rm_path, i)
        sol.extend(combs)
    return sol
    

def find_all_paths(curr_rm: RewardMachine, curr_state: str, already_visited: Set[str]) -> List[List[Tuple[str]]]:
    """
    Enumerate all label-only paths from curr_state to any terminal, stripping negated literals and avoiding cycles.
    Returns lists of tuples of positive labels per time step; terminals yield `[[]]` for concatenation.
    """
    paths = []

    if curr_rm.is_accepting_state(curr_state) or curr_rm.is_rejecting_state(curr_state):
        return [[]]

    if curr_state in already_visited:
        # Cycle. Just skip it.
        return []

    already_visited.add(curr_state)

    sols = []

    for condition in curr_rm.transitions[curr_state]:
        next_state = curr_rm.transitions[curr_state][condition]
        paths = find_all_paths(curr_rm, next_state, already_visited)
        stripped_condition = tuple(c for c in condition if not c.startswith("~"))
        new_paths = [[stripped_condition] + path for path in paths]
        sols.extend(new_paths)

    already_visited.remove(curr_state)
    return sols