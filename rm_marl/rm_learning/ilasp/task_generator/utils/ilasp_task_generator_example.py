import itertools
from typing import List

from ...ilasp_common import OBS_STR, generate_injected_statement
from ...noisy_learner.ilasp_example_representation import ISAILASPExample


def generate_examples(goal_examples: List[ISAILASPExample], dend_examples: List[ISAILASPExample],
                      inc_examples: List[ISAILASPExample]):
    examples = '\n'.join([str(ex) for ex in itertools.chain(goal_examples, dend_examples, inc_examples)])
    examples += '\n'

    examples += (
            _generate_examples_injection(goal_examples, dend_examples, inc_examples) + "\n"
    )
    return examples


def _generate_examples_injection(goal_examples, neg_examples, inc_examples):
    ret = ""
    for ex in itertools.chain(goal_examples, neg_examples, inc_examples):
        ret += generate_injected_statement(f"example_active({ex.ex_id}).")
        ret += "\n"
    return ret


def get_longest_example_length(goal_examples, neg_examples, inc_examples):
    max_len = 0
    for ex in itertools.chain(goal_examples, neg_examples, inc_examples):
        max_len = max(max_len, ex.last_predicate.time_step)
    return max_len
