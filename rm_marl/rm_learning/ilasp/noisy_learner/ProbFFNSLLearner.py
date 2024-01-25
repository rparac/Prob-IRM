# import itertools
# from collections import Counter
# from typing import List
#
# from rm_marl.rm_learning import RMLearner
# from rm_marl.rm_learning.ilasp.noisy_learner.example_generator import NoisyILASPExampleGenerator
# from rm_marl.rm_learning.ilasp.noisy_learner.ilasp_example_representation import ISAILASPExample, AcceptTerm, \
#     RejectTerm
# from rm_marl.rm_learning.ilasp.task_generator.new_ilasp_task_generator import new_generate_ilasp_task
# from rm_marl.rm_learning.trace_tracker import TraceTracker, NoisyTraceTracker
# from rm_marl.utils.logging import getLogger
#
# LOGGER = getLogger(__name__)
#
#
# class ProbFFNSLLearner(RMLearner):
#     def __init__(self, agent_id):
#         super().__init__(agent_id)
#
#         self.rm_num_states = 1
#
#         self.examples = []
#         self.ex_generator = NoisyILASPExampleGenerator()
#
#         self.rm_num_states = 1
#
#     def update_rm(self, observables, rm, trace):
#         # We assume this function be called when a trace is fully generated
#         # TODO: check if this is reasonable
#         assert trace.is_complete
#
#         self._update_examples(trace)
#         # examples are always updated
#
#         # TODO: implement condition for checking
#         if True:
#             self._update_reward_machine()
#
#     # TODO: keep track of RM relearning (need to implement relearning condition)
#     def learn(
#             self, observables, rm, positive_examples, dend_examples, incomplete_examples,
#     ):
#         raise NotImplementedError("Should not use this method. It is deprecated.")
#
#     def _update_examples(self, trace: NoisyTraceTracker):
#         if not trace:
#             return False
#
#         examples = self.ex_generator.create_examples_from(trace)
#         for ex in examples:
#             ex.compact_observations()
#             self._add_example(self.examples, ex)
#             for inc_ex in ex.generate_incomplete_examples():
#                 self._add_example(self.examples, inc_ex)
#
#     def _add_example(self, curr_examples: List[ISAILASPExample], new_example: ISAILASPExample) -> List[ISAILASPExample]:
#         ex_updated = False
#         for ex in curr_examples:
#             if ex == new_example:
#                 ex_updated = True
#                 ex.penalty += new_example.penalty
#         if not ex_updated:
#             curr_examples.append(new_example)
#         return curr_examples
#
#     def _update_reward_machine(self):
#         self.rm_learning_counter += 1
#
#         new_generate_ilasp_task(self.examples)
