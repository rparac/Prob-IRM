from typing import List, Dict, Tuple

from rm_marl.rm_learning import RMLearner


class FFNSLLearner(RMLearner):
    def __init__(self, agent_id):
        super().__init__(agent_id)

    # traces look like: [ {'by': val1, 'g': val2}*   ]
    #  examples:
    #     if 'by' >= 0.5:
    #        obs(
    #     else:
    #
    #   example -> []


    def learn(self, traces):
        # generate examples
        pass
