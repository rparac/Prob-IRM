from rm_marl.agent import RewardMachineAgent
from rm_marl.new_stack.learner.NewProbFFNSLLearner import NewProbFFNSLLearner

rm = RewardMachineAgent.default_rm()
a = NewProbFFNSLLearner.remote(rm)
b = NewProbFFNSLLearner.remote(rm)

print(a)
print(b)
