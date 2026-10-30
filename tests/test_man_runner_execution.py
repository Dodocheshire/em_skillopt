import json
import unittest
from em_skillopt.core.types import EpisodeSpec, Observation, StepResult, LLMResponse, TokenUsage
from em_skillopt.harness.runner import EpisodeRunner, RunnerConfig
from em_skillopt.harness.parsing import ManipulationPlanParser


class Context:
    def build(self, spec, observation, history, skill, *, planner_step):
        return [{"role":"user","content":spec.instruction}]


class Environment:
    max_steps=15
    action_space=[]
    replan_on_invalid=False
    def __init__(self): self.executed=[]
    def reset(self, spec): return Observation(text="environment instruction")
    def step(self, action):
        self.executed.append(action)
        n=len(self.executed)
        return StepResult(Observation(text="Could not create path." if n==1 else "success"),
                          feedback="Could not create path." if n==1 else "success",
                          reward=-1 if n==1 else 1, action_valid=n!=1,
                          done=n==2, success=n==2)


class Model:
    def __init__(self): self.calls=0
    def generate(self, messages, **kwargs):
        self.calls+=1
        return LLMResponse(json.dumps({"executable_plan":[
            {"action":[50,42,17,6,61,36,1]},
            {"action":[50,42,17,6,61,36,0]},
        ]}),usage=TokenUsage(calls=1))


class ManRunnerExecution(unittest.TestCase):
    def test_failed_action_keeps_remaining_plan_and_later_success(self):
        env, model=Environment(),Model()
        result=EpisodeRunner(env,Context(),ManipulationPlanParser(),model,
            RunnerConfig(max_planner_steps=20)).run_episode(
                EpisodeSpec('eb-man:base:0000','eb-man',0,'stale instruction',max_steps=15))
        self.assertEqual(model.calls,1)
        self.assertEqual(len(env.executed),2)
        self.assertEqual(result.invalid_actions,1)
        self.assertTrue(result.success)
        self.assertEqual(result.spec.instruction,'environment instruction')
        self.assertEqual(result.trajectory[0].feedback,'Could not create path.')


if __name__=='__main__': unittest.main()
