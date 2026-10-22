"""agent loop in an episode"""
from core.types import EpisodeResult

def run_episode(spec, env, context, parser, llm, skill, max_planner_steps):
    obs = env.reset(spec)
    history = []
    for _ in range(max_planner_steps):
        messages = context.build(spec, obs, history, skill)
        response = llm.generate(messages)
        plan = parser.parse(response.text, env.action_space)
        if plan.stop or not plan.actions:
            break
        for action in plan.actions:
            step = env.step(action)
            history.append(...)
            if step.done:
                break
            if not step.action_valid and env.replan_on_invalid:
                break
        if step.done:
            break

    return EpisodeResult(...)