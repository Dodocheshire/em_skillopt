from core.types import EpisodeSpec, Observation, StepResult


class EnvironmentAdaptor(Protocol):
    action_space: list
    max_steps: int
    replan_on_invalid: bool

    def reset(self, spec: EpisodeSpec) -> Observation: ...
    def step(self, action) -> StepResult: ...
    def save_image(self, obs, out_dir) -> list[str]: ...
    def close(self) -> None: ...
    

