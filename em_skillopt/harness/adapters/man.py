"""EB-Manipulation adapter.

Wraps ``embodiedbench.envs.eb_manipulation.EBManEnv``.

Key mappings:
- env.reset() -> (description, obs)
- env.step(7D action) -> (obs, reward, terminate, info)
- info["action_success"] -> action_valid
- info["task_success"] -> success
- terminate -> done
"""
from __future__ import annotations

import copy
from typing import Any

from ...core.types import ActionValue, EpisodeSpec, Observation, StepResult
from .base import BaseAdapter


def _as_paths(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return []


class ManAdapter(BaseAdapter):
    def __init__(
        self,
        env: Any,
        *,
        log_path: str = "",
        max_steps: int | None = None,
        camera_views: list[str] | None = None,
        detection_box: bool = False,
        draw_coordinates: bool = True,
        language_only: bool = False,
        resolution: int = 500,
    ) -> None:
        super().__init__(
            env,
            action_space=[],  # 7D actions are not action ids
            max_steps=max_steps,
            # Official MAN evaluator continues the current plan after a failed
            # action unless the environment terminates the episode.
            replan_on_invalid=False,
        )
        self.camera_views = list(camera_views or ["front_rgb"])
        self.detection_box = detection_box
        self.language_only = language_only
        self.draw_coordinates = draw_coordinates
        self.resolution = resolution

        if log_path:
            self.env.log_path = log_path
        if draw_coordinates and resolution not in (300, 500):
            raise ValueError("draw_coordinates only supports resolution 300/500")

    def reset(self, spec: EpisodeSpec) -> Observation:
        self.seek_episode(spec)
        description, obs = self.env.reset()

        text = (
            str(description[0])
            if isinstance(description, (list, tuple)) and description
            else str(description or spec.instruction)
        )
        return self._observe(obs, text)

    def step(self, action: ActionValue) -> StepResult:
        if not isinstance(action, list) or len(action) != 7:
            raise TypeError(
                f"ManAdapter expects a 7-element list, got {action!r}"
            )

        obs, reward, terminate, info = self.env.step(action)

        feedback = str(info.get("env_feedback", ""))
        observation = self._observe(obs, feedback)

        return self.make_step_result(
            observation,
            reward=reward,
            done=terminate,
            success=bool(info.get("task_success", 0)),
            feedback=feedback,
            action_valid=bool(info.get("action_success", 0)),
            info={
                "env_step": info.get("env_step"),
                "action_success": info.get("action_success"),
                "task_success": info.get("task_success"),
                "episode_num": info.get("episode_num"),
            },
        )

    def _observe(self, obs: Any, text: str) -> Observation:
        from embodiedbench.envs.eb_manipulation.eb_man_utils import (
            draw_bounding_boxes,
            draw_xyz_coordinate,
            form_object_coord_for_input,
        )

        images = _as_paths(self.env.save_image(self.camera_views))

        # EBManEnv.step() returns a dict on an execution error and an
        # Observation object on success.  Normalise both before the helper.
        raw_obs = (
            copy.deepcopy(obs)
            if isinstance(obs, dict)
            else vars(copy.deepcopy(obs))
        )
        avg_coord, all_points, extrinsics, intrinsics = (
            form_object_coord_for_input(
                raw_obs,
                self.env.task_class,
                self.camera_views,
            )
        )

        if not self.language_only:
            if self.draw_coordinates:
                for index, path in enumerate(images):
                    if "front_rgb" in path:
                        images[index] = draw_xyz_coordinate(
                            path, self.resolution
                        )
            if self.detection_box:
                images = draw_bounding_boxes(
                    images,
                    all_points,
                    extrinsics,
                    intrinsics,
                )

        return Observation(
            images=[] if self.language_only else images,
            text=text,
            metadata={
                "object_coords": avg_coord,
                "task_variation": self.env.current_task_variation,
            },
        )