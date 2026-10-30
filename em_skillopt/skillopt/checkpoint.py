"""Recover an interrupted MAN gate without generating another candidate."""
import json
from pathlib import Path
from .rollout import _skill_hash
from .update import apply_patches


def recover_candidate(step_dir, current_skill, max_chars):
    root = Path(step_dir)
    required = ['candidate.md', 'merged_patch.json', 'selected_patch.json',
                'rollout/rollout_config.json']
    if not all((root/name).is_file() for name in required):
        raise RuntimeError('Incomplete MAN candidate checkpoint; refusing regeneration')
    config = json.loads((root/'rollout/rollout_config.json').read_text())
    if config.get('skill_sha256') != _skill_hash(current_skill):
        raise RuntimeError('Candidate checkpoint source skill mismatch')
    merged = json.loads((root/'merged_patch.json').read_text())
    selected = json.loads((root/'selected_patch.json').read_text())
    update = apply_patches(current_skill, [selected], max_chars=max_chars)
    if update.content.strip() != (root/'candidate.md').read_text().strip():
        raise RuntimeError('Saved MAN candidate differs from deterministic patch application')
    return merged, selected, update
