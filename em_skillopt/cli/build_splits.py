"""data/em_splits/eb-man/
├── train/items.json       # 60%
├── val/items.json         # 20%
├── test/items.json        # 20%
└── split_manifest.json
按 scene / task family / instruction template 分组，不要按 frame 随机分
同一 scene 或同一 task variation 不跨 train/test
固定 seed
记录 episode_id、index、capability、scene、task_type
baseline 和 Method A 共用同一份 test
"""