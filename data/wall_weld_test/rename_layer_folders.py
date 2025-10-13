"""
Rename layer and baselayer folders inside each run's weld_data directory.

- layerXX  -> layer_{id}_{XX}   (id = order after sorting by XX ascending)
- baselayerXX -> baselayer_XX

Example:
    run_1/weld_data/layer0    -> run_1/weld_data/layer_0_0
    run_1/weld_data/layer58   -> run_1/weld_data/layer_1_58
    run_1/weld_data/baselayer7 -> run_1/weld_data/baselayer_7
"""

import os
import re
import uuid
from typing import List, Tuple

# ---------------- Configuration ----------------
# RUN_DIRS = ['2025_06_11_16_27_SSWL0_datacollection/'
#             ]    # <-- put your run dirs here
# find all run dirs in current folder starting with '2025'
RUN_DIRS = [d for d in os.listdir('.') if os.path.isdir(d) and d.startswith('2025')]
print(f'Found {len(RUN_DIRS)} run dirs to process.')
for d in RUN_DIRS:
    print(f'  {d}')
DRY_RUN = False                    # Set to False to actually rename
# ------------------------------------------------

RE_LAYER = re.compile(r'^layer(\d{1,3})$')          # matches layerXX
RE_BASELAYER = re.compile(r'^baselayer(\d{1,3})$')  # matches baselayerXX


def find_matching_dirs(parent: str, pattern: re.Pattern) -> List[Tuple[int, str, str]]:
    """
    Find subdirectories in `parent` whose names match `pattern`.
    Returns list of tuples: (number, name, full_path)
    """
    results = []
    if not os.path.isdir(parent):
        return results

    with os.scandir(parent) as it:
        for entry in it:
            if entry.is_dir():
                m = pattern.match(entry.name)
                if m:
                    num = int(m.group(1))
                    results.append((num, entry.name, entry.path))
    return results


def safe_rename(src: str, dst: str, dry_run: bool = True) -> None:
    """
    Print and optionally perform os.rename.
    """
    if src == dst:
        # Nothing to do
        return
    if not dry_run:
        os.rename(src, dst)
    else:
        print(f'[DRY_RUN] Rename: {src} -> {dst}')


def two_phase_rename(pairs: List[Tuple[str, str]], dry_run: bool = True) -> None:
    """
    Rename with a two-phase approach to avoid name collisions.

    pairs: list of (src, dst) paths
    """
    # Phase 1: rename all src -> temp unique names
    temp_map = {}
    for src, dst in pairs:
        if src == dst:
            continue
        temp = f"{src}.renaming.{uuid.uuid4().hex}"
        temp_map[src] = temp
        safe_rename(src, temp, dry_run=dry_run)

    # Phase 2: rename temp -> final dst
    for src, dst in pairs:
        if src == dst:
            continue
        temp = temp_map[src]
        safe_rename(temp, dst, dry_run=dry_run)


def rename_layers_and_baselayers(run_dir: str, dry_run: bool = True) -> None:
    """
    For one run directory:
      - Sort layerXX by XX ascending and rename to layer_{id}_{XX}
      - Rename baselayerXX to baselayer_XX
    """
    weld_data = os.path.join(run_dir, 'weld_data')
    if not os.path.isdir(weld_data):
        print(f'[SKIP] {weld_data} not found')
        return

    # 1) Handle layerXX -> layer_{id}_{XX}
    layer_dirs = find_matching_dirs(weld_data, RE_LAYER)  # [(XX, name, path), ...]
    layer_dirs.sort(key=lambda x: x[0])  # sort by XX

    layer_pairs = []
    for idx, (xx, name, src_path) in enumerate(layer_dirs):
        dst_name = f'layer_{idx}_{xx}'
        dst_path = os.path.join(weld_data, dst_name)
        layer_pairs.append((src_path, dst_path))

    if layer_pairs:
        # print(f'\n[Run: {run_dir}] Renaming layerXX -> layer_id_XX')
        two_phase_rename(layer_pairs, dry_run=dry_run)
    else:
        print(f'\n[Run: {run_dir}] No layerXX folders found')

    # 2) Handle baselayerXX -> baselayer_XX
    base_dirs = find_matching_dirs(weld_data, RE_BASELAYER)
    base_pairs = []
    for xx, name, src_path in base_dirs:
        dst_name = f'baselayer_{xx}'
        dst_path = os.path.join(weld_data, dst_name)
        base_pairs.append((src_path, dst_path))

    if base_pairs:
        # print(f'[Run: {run_dir}] Renaming baselayerXX -> baselayer_XX')
        two_phase_rename(base_pairs, dry_run=dry_run)
    else:
        print(f'[Run: {run_dir}] No baselayerXX folders found')


def main():
    print(f'DRY_RUN = {DRY_RUN} (set to False to apply changes)\n')
    for rd in RUN_DIRS:
        rename_layers_and_baselayers(rd, dry_run=DRY_RUN)


if __name__ == '__main__':
    main()
