from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch


def detect_available_gpu_ids() -> List[int]:
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


def resolve_gpu_ids(config_gpu: Dict) -> List[int]:
    requested = [int(x) for x in config_gpu.get("device_ids", [])]
    available = detect_available_gpu_ids()
    if config_gpu.get("auto_detect", True):
        if requested:
            active = [gpu for gpu in requested if gpu in available]
        else:
            active = available
    else:
        active = requested
    if not active:
        raise RuntimeError(
            "No usable GPU selected. Check gpu.auto_detect / gpu.device_ids and CUDA runtime."
        )
    return sorted(set(active))


def assign_tasks_round_robin(tasks: List[Path], gpu_ids: List[int]) -> Dict[int, List[Path]]:
    assignments: Dict[int, List[Path]] = defaultdict(list)
    for idx, task in enumerate(tasks):
        gpu = gpu_ids[idx % len(gpu_ids)]
        assignments[gpu].append(task)
    return dict(assignments)


def build_gpu_slots(gpu_ids: List[int], max_workers_per_gpu: int) -> List[Tuple[int, int]]:
    slots: List[Tuple[int, int]] = []
    for gpu_id in gpu_ids:
        for slot_idx in range(max_workers_per_gpu):
            slots.append((gpu_id, slot_idx))
    return slots


def assign_tasks_to_slots(tasks: List[Path], slots: List[Tuple[int, int]]) -> Dict[Tuple[int, int], List[Path]]:
    assignments: Dict[Tuple[int, int], List[Path]] = defaultdict(list)
    for idx, task in enumerate(tasks):
        slot = slots[idx % len(slots)]
        assignments[slot].append(task)
    return dict(assignments)
