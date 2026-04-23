import json
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_CONFIG: Dict[str, Any] = {
    "model": {
        "model_path": "deepseek-ai/DeepSeek-OCR-2",
        "prompt": "<image>\n<|grounding|>Convert the document to markdown.",
    },
    "runtime": {"backend": "vllm"},
    "gpu": {
        "auto_detect": True,
        "device_ids": [],
        "max_workers_per_gpu": 1,
    },
    "input": {"path": ""},
    "output": {
        "root": "./outputs",
        "save_images": True,
        "save_det_markdown": True,
        "save_layout_pdf": False,
    },
    "resume": {
        "enabled": True,
        "require_det_markdown": False,
        "require_layout_pdf": False,
        "require_images_dir": False,
        "min_markdown_bytes": 16,
    },
    "pdf": {"dpi": 144},
    "vllm": {
        "max_model_len": 8192,
        "max_num_seqs": 48,
        "gpu_memory_utilization": 0.9,
        "block_size": 256,
        "swap_space": 0,
        "enforce_eager": False,
        "disable_mm_preprocessor_cache": True,
        "trust_remote_code": True,
    },
    "postprocess": {"skip_repeat": True, "include_page_split": True},
    "preprocess": {"crop_mode": True, "num_workers": 24},
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "YAML config requires PyYAML. Install with: pip install pyyaml"
        ) from exc

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config root must be an object/dict.")
    return data


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Config root must be an object/dict.")
    return data


def load_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        user_config = _load_yaml(path)
    elif suffix == ".json":
        user_config = _load_json(path)
    else:
        raise ValueError("Only .yaml/.yml/.json config files are supported.")

    config = _deep_merge(dict(DEFAULT_CONFIG), user_config)
    _validate_config(config)
    return config


def _validate_config(config: Dict[str, Any]) -> None:
    if config["runtime"]["backend"] != "vllm":
        raise ValueError("Only runtime.backend='vllm' is supported in this project.")
    input_path = str(config["input"]["path"]).strip()
    if not input_path:
        raise ValueError("input.path is required.")

    model_path = str(config["model"]["model_path"]).strip()
    if not model_path:
        raise ValueError("model.model_path is required.")

    prompt = str(config["model"]["prompt"])
    if "<image>" not in prompt:
        raise ValueError("model.prompt must include exactly one <image> token.")
    if prompt.count("<image>") != 1:
        raise ValueError("model.prompt must contain only one <image> token.")

    device_ids = config["gpu"]["device_ids"]
    if not isinstance(device_ids, list):
        raise ValueError("gpu.device_ids must be a list of integer GPU ids.")
    if any(not isinstance(x, int) for x in device_ids):
        raise ValueError("gpu.device_ids must contain only integers.")

    max_workers_per_gpu = int(config["gpu"]["max_workers_per_gpu"])
    if max_workers_per_gpu < 1:
        raise ValueError("gpu.max_workers_per_gpu must be >= 1.")

    dpi = int(config["pdf"]["dpi"])
    if dpi < 72:
        raise ValueError("pdf.dpi should be >= 72.")

    min_markdown_bytes = int(config["resume"]["min_markdown_bytes"])
    if min_markdown_bytes < 1:
        raise ValueError("resume.min_markdown_bytes must be >= 1.")


def normalize_paths(config: Dict[str, Any], workspace_root: Path) -> Dict[str, Any]:
    _ = workspace_root
    updated = dict(config)
    updated["input"] = dict(config["input"])
    updated["output"] = dict(config["output"])

    updated["input"]["path"] = str(Path(config["input"]["path"]).expanduser().resolve())
    updated["output"]["root"] = str(Path(config["output"]["root"]).expanduser().resolve())

    return updated


def as_gpu_ids(config: Dict[str, Any]) -> List[int]:
    return [int(x) for x in config["gpu"]["device_ids"]]
