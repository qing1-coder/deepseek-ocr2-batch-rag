import os


# Project-scoped defaults. Worker overrides these via environment variables
# derived from config.yaml so vendor modules stay aligned with runtime config.
BASE_SIZE = int(os.environ.get("DS_OCR2_BASE_SIZE", "1024"))
IMAGE_SIZE = int(os.environ.get("DS_OCR2_IMAGE_SIZE", "768"))
CROP_MODE = os.environ.get("DS_OCR2_CROP_MODE", "true").lower() in {"1", "true", "yes"}
MIN_CROPS = int(os.environ.get("DS_OCR2_MIN_CROPS", "2"))
MAX_CROPS = int(os.environ.get("DS_OCR2_MAX_CROPS", "6"))
PRINT_NUM_VIS_TOKENS = os.environ.get("DS_OCR2_PRINT_VIS_TOKENS", "false").lower() in {
    "1",
    "true",
    "yes",
}
PROMPT = os.environ.get(
    "DS_OCR2_PROMPT",
    "<image>\n<|grounding|>Convert the document to markdown.",
)
MODEL_PATH = os.environ.get("DS_OCR2_MODEL_PATH", "deepseek-ai/DeepSeek-OCR-2")
