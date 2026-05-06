import os


# Project-scoped defaults. The worker overrides these via environment variables
# derived from config.yaml so the vendor modules stay aligned with runtime config.
#
# DS_OCR2_MODEL_VERSION selects which official DeepSeek-OCR generation we adapt
# to. It also drives the default tile size:
#   - "v1" (DeepSeek-OCR / model1):  IMAGE_SIZE default 640 (Gundam preset)
#   - "v2" (DeepSeek-OCR2 / model2): IMAGE_SIZE default 768
# BASE_SIZE defaults to 1024 in both cases. Either can be overridden via env.
MODEL_VERSION = os.environ.get("DS_OCR2_MODEL_VERSION", "v2").lower()
_DEFAULT_IMAGE_SIZE = "640" if MODEL_VERSION == "v1" else "768"
_DEFAULT_BASE_SIZE = "1024"

BASE_SIZE = int(os.environ.get("DS_OCR2_BASE_SIZE", _DEFAULT_BASE_SIZE))
IMAGE_SIZE = int(os.environ.get("DS_OCR2_IMAGE_SIZE", _DEFAULT_IMAGE_SIZE))
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
