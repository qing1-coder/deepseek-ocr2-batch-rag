import os
import traceback
from pathlib import Path
from typing import Dict, List

from transformers import AutoTokenizer

from pathing import build_output_paths
from pipeline import process_pdf_document


def worker_entry(
    gpu_id: int,
    pdf_paths: List[str],
    config: Dict,
    input_root: str,
    output_root: str,
    result_queue,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["VLLM_USE_V1"] = "0"
    os.environ["DS_OCR2_MODEL_PATH"] = str(config["model"]["model_path"])
    os.environ["DS_OCR2_PROMPT"] = str(config["model"]["prompt"])
    os.environ["DS_OCR2_CROP_MODE"] = str(bool(config["preprocess"]["crop_mode"])).lower()

    try:
        from vendor.deepseek_ocr2 import DeepseekOCR2ForCausalLM  # type: ignore
    except ModuleNotFoundError as exc:
        if getattr(exc, "name", "") == "deepencoderv2":
            raise ModuleNotFoundError(
                "Missing dependency module 'deepencoderv2'. "
                "Please copy required DeepSeek-OCR2 encoder sources into project runtime path."
            ) from exc
        raise

    from vendor.process.ngram_norepeat import NoRepeatNGramLogitsProcessor  # type: ignore
    from vendor.process.image_process import DeepseekOCR2Processor  # type: ignore
    from vllm import LLM, SamplingParams
    from vllm.model_executor.models.registry import ModelRegistry

    # Preload tokenizer to ensure model assets are available before worker inference starts.
    _ = AutoTokenizer.from_pretrained(config["model"]["model_path"], trust_remote_code=True)

    ModelRegistry.register_model("DeepseekOCR2ForCausalLM", DeepseekOCR2ForCausalLM)
    llm = LLM(
        model=config["model"]["model_path"],
        hf_overrides={"architectures": ["DeepseekOCR2ForCausalLM"]},
        block_size=int(config["vllm"]["block_size"]),
        enforce_eager=bool(config["vllm"]["enforce_eager"]),
        trust_remote_code=bool(config["vllm"]["trust_remote_code"]),
        max_model_len=int(config["vllm"]["max_model_len"]),
        swap_space=int(config["vllm"]["swap_space"]),
        max_num_seqs=int(config["vllm"]["max_num_seqs"]),
        tensor_parallel_size=1,
        gpu_memory_utilization=float(config["vllm"]["gpu_memory_utilization"]),
        disable_mm_preprocessor_cache=bool(config["vllm"]["disable_mm_preprocessor_cache"]),
    )

    logits_processors = [
        NoRepeatNGramLogitsProcessor(
            ngram_size=20,
            window_size=50,
            whitelist_token_ids={128821, 128822},
        )
    ]
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=int(config["vllm"]["max_model_len"]),
        logits_processors=logits_processors,
        skip_special_tokens=False,
        include_stop_str_in_output=True,
    )

    processor = DeepseekOCR2Processor()
    input_root_path = Path(input_root).resolve()
    output_root_path = Path(output_root).resolve()
    failures = []
    success_count = 0

    for pdf_str in pdf_paths:
        pdf_path = Path(pdf_str).resolve()
        output_paths = build_output_paths(pdf_path, input_root_path, output_root_path)
        try:
            process_pdf_document(
                llm=llm,
                processor=processor,
                sampling_params=sampling_params,
                pdf_path=pdf_path,
                output_paths=output_paths,
                config=config,
            )
            success_count += 1
        except Exception as exc:
            failures.append(
                {
                    "gpu_id": gpu_id,
                    "pdf_path": str(pdf_path),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

    result_queue.put(
        {
            "gpu_id": gpu_id,
            "assigned": len(pdf_paths),
            "success": success_count,
            "failed": len(failures),
            "failures": failures,
        }
    )
