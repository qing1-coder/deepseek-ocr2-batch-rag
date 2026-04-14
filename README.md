# deepseek-ocr2-batch-rag

An independent, non-invasive batch extension for DeepSeek-OCR2 focused on RAG data preprocessing.

This project converts PDFs into Markdown (with optional extracted image assets and layout previews) using only this repository's workflow code.

## Positioning

- Non-invasive extension based on DeepSeek-OCR2 (not an official DeepSeek release).
- Built for RAG preprocessing pipelines: `PDF -> Markdown (+ optional assets)`.
- Supports recursive PDF discovery and mirrored output directory structure.
- Supports multi-GPU data parallel processing (one worker process per GPU slot).
- Supports resume mode to skip already processed files.

## Relationship with DeepSeek-OCR2

This repository does not ship official DeepSeek-OCR2 source code.
You only need to prepare the OCR model weights (for example from HuggingFace or ModelScope).
No local clone/path of the official DeepSeek-OCR2 repository is required.

References:

- DeepSeek-OCR2: [https://github.com/deepseek-ai/DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR)
- Model: [https://huggingface.co/deepseek-ai/DeepSeek-OCR-2](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2)

## Install

```bash
pip install -e .
```

## Quick Start

1. Copy and edit config:

```bash
cp config.example.yaml config.yaml
```

2. Run directly from project root:

```bash
python main.py --config config.yaml
```

## Minimal Config Keys

- `model.model_path`
- `input.path`
- `output.root`

For full options, see `config.example.yaml`.

## Notes on Dependencies

- This extension depends on `vllm`, `transformers`, and PDF/image processing libraries.
- PyYAML is required when using `.yaml`/`.yml` config files.
- Prepare model weights separately (HuggingFace / ModelScope).

## License

This project is licensed under MIT. See `LICENSE`.

## Disclaimer

This project is an extension for DeepSeek-OCR2, not an official DeepSeek release.
