import argparse
import json
import multiprocessing as mp
import queue
import time
from pathlib import Path
from typing import Dict, List

from config_loader import load_config, normalize_paths
from gpu_manager import assign_tasks_to_slots, build_gpu_slots, resolve_gpu_ids
from pathing import build_output_paths
from scanner import collect_pdf_files


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DeepSeek-OCR2 non-invasive PDF batch runner")
    parser.add_argument("--config", required=True, help="Path to config.yaml/yml/json")
    return parser


def _compute_input_root(input_path: str, pdf_files: List[Path]) -> Path:
    source = Path(input_path).resolve()
    if source.is_file():
        return source
    if source.is_dir():
        return source
    return pdf_files[0].parent.resolve()


def _write_failures(output_root: Path, results: List[Dict]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    failures_path = output_root / "failed_tasks.jsonl"
    with failures_path.open("w", encoding="utf-8") as f:
        for result in results:
            for item in result.get("failures", []):
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _is_already_processed(pdf_path: Path, input_root: Path, output_root: Path, resume_cfg: Dict) -> bool:
    paths = build_output_paths(pdf_path=pdf_path, input_root=input_root, output_root=output_root)
    if not paths.markdown_path.exists():
        return False
    if paths.markdown_path.stat().st_size < int(resume_cfg["min_markdown_bytes"]):
        return False
    if bool(resume_cfg["require_det_markdown"]) and not paths.det_markdown_path.exists():
        return False
    if bool(resume_cfg["require_layout_pdf"]) and not paths.layout_pdf_path.exists():
        return False
    if bool(resume_cfg["require_images_dir"]):
        if (not paths.image_dir.exists()) or (not any(paths.image_dir.iterdir())):
            return False
    return True


def run(config_path: str) -> None:
    # Lazy import avoids requiring heavy runtime deps for simple CLI actions
    # such as `python main.py --help`.
    from worker import worker_entry

    workspace_root = Path(__file__).resolve().parent
    config = normalize_paths(load_config(config_path), workspace_root=workspace_root)

    pdf_files = collect_pdf_files(config["input"]["path"])
    if not pdf_files:
        print("No PDF files found under input path.")
        return

    input_root = _compute_input_root(config["input"]["path"], pdf_files)
    output_root = Path(config["output"]["root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if bool(config["resume"]["enabled"]):
        pending_files = []
        skipped_files = []
        for pdf_path in pdf_files:
            if _is_already_processed(pdf_path, input_root, output_root, config["resume"]):
                skipped_files.append(pdf_path)
            else:
                pending_files.append(pdf_path)
        pdf_files = pending_files
        print(f"Resume mode: ON | skipped={len(skipped_files)} pending={len(pdf_files)}")
        if not pdf_files:
            print("All PDFs are already processed. Nothing to do.")
            return

    gpu_ids = resolve_gpu_ids(config["gpu"])
    max_workers_per_gpu = int(config["gpu"]["max_workers_per_gpu"])
    slots = build_gpu_slots(gpu_ids, max_workers_per_gpu=max_workers_per_gpu)
    assignment = assign_tasks_to_slots(pdf_files, slots)

    print(f"Detected GPUs: {gpu_ids}")
    print(f"Total PDFs: {len(pdf_files)}")
    print(f"Output root: {output_root}")
    print(f"GPU workers: {len(slots)} ({max_workers_per_gpu} per GPU)")

    start = time.time()
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes = []

    for (gpu_id, slot_id), tasks in assignment.items():
        process = ctx.Process(
            target=worker_entry,
            args=(
                gpu_id,
                [str(x) for x in tasks],
                config,
                str(input_root),
                str(output_root),
                result_queue,
            ),
            daemon=False,
        )
        process.start()
        process.name = f"gpu-{gpu_id}-slot-{slot_id}"
        processes.append(process)

    for process in processes:
        process.join()

    results = []
    for _ in range(len(processes)):
        try:
            results.append(result_queue.get(timeout=1.0))
        except queue.Empty:
            break

    reported_gpu_ids = {item.get("gpu_id") for item in results}
    for process in processes:
        if process.exitcode and process.exitcode != 0:
            gpu_id = None
            try:
                gpu_id = int(process.name.split("-")[1])
            except Exception:
                pass
            if gpu_id in reported_gpu_ids:
                continue
            results.append(
                {
                    "gpu_id": gpu_id if gpu_id is not None else -1,
                    "assigned": 0,
                    "success": 0,
                    "failed": 1,
                    "failures": [
                        {
                            "gpu_id": gpu_id if gpu_id is not None else -1,
                            "pdf_path": "",
                            "error": f"Worker process exited unexpectedly (exitcode={process.exitcode})",
                            "traceback": "",
                        }
                    ],
                }
            )

    total_success = sum(x.get("success", 0) for x in results)
    total_failed = sum(x.get("failed", 0) for x in results)
    elapsed = time.time() - start
    _write_failures(output_root, results)

    print("----- Batch Summary -----")
    print(f"Success: {total_success}")
    print(f"Failed: {total_failed}")
    print(f"Elapsed: {elapsed:.2f}s")
    for result in sorted(results, key=lambda x: x["gpu_id"]):
        print(
            f"GPU {result['gpu_id']}: assigned={result['assigned']} "
            f"success={result['success']} failed={result['failed']}"
        )


def main() -> None:
    args = _build_parser().parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
