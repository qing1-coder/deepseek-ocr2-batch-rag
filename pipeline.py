import io
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import fitz
import img2pdf
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


EOS_TOKEN = "<｜end▁of▁sentence｜>"
PAGE_SPLIT = "\n<--- Page Split --->\n"


def pdf_to_images_high_quality(pdf_path: Path, dpi: int = 144) -> List[Image.Image]:
    images: List[Image.Image] = []
    pdf_document = fitz.open(str(pdf_path))
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        Image.MAX_IMAGE_PIXELS = None
        img_data = pixmap.tobytes("png")
        images.append(Image.open(io.BytesIO(img_data)).convert("RGB"))

    pdf_document.close()
    return images


def pil_to_pdf_img2pdf(pil_images: List[Image.Image], output_path: Path) -> None:
    if not pil_images:
        return

    image_bytes_list = []
    for img in pil_images:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_buffer = io.BytesIO()
        img.save(img_buffer, format="JPEG", quality=95)
        image_bytes_list.append(img_buffer.getvalue())

    pdf_bytes = img2pdf.convert(image_bytes_list)
    with output_path.open("wb") as f:
        f.write(pdf_bytes)


def re_match(text: str) -> Tuple[list, list, list]:
    pattern = r"(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)"
    matches = re.findall(pattern, text, re.DOTALL)
    matches_image = [m[0] for m in matches if "<|ref|>image<|/ref|>" in m[0]]
    matches_other = [m[0] for m in matches if "<|ref|>image<|/ref|>" not in m[0]]
    return matches, matches_image, matches_other


def extract_coordinates_and_label(ref_text: Tuple, image_width: int, image_height: int):
    _ = (image_width, image_height)
    label_type = ref_text[1]
    cor_list = eval(ref_text[2])  # keep behavior aligned with official script format
    return label_type, cor_list


def draw_bounding_boxes(
    image: Image.Image,
    refs: list,
    page_index: int,
    image_dir: Path,
    save_images: bool,
) -> Image.Image:
    image_width, image_height = image.size
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    overlay = Image.new("RGBA", img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    img_idx = 0

    for ref in refs:
        try:
            label_type, points_list = extract_coordinates_and_label(ref, image_width, image_height)
        except Exception:
            continue

        color = (
            int(np.random.randint(0, 200)),
            int(np.random.randint(0, 200)),
            int(np.random.randint(0, 255)),
        )
        color_a = color + (20,)

        for points in points_list:
            x1, y1, x2, y2 = points
            x1 = int(x1 / 999 * image_width)
            y1 = int(y1 / 999 * image_height)
            x2 = int(x2 / 999 * image_width)
            y2 = int(y2 / 999 * image_height)

            if label_type == "image":
                try:
                    if save_images:
                        cropped = image.crop((x1, y1, x2, y2))
                        cropped.save(image_dir / f"{page_index}_{img_idx}.jpg")
                except Exception:
                    pass
                img_idx += 1

            try:
                width = 4 if label_type == "title" else 2
                draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
                draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                text_x = x1
                text_y = max(0, y1 - 15)
                text_bbox = draw.textbbox((0, 0), label_type, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                draw.rectangle(
                    [text_x, text_y, text_x + text_width, text_y + text_height],
                    fill=(255, 255, 255, 30),
                )
                draw.text((text_x, text_y), label_type, font=font, fill=color)
            except Exception:
                pass

    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw


def process_pdf_document(
    llm,
    processor,
    sampling_params,
    pdf_path: Path,
    output_paths,
    config: Dict,
) -> Dict:
    output_paths.doc_dir.mkdir(parents=True, exist_ok=True)
    save_images = bool(config["output"].get("save_images", True))
    save_det_markdown = bool(config["output"].get("save_det_markdown", True))
    save_layout_pdf = bool(config["output"].get("save_layout_pdf", False))
    if save_images:
        output_paths.image_dir.mkdir(parents=True, exist_ok=True)

    images = pdf_to_images_high_quality(pdf_path, dpi=int(config["pdf"]["dpi"]))
    prompt = config["model"]["prompt"]
    crop_mode = bool(config["preprocess"]["crop_mode"])
    num_workers = int(config["preprocess"]["num_workers"])

    def _build_input(image: Image.Image) -> Dict:
        return {
            "prompt": prompt,
            "multi_modal_data": {
                "image": processor.tokenize_with_images(images=[image], bos=True, eos=True, cropping=crop_mode)
            },
        }

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        batch_inputs = list(
            tqdm(
                executor.map(_build_input, images),
                total=len(images),
                desc=f"Preprocess {pdf_path.name}",
                leave=False,
            )
        )

    outputs_list = llm.generate(batch_inputs, sampling_params=sampling_params)

    contents_det = ""
    contents = ""
    draw_images: List[Image.Image] = []
    accepted_pages = 0
    skip_repeat = bool(config["postprocess"]["skip_repeat"])
    include_page_split = bool(config["postprocess"].get("include_page_split", True))
    page_split_text = PAGE_SPLIT if include_page_split else "\n"

    for page_index, (output, image) in enumerate(zip(outputs_list, images)):
        content = output.outputs[0].text
        if EOS_TOKEN in content:
            content = content.replace(EOS_TOKEN, "")
        elif skip_repeat:
            continue

        contents_det += content + page_split_text
        matches_ref, matches_images, matches_other = re_match(content)
        if save_images or save_layout_pdf:
            result_image = draw_bounding_boxes(
                image=image.copy(),
                refs=matches_ref,
                page_index=page_index,
                image_dir=output_paths.image_dir,
                save_images=save_images,
            )
            if save_layout_pdf:
                draw_images.append(result_image)

        for idx, item in enumerate(matches_images):
            if save_images:
                content = content.replace(item, f"![](images/{page_index}_{idx}.jpg)\n")
            else:
                content = content.replace(item, "")
        for item in matches_other:
            content = (
                content.replace(item, "")
                .replace("\\coloneqq", ":=")
                .replace("\\eqqcolon", "=:")
                .replace("\n\n\n\n", "\n\n")
                .replace("\n\n\n", "\n\n")
            )
        contents += content + page_split_text
        accepted_pages += 1

    if save_det_markdown:
        output_paths.det_markdown_path.write_text(contents_det, encoding="utf-8")
    output_paths.markdown_path.write_text(contents, encoding="utf-8")
    if save_layout_pdf:
        pil_to_pdf_img2pdf(draw_images, output_paths.layout_pdf_path)

    return {
        "pdf_path": str(pdf_path),
        "pages_total": len(images),
        "pages_output": accepted_pages,
        "markdown": str(output_paths.markdown_path),
    }
