# %%
from datasets import load_dataset, DatasetDict, concatenate_datasets, Value
import string, re
# %%
import re
import json
from PIL import Image, ImageDraw, ImageFont
import textwrap
import os
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets

def create_text_images2(
    text: str,
    answer_char: str,
    max_chars_per_line: int = 50,
    max_lines_per_image: int = 15,
    font_path: str = "arial.ttf",
    font_size: int = 20,
    margin: int = 10,
    make_evidence_data: bool = False,
):
    """
    The pass key is X. 패턴 내의 X 한 글자에 대해서만 bbox를 뽑습니다.
    """
    # 1) 줄 단위로 split + wrap
    raw_lines = text.split("\n")
    wrapped = []
    for line in raw_lines:
        wrapped += textwrap.wrap(line, width=max_chars_per_line)

    # 2) 폰트 및 더미 드로잉 객체
    font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    dummy = Image.new("RGB", (1, 1))
    dummy_draw = ImageDraw.Draw(dummy)

    images = []
    all_bboxes = {}
    chunk_idx = 0

    # 3) chunk 단위 이미지 생성
    for i in range(0, len(wrapped), max_lines_per_image):
        chunk = wrapped[i : i + max_lines_per_image]

        # 이미지 크기 계산
        widths = [dummy_draw.textbbox((0, 0), line, font=font)[2] for line in chunk]
        max_w = max(widths) if widths else 0
        hh = dummy_draw.textbbox((0, 0), "Ay", font=font)[3]
        total_h = hh * len(chunk) + margin * 2

        img = Image.new("RGB", (max_w + margin * 2, total_h), "white")
        draw = ImageDraw.Draw(img)

        y = margin
        for line in chunk:
            draw.text((margin, y), line, font=font, fill="black")

            # 패턴: “The pass key is X.” 내의 X 한 글자만
            pattern = rf"(?<=The pass key is ){re.escape(answer_char)}(?=\.)"
            for m in re.finditer(pattern, line):
                # bbox 좌표 계산
                pre_text = line[: m.start()]
                pb = dummy_draw.textbbox((0, 0), pre_text, font=font)
                x0 = margin + (pb[2] - pb[0])
                cbb = dummy_draw.textbbox((0, 0), answer_char, font=font)
                x1 = x0 + (cbb[2] - cbb[0])
                y0, y1 = y, y + hh

                if make_evidence_data:
                    draw.rectangle((x0, y0, x1, y1), outline="red", width=2)

                all_bboxes.setdefault(str(chunk_idx), []).append([x0, y0, x1, y1])

            y += hh

        images.append(img)
        chunk_idx += 1

    return images, all_bboxes


def create_text_images(
    text: str,
    answer_char: str,
    max_chars_per_line: int = 50,
    max_lines_per_image: int = 15,
    font_path: str = "arial.ttf",
    font_size: int = 20,
    margin: int = 10,
    make_evidence_data: bool = False,
):
    """
    The pass key is X. 패턴 내의 X 한 글자에 대해서만 bbox 리스트를 뽑습니다.
    """
    raw_lines = text.split("\n")
    wrapped = []
    for line in raw_lines:
        wrapped += textwrap.wrap(line, width=max_chars_per_line)

    font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    dummy = Image.new("RGB", (1, 1))
    dummy_draw = ImageDraw.Draw(dummy)

    images = []
    bboxes = []
    number = 0

    for i in range(0, len(wrapped), max_lines_per_image):
        chunk = wrapped[i : i + max_lines_per_image]

        widths = [dummy_draw.textbbox((0, 0), line, font=font)[2] for line in chunk]
        max_w = max(widths) if widths else 0
        hh = dummy_draw.textbbox((0, 0), "Ay", font=font)[3]
        total_h = hh * len(chunk) + margin * 2

        img = Image.new("RGB", (max_w + margin * 2, total_h), "white")
        draw = ImageDraw.Draw(img)

        y = margin
        for line in chunk:
            draw.text((margin, y), line, font=font, fill="black")

            pattern = rf"(?<=The pass key is ){re.escape(answer_char)}(?=\.)"
            for m in re.finditer(pattern, line):
                pre_text = line[: m.start()]
                pb = dummy_draw.textbbox((0, 0), pre_text, font=font)
                x0 = margin + (pb[2] - pb[0])
                cbb = dummy_draw.textbbox((0, 0), answer_char, font=font)
                x1 = x0 + (cbb[2] - cbb[0])
                y0, y1 = y, y + hh

                if make_evidence_data:
                    draw.rectangle((x0, y0, x1, y1), outline="red", width=2)

                bboxes.append({str(number): (x0, y0, x1, y1)})

            y += hh

        images.append(img)
        number += 1

    return images, bboxes

def main(args):
# 1) 원본 로드
    dataset=load_dataset("nanotron/simple_needle_in_a_hay_stack")
    context_lengths = [1024, 2048, 4096, 8192]#, 16384, 32768]
    sampled_subsets = []
    for cl in context_lengths:
        subset = dataset['train'].filter(lambda x: x['context_length'] == cl)
        
        sampled = subset.shuffle(seed=args.seed).select(range(150))
        sampled_subsets.append(sampled)

    raw_ds = concatenate_datasets(sampled_subsets)
    # import pdb;pdb.set_trace()
    # raw_ds = load_dataset("nanotron/simple_needle_in_a_hay_stack")

    # 2) 36개 키
    chars = list(string.digits + string.ascii_lowercase)

    # 3) 문자 교체 함수
    def make_replace_fn(c):
        def fn(example):
            orig = str(example["answer"])                # 안전하게 str()
            pattern = rf"\b{re.escape(orig)}\b"
            new_prompt = re.sub(pattern, c, example["prompt"])
            return {"prompt": new_prompt.replace(f"{c} is the pass key.",''), "answer": c}
        return fn

    new_splits = {}
    # for split in raw_ds.keys():  # train, validation, test 등
        
    limited = raw_ds.select(range(min(100, raw_ds.num_rows)))
    # — b) answer 컬럼을 문자열(Value("string"))로 캐스팅
    limited = limited.cast_column("answer", Value("string"))
    
    # — c) 36개 문자 map & 모아두기
    expanded = []
    for c in chars:
        ds_c = limited.map(make_replace_fn(c),  
                        desc=f"Map {'asd'}→'{c}'")
        expanded.append(ds_c)
    
    # — d) 스키마가 완전히 일치하므로 안전하게 합치기
    new_splits['train'] = concatenate_datasets(expanded)

    subset = DatasetDict(new_splits)
    final_dataset = subset['train']
    # final_dataset = concatenate_datasets(sampled_subsets)

    output_dir = 'img_dataset_an_dev/' if args.seed!=42 else 'img_dataset_an/'
    os.makedirs(output_dir, exist_ok=True)

    save_evidences = []
    for idx in tqdm(range(len(final_dataset))):
        prompt_txt = final_dataset[idx]['prompt'].replace(
            f"{final_dataset[idx]['answer']} is the pass key.", ""
        )
        ans = final_dataset[idx]['answer']  # single-char (0–9, a–z)

        if args.fine_grained:
            images, bboxes = create_text_images2(
                prompt_txt, answer_char=ans,
                max_chars_per_line=50, max_lines_per_image=15,
                font_path="", font_size=30, margin=15,
                make_evidence_data=False
            )
        else:
            images, bboxes = create_text_images(
                prompt_txt, answer_char=ans,
                max_chars_per_line=50, max_lines_per_image=15,
                font_path="", font_size=30, margin=15,
                make_evidence_data=False
            )

        # 이미지 저장
        subdir = os.path.join(output_dir, str(idx))
        os.makedirs(subdir, exist_ok=True)
        for i, img in enumerate(images):
            img.save(os.path.join(subdir, f"{idx}_{i}.jpg"))

        # bbox 데이터 축적
        entry = {"answer": ans}
        if args.seed != 42: entry["prompt"] = final_dataset[idx]["prompt"]
        if args.fine_grained:
            entry["bboxes_per_chunk"] = bboxes
        else:
            entry["bboxes"] = bboxes
        save_evidences.append({str(idx): entry})

    # JSON으로 저장
    json_name = os.path.join(
        output_dir,
        "bbox_fg.json" if args.fine_grained else "bbox.json"
    )
    with open(json_name, "w", encoding="utf-8") as f:
        json.dump(save_evidences, f, indent=4, ensure_ascii=False)
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fine_grained', action='store_true')
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()
    main(args)


'''
python exp_case.py --fine_grained
'''