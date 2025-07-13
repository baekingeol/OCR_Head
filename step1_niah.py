#%%
from datasets import load_dataset, Dataset

import os, re, json, random, textwrap
from tqdm import tqdm

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

def main():
    SIZES = [1024, 2048, 4096, 8192]
    NEEDLE_START = 191
    NEEDLES_PER_LEN = 150
    SEED = 42
    random.seed(SEED)

    hay_ds = load_dataset("opencompass/NeedleBench", "en_haystack_texts", split="test")
    ret_ds = load_dataset("opencompass/NeedleBench", "retrieval_needles", split="test")

    candidate_idxs = list(range(NEEDLE_START, len(ret_ds)))   # 191 ~ 375 → 185개
    needles_idx = random.sample(candidate_idxs, NEEDLES_PER_LEN)
    needles_idx.sort()                                        # 순서 유지(선택)

    hay_big = "".join(hay_ds["text"])
    cursor  = 0
    L_hay   = len(hay_big)

    def take_chars(n: int) -> str:
        global cursor
        if n <= 0:
            return ""
        if cursor + n <= L_hay:
            chunk = hay_big[cursor : cursor + n]
            cursor += n
        else:  # 끝을 넘어가면 앞부분 이어 붙이기
            part1 = hay_big[cursor:]
            part2_len = n - len(part1)
            part2 = hay_big[:part2_len]
            chunk = part1 + part2
            cursor = part2_len
        return chunk

    samples = []
    for idx in tqdm(needles_idx, desc="building 600 samples"):
        q_text = ret_ds["retrieval_question"][idx]
        a_text = ret_ds["arg2"][idx]
        needle = ret_ds["needle"][idx]
        n_len  = len(needle)

        for tgt_len in SIZES:
            remain = tgt_len - n_len
            if remain <= 0:
                raise ValueError(f"Needle too long for target {tgt_len}")
            context = take_chars(remain)
            insert_pos = random.randint(0, remain)
            full_ctx = context[:insert_pos] + needle + context[insert_pos:]

            samples.append({
                "id"      : f"idx{idx}_len{tgt_len}",
                "length"  : tgt_len,
                "question": q_text,
                "answer"  : a_text,
                "needle"  : needle,
                "context" : full_ctx
            })

    assert len(samples) == NEEDLES_PER_LEN * len(SIZES) 

    FONT_PATH, FONT_SIZE = "", 28
    MAX_CHARS, MAX_LINES = 50, 15
    MARGIN, SEED = 15, 42

    SAVE_DIR = "img_dataset_p" if SEED == 42 else "img_dataset_p_dev"
    Path(SAVE_DIR).mkdir(exist_ok=True)

    def render_with_bbox(full_text, needle_sentence,
                        font_path=FONT_PATH, font_size=FONT_SIZE,
                        max_chars=MAX_CHARS, max_lines=MAX_LINES, margin=MARGIN):
        
        # 0) needle 전역 위치
        start_idx = full_text.lower().find(needle_sentence.lower())
        if start_idx == -1:
            raise ValueError("needle not found in prompt")
        end_idx = start_idx + len(needle_sentence)

        # 1) 줄 wrap + 원본 위치 매핑
        raw_lines = full_text.split("\n")
        wrapped, offsets = [], []
        char_ptr = 0 
        
        for ln in raw_lines:
            for part in textwrap.wrap(ln, width=max_chars) or [""]:
                wrapped.append(part)
                
                pos = full_text.lower().find(part.lower(), char_ptr)
                if pos == -1:
                    pos = full_text.lower().find(part.lower())
                offsets.append(pos)
                char_ptr = pos + len(part)

        font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
        dummy = Image.new("RGB", (1, 1)); d0 = ImageDraw.Draw(dummy)
        _, _, _, line_h = d0.textbbox((0, 0), "Ay", font=font)

        images, bboxes_dict = [], {}
        for img_idx in range(0, len(wrapped), max_lines):
            chunk  = wrapped[img_idx:img_idx+max_lines]
            offs   = offsets[img_idx:img_idx+max_lines]

            max_w  = max(d0.textbbox((0, 0), ln, font=font)[2] for ln in chunk) if chunk else 0
            img_h  = line_h * len(chunk) + margin*2
            img    = Image.new("RGB", (max_w + margin*2, img_h), "white")
            draw   = ImageDraw.Draw(img)

            y      = margin
            bbox_list = []
            for ln, ln_start in zip(chunk, offs):
                draw.text((margin, y), ln, font=font, fill="black")
                
                for m in re.finditer(r"\b\w+\b", ln):
                    g_start = ln_start + m.start()
                    g_end   = ln_start + m.end()
                    if not (start_idx <= g_start < end_idx):
                        continue
                    
                    pre_w   = d0.textbbox((0, 0), ln[:m.start()], font=font)[2]
                    word_w  = d0.textbbox((0, 0), m.group(),     font=font)[2]
                    
                    x0, y0  = margin + pre_w, y
                    x1, y1  = x0 + word_w, y + line_h
                    bbox_list.append([x0, y0, x1, y1])
                
                y += line_h

            images.append(img)
            bboxes_dict[str(img_idx // max_lines)] = bbox_list
        
        return images, bboxes_dict

    save_evidences = []
    for idx in tqdm(range(len(samples)), desc="render"):
        prompt  = samples[idx]["context"]
        needle  = samples[idx]["answer"]
        questions = samples[idx]['question']
        imgs, bbox_per_img = render_with_bbox(prompt.replace('\n',''), needle)

        folder = Path(SAVE_DIR) / str(idx); folder.mkdir(exist_ok=True)
        for n, im in enumerate(imgs):
            im.save(folder / f"{idx}_{n}.jpg")
        
        bb = {}
        bb[f'{str(idx)}'] = {}
        for key, val in bbox_per_img.items():
            if val == []:
                continue
            else:
                bb[f'{str(idx)}'][key] = val 
                
        save_evidences.append({
            str(idx): bb[f'{str(idx)}'],
            "answer": needle,
            "prompt": prompt,
            "question": questions
        })
    
    with open(Path(SAVE_DIR) / "bbox_fg.json", "w", encoding="utf-8") as f:
        json.dump(save_evidences, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()