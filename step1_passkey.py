#%%
import re
import json
from PIL import Image, ImageDraw, ImageFont
import textwrap
import os
from tqdm import tqdm

from datasets import load_dataset, concatenate_datasets
def main(args):
    #%%
    dataset=load_dataset("nanotron/simple_needle_in_a_hay_stack")
    context_lengths = [1024, 2048, 4096, 8192]#, 16384, 32768]
    sampled_subsets = []
    for cl in context_lengths:
        subset = dataset['train'].filter(lambda x: x['context_length'] == cl)
        
        sampled = subset.shuffle(seed=args.seed).select(range(150))
        sampled_subsets.append(sampled)

    final_dataset = concatenate_datasets(sampled_subsets)
    #%%
    import textwrap
    import re
    from PIL import Image, ImageDraw, ImageFont

    def create_text_images2(text, max_chars_per_line=50, max_lines_per_image=15, 
                        font_path="arial.ttf", font_size=20, margin=10, make_evidence_data=False):
        # 먼저, 텍스트를 \n 기준으로 분리한 뒤, 각 줄을 max_chars_per_line 단위로 wrap합니다.
        raw_lines = text.split('\n')
        wrapped_lines = []
        for line in raw_lines:
            wrapped_lines.extend(textwrap.wrap(line, width=max_chars_per_line))
        
        images = []
        bbox_evidence = []  # 각 숫자에 대해 bounding box 정보를 저장할 리스트입니다.
        box_count = 0      # 각 bounding box에 대한 고유 인덱스

        # 폰트 로드: font_path가 지정되어 있으면 truetype, 그렇지 않으면 기본 폰트를 사용합니다.
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()
        
        # 텍스트 크기 측정을 위해 임시 이미지와 draw 객체 생성
        dummy_img = Image.new("RGB", (1, 1))
        dummy_draw = ImageDraw.Draw(dummy_img)
        number = 0
        tmp2 = {}
        # wrapped_lines를 max_lines_per_image 단위로 나눕니다.
        for i in range(0, len(wrapped_lines), max_lines_per_image):
            chunk = wrapped_lines[i:i+max_lines_per_image]
            
            # 각 줄의 너비를 측정하여 이미지의 최대 너비 결정
            line_widths = [
                dummy_draw.textbbox((0, 0), line, font=font)[2] - dummy_draw.textbbox((0, 0), line, font=font)[0]
                for line in chunk
            ]
            max_width = max(line_widths) if line_widths else 0
            
            # "Ay" 텍스트의 높이를 측정하여 줄 높이 산출
            bbox = dummy_draw.textbbox((0, 0), "Ay", font=font)
            line_height = bbox[3] - bbox[1]
            total_height = line_height * len(chunk) + margin * 2
            
            # 흰색 배경 이미지를 생성
            img = Image.new("RGB", (max_width + margin * 2, total_height), "white")
            draw = ImageDraw.Draw(img)
            
            y = margin  # 텍스트를 그릴 y 좌표
            
            
            for line in chunk:
                # 전체 텍스트를 그림
                draw.text((margin, y), line, font=font, fill="black")
                # 정규 표현식을 사용해 각 줄에서 하나 이상의 연속된 숫자를 찾습니다.
                for match in re.finditer(r'\d+', line):
                    # match 시작 전 텍스트 길이 계산
                    preceding_text = line[:match.start()]
                    pre_bbox = dummy_draw.textbbox((0, 0), preceding_text, font=font)
                    preceding_width = pre_bbox[2] - pre_bbox[0]
                    
                    # match 그룹 전체의 시작 x 좌표
                    current_x = margin + preceding_width
                    # match.group()는 여러 숫자가 이어진 문자열이므로, 각 숫자별로 별도의 box를 생성합니다.
                    
                    for digit in match.group():
                        # 각 digit의 너비 측정
                        digit_bbox = dummy_draw.textbbox((0, 0), digit, font=font)
                        digit_width = digit_bbox[2] - digit_bbox[0]
                        
                        x0 = current_x
                        y0 = y
                        x1 = current_x + digit_width
                        y1 = y + line_height
                        if make_evidence_data:
                            draw.rectangle((x0, y0, x1, y1), outline="red", width=2)
                        # 각 bounding box 정보를 box_count를 key로 저장합니다.
                        tmp = [x0, y0, x1, y1]
                        
                        bbox_evidence.append(tmp)
                        
                        tmp2[f'{str(number)}'] = bbox_evidence
                        # bbox_evidence
                        box_count += 1
                        # 다음 숫자의 시작 x 좌표 업데이트
                        current_x += digit_width
                    # import pdb;pdb.set_trace()
                y += line_height  # 다음 줄로 이동
            number += 1
            images.append(img)
            
        return images, tmp2

    def create_text_images(text, max_chars_per_line=50, max_lines_per_image=15, 
                        font_path="arial.ttf", font_size=20, margin=10, make_evidence_data=False):

        raw_lines = text.split('\n')
        wrapped_lines = []
        for line in raw_lines:
            wrapped_lines.extend(textwrap.wrap(line, width=max_chars_per_line))
        
        images = []
        
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()
        
        dummy_img = Image.new("RGB", (1, 1))
        dummy_draw = ImageDraw.Draw(dummy_img)
        
        bbox_evidence = []
        number = 0
        for i in range(0, len(wrapped_lines), max_lines_per_image):
            chunk = wrapped_lines[i:i+max_lines_per_image]
            
            line_widths = [
                dummy_draw.textbbox((0, 0), line, font=font)[2] - dummy_draw.textbbox((0, 0), line, font=font)[0]
                for line in chunk
            ]
            max_width = max(line_widths) if line_widths else 0
            
            bbox = dummy_draw.textbbox((0, 0), "Ay", font=font)
            line_height = bbox[3] - bbox[1]
            total_height = line_height * len(chunk) + margin * 2
            
            img = Image.new("RGB", (max_width + margin * 2, total_height), "white")
            draw = ImageDraw.Draw(img)
            
            y = margin

            for line in chunk:
                draw.text((margin, y), line, font=font, fill="black")
                for match in re.finditer(r'\d+', line):
                    digits = match.group()
                    preceding_text = line[:match.start()]
                    pre_bbox = dummy_draw.textbbox((0, 0), preceding_text, font=font)
                    preceding_width = pre_bbox[2] - pre_bbox[0]
                    digits_bbox = dummy_draw.textbbox((0, 0), digits, font=font)
                    digits_width = digits_bbox[2] - digits_bbox[0]
                    
                    x0 = margin + preceding_width
                    y0 = y
                    x1 = x0 + digits_width
                    y1 = y + line_height
                    if make_evidence_data:
                        draw.rectangle((x0, y0, x1, y1), outline="red", width=2)
                    tmp = {f"{number}":(x0, y0, x1, y1)}
                    bbox_evidence.append(tmp)
                y += line_height
            number+=1
            images.append(img)
        
        return images, bbox_evidence

    if args.seed != 42:
        path = 'img_dataset_dev/'  
    else:  
        path = 'img_dataset/'
    if os.path.exists(path)== False:
        os.mkdir(path)
        
    save_evidences = []
    for number in tqdm(range(len(final_dataset))):
        if args.seed != 42:
            path = 'img_dataset_dev/'  
        else:  
            path = 'img_dataset/'
            
        if args.fine_grained:
            # import pdb;pdb.set_trace()
            images, bbox_evidence = create_text_images2(final_dataset[number]['prompt'].replace(f"{str(final_dataset[number]['answer'])} is the pass key.", ''), max_chars_per_line=50, max_lines_per_image=15,
                                font_path="", font_size=30, margin=15, make_evidence_data = False)
            # import pdb;pdb.set_trace()
        else:
            images, bbox_evidence = create_text_images(final_dataset[number]['prompt'].replace(f"{str(final_dataset[number]['answer'])} is the pass key.", ''), max_chars_per_line=50, max_lines_per_image=15,
                                    font_path="", font_size=30, margin=15, make_evidence_data = False)
        # break
        path += f"{number}/"
        if os.path.exists(path)== False:
            os.mkdir(path)
            
        for img_num in range(len(images)):
            save_img_name = f"{number}_{img_num}.jpg" 
            images[img_num].save(path + save_img_name)
            
        if args.fine_grained:
            tmp = {f"{str(number)}": bbox_evidence}
            tmp['answer'] = final_dataset[number]['answer']
            if args.seed != 42:
                tmp['prompt'] = final_dataset[number]['prompt']
            save_evidences.append(tmp)
        else:
            if isinstance(bbox_evidence[0], dict):
                tmp = {f"{str(number)}": bbox_evidence[0]}
                tmp['answer'] = final_dataset[number]['answer']
                if args.seed != 42:
                    tmp['prompt'] = final_dataset[number]['prompt']
                save_evidences.append(tmp)

    
    if args.seed != 42:
        path = 'img_dataset_dev/'
        if args.fine_grained: save_name = path+'bbox_fg.json'
        else: save_name = path+'bbox.json'
    else:
        path = 'img_dataset/'
        if args.fine_grained: save_name = path+'bbox_fg.json'
        else: save_name = path+'bbox.json'
    
    with open(save_name, 'w') as f:
        json.dump(save_evidences, f, indent=4, ensure_ascii=False)
# %%
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fine_grained', action='store_true')
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()
    main(args)
'''
python passkey_image_dataset.py --fine_grained
python passkey_image_dataset.py

python passkey_image_dataset.py --fine_grained --seed 22
python passkey_image_dataset.py --seed 22


'''