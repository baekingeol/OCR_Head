from io import BytesIO
from PIL import Image
import base64, os, spacy
from typing import Tuple

import torch.nn.functional as F
import torch

from transformers.image_transforms import resize
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from .prompt import prompt

from qwen_vl_utils.vision_process import extract_vision_info, fetch_image
from qwen_vl_utils import process_vision_info

import numpy as np

nlp = spacy.load("en_core_web_sm")
def return_weights(args, messages, xmin=0, ymin=0, xmax=0, ymax=0):
    if 'qwen' in args.model_id.lower():
        vision_info = extract_vision_info(messages)
        resized_step1_img = fetch_image(vision_info[0])
        height, width, _ = np.array(resized_step1_img).shape

        resized_height, resized_width = smart_resize( 
            height,
            width,
            factor=args.processor.image_processor.patch_size * args.processor.image_processor.merge_size,
            min_pixels=args.processor.image_processor.min_pixels,
            max_pixels=args.processor.image_processor.max_pixels,
        ) 

        resized_img_array = resize(
            np.array(resized_step1_img), size=(resized_height, resized_width), resample=args.processor.image_processor.resample, input_data_format=None
        ) 
        y_scale, x_scale = resized_img_array.shape[0] / args.label_array.shape[0], resized_img_array.shape[1] / args.label_array.shape[1]
        
    elif 'intern' in args.model_id.lower():
        y_scale, x_scale = 448 / args.label_array.shape[0], 448 / args.label_array.shape[1]
        # TODO 잘 만들어진건지 확인해봐야함. 
        
    new_xmin, new_ymin, new_xmax, new_ymax = int(xmin * x_scale), int(ymin * y_scale), int(xmax * x_scale), int(ymax * y_scale)

    if 'qwen' in args.model_id.lower():
        weights = compute_tile_weights(resized_img_array.shape[1], resized_img_array.shape[0], 28, 28, (new_xmin, new_ymin, new_xmax, new_ymax))
        
    elif 'intern' in args.model_id.lower():
        weights = compute_tile_weights(448, 448, 28, 28, (new_xmin, new_ymin, new_xmax, new_ymax))
    
    return weights

def valuable_weights(weights):
    new_weight = {}
    for key, value in weights.items():
        if value > 0.1: # hyperparameter
            new_weight[key] = value
    return new_weight
    
def normalize_rows(tensor):
    row_sums = tensor.sum(dim=1, keepdim=True)
    row_sums[row_sums == 0] = 1.0

    normalized_tensor = tensor / row_sums
    return normalized_tensor

def return_attention(model_id, attentions: Tuple, layer = -1,head_n = 0, norm = True, bos_as_weight = False,
                    mean_head = False, is_self_attn = False, masking = None) -> torch.Tensor: # output: [gened tokens, ]
    total_length = attentions[-1][layer][0].shape[-1]
    
    # TODO deepseek 에서 attentions 의 output이 qwen이랑 다르기 때문에 맞춰주는 과정이 필요함
        
    if is_self_attn: return attentions[0][layer][0].mean(dim=0).to('cpu').squeeze() # 평균으로 관찰하는것이 맞는걸까?
    else:
        for num, attention in enumerate(attentions):
            if num == 0: 
                if ('qwen' in model_id.lower()) or ('intern' in model_id.lower()):
                    
                    if mean_head:
                        attn = torch.tril(attention[layer][0].mean(dim = 0))
                    else:
                        attn = torch.tril(attention[layer][0][head_n])
                    value = torch.nn.functional.pad(attn, (0, total_length - attn.shape[0]))                
                else:
                    value = torch.zeros((attentions[0][layer].shape[-1], total_length)).to('cuda')

            else:
                if masking is not None:
                    if masking[num-1] == 1: pass
                    else: continue
                else: pass
                try:
                    if mean_head:
                        sample = attention[layer][0].mean(dim=0)
                    else:
                        sample = attention[layer][0][head_n]

                    if norm:
                        n_sample = normalize_rows(sample[:,1:])
                        sample=torch.cat([sample[:,:1], n_sample], dim = 1)
                        
                    if bos_as_weight:
                        (1-sample[:,:1].item()) * sample
                        
                    attn = torch.nn.functional.pad(sample, (0, total_length - sample.shape[1]))
                    value=torch.cat((value, attn), dim = 0)
                    
                except:
                    import pdb;pdb.set_trace()
        return value.to('cpu')

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=1):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=False, min_num = 1, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def generate(args, model, tokenizer, processor, input_content=None, conversation=None):
    if 'qwen' in args.model_id.lower():
        messages = [{"role": "user", "content": input_content}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        # import pdb;pdb.set_trace()
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        # additional_inputs = tokenizer(["the pass key is "], return_tensors="pt")
        # inputs.input_ids = torch.cat([inputs.input_ids, additional_inputs.input_ids], axis = 1)
        # inputs.attention_mask = torch.cat([inputs.attention_mask, additional_inputs.attention_mask], axis = 1)
        
        # import pdb;pdb.set_trace()
        inputs = inputs.to('cuda')
        attention_mask = torch.ones_like(inputs.input_ids)
        with torch.no_grad():
            outputs = model.generate(**inputs,
                                        max_new_tokens=128,
                                        # do_sample=False,
                                        return_dict_in_generate=True,
                                        # output_scores=True,
                                        # output_hidden_states=True,
                                        output_attentions=True,
                                        # return_legacy_cache=True
                                        )
            
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs.sequences)
        ]
        # generated_ids_trimmed = [
        #     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs)
        # ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        with_sp_output_text = processor.batch_decode(
            generated_ids_trimmed, clean_up_tokenization_spaces=False)
        # import pdb;pdb.set_trace()
        return outputs, generated_ids_trimmed, output_text, with_sp_output_text, inputs
    

    elif 'internvl' in args.model_id.lower():
        pass
    
def generate_w_masking(args, model, tokenizer, processor, input_content=None, conversation=None):
    if 'qwen' in args.model_id.lower():
        messages = [{"role": "user", "content": input_content}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to('cuda')
        attention_mask = torch.ones_like(inputs.input_ids)
        # import pdb;pdb.set_trace()
        with torch.no_grad():
            outputs = model.generate(**inputs,
                                        max_new_tokens=256)
        # import pdb;pdb.set_trace()
        # import pdb;pdb.set_trace() # processor.tokenizer.decode(outputs[0])
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        with_sp_output_text = processor.batch_decode(
            generated_ids_trimmed, clean_up_tokenization_spaces=False)

        return outputs, generated_ids_trimmed, output_text, with_sp_output_text, inputs
    
    
def start_end_ids(args, inputs, outputs, q_s=None, q_e=None):
    if 'Qwen' in args.model_id:
        start_ids = torch.nonzero(inputs.input_ids.squeeze().to('cpu') == q_s).squeeze()
        end_ids = torch.nonzero(inputs.input_ids.squeeze().to('cpu') == q_e).squeeze()
    else:
        if "deepseek-ai/deepseek-vl2-tiny" in args.model_id:
            target = 128815
        else:
            target = 100003
        is_target = (outputs.sequences.squeeze().to('cpu') == target).int()
        differences = is_target.diff(prepend=torch.tensor([0]))

        start_ids = (differences == 1).nonzero(as_tuple=True)[0]
        end_ids = (differences == -1).nonzero(as_tuple=True)[0] - 1
    return start_ids, end_ids

def compute_tile_weights(space_width, space_height, tile_width, tile_height, roi):
    roi_xmin, roi_ymin, roi_xmax, roi_ymax = roi

    roi_area = (roi_xmax - roi_xmin) * (roi_ymax - roi_ymin)
    n_cols = space_width // tile_width
    n_rows = space_height // tile_height
    tile_weights = {}
    if roi_area == 0:
        # raise ValueError("ROI의 면적이 0입니다.")
        
        for row in range(n_rows):
            for col in range(n_cols):
                tile_idx = row * n_cols + col
                tile_weights[tile_idx] = 0
        # print('1')
        return tile_weights
    else:

        for row in range(n_rows):
            for col in range(n_cols):
                tile_xmin = col * tile_width
                tile_ymin = row * tile_height
                tile_xmax = tile_xmin + tile_width
                tile_ymax = tile_ymin + tile_height
                
                inter_xmin = max(tile_xmin, roi_xmin)
                inter_ymin = max(tile_ymin, roi_ymin)
                inter_xmax = min(tile_xmax, roi_xmax)
                inter_ymax = min(tile_ymax, roi_ymax)
                
                inter_width = max(0, inter_xmax - inter_xmin)
                inter_height = max(0, inter_ymax - inter_ymin)
                inter_area = inter_width * inter_height
                
                tile_idx = row * n_cols + col
                
                weight_percentage = (inter_area / roi_area) * 100
                tile_weights[tile_idx] = weight_percentage
        # import pdb;pdb.set_trace()
        # print('2')
        return tile_weights
