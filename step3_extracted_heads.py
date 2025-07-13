#%%
import torch

import json
import os
import random
random.seed(22)

from transformers import AutoConfig

from dataclasses import dataclass

import numpy as np
#%%
def make_bd(image_mean):
    keys = ['0', '1', 'save']
    boundary_dict = {key: [] for key in keys} 

    for key, value in image_mean.items():
        if value ==0:
            boundary_dict['0'].append(key)
        elif (value > 0) and (value < 0.1):
            boundary_dict['1'].append(key)
        else:
            boundary_dict['save'].append(key)
        
    return boundary_dict

#%%
def main(args):

# @dataclass
# class Config:
#     model_id = ""
#     te_fg_cg = ""
# args = Config()
# args.model_id = "Qwen/Qwen2-VL-7B-Instruct" # Qwen/Qwen2-VL-7B-Instruct, OpenGVLab/InternVL2-8B
# args.te_fg_cg = "fg"

    path = "retrieval_score/"
    if args.te_fg_cg == 'te':
        path += "text/"
        path += f"{args.model_id.split('/')[1]}_{args.te_fg_cg}_2.json"
    else:
        path += "image/"
        path += f"{args.model_id.split('/')[1]}_{args.te_fg_cg}_2.json"


    with open(path, 'r') as f:
        data = json.load(f)

    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
  
    total_layer_head = {}
    if 'intern' in args.model_id.lower():
        n_heads = config.llm_config.num_attention_heads
        n_layer = config.llm_config.num_hidden_layers
    else:
        n_heads = config.num_attention_heads # 10
        n_layer = config.num_hidden_layers # 12
    total_layer_head[f"{args.model_id}"] = [n_layer, n_heads]

    layers, heads = total_layer_head[args.model_id]    
    default_dictionary = {}

    for layer in range(layers):
        for head in range(heads):
            default_dictionary[f'l{layer}_h{head}'] = 0
            
    
    for layer in range(layers):
        for head in range(heads):
            _t = torch.tensor(data[f'l{layer}_h{head}'])
            if sum(torch.where(_t != 0, 0, 1)) > 0.1 * len(data[f'l{layer}_h{head}']):
                default_dictionary[f'l{layer}_h{head}'] = sum(data[f'l{layer}_h{head}'])/len(data[f'l{layer}_h{head}'])
                # default_dictionary[f'l{layer}_h{head}'] = 0
            else:
                default_dictionary[f'l{layer}_h{head}'] = 0
    
    boundary_dict=make_bd(default_dictionary)
    save_path = 'retrieval_score'

    if os.path.exists(save_path): pass
    else: os.mkdir(save_path)
        
    save_path += "/retrieval_head"

    if os.path.exists(save_path): pass
    else: os.mkdir(save_path)
    save_path += f"/{args.model_id.split('/')[1]}_{args.te_fg_cg}2.json"

    with open(save_path, 'w') as f:
        json.dump(boundary_dict['save'], f, indent=4, ensure_ascii=False)
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="", choices=["Qwen/Qwen2-VL-7B-Instruct",
                                                           "Qwen/Qwen2-VL-2B-Instruct", 
                                                           "OpenGVLab/InternVL2-8B",
                                                           "OpenGVLab/InternVL2-2B"])
    parser.add_argument("--te_fg_cg", default="fg")
    args = parser.parse_args()
    main(args)
'''
python step3_extracted_heads.py --model_id Qwen/Qwen2-VL-7B-Instruct --te_fg_cg fg
python step3_extracted_heads.py --model_id Qwen/Qwen2-VL-2B-Instruct --te_fg_cg fg
python step3_extracted_heads.py --model_id OpenGVLab/InternVL2-8B --te_fg_cg fg

python step3_extracted_heads.py --model_id Qwen/Qwen2-VL-7B-Instruct --te_fg_cg te
python step3_extracted_heads.py --model_id Qwen/Qwen2-VL-2B-Instruct --te_fg_cg te
python step3_extracted_heads.py --model_id OpenGVLab/InternVL2-8B --te_fg_cg te
'''
