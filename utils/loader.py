#%%
from transformers import (Qwen2VLForConditionalGeneration, 
                          LlavaForConditionalGeneration, 
                          AutoTokenizer, 
                          AutoProcessor, 
                          AutoModel, 
                          CLIPImageProcessor,
                          AutoModelForCausalLM,
                          AutoConfig, Qwen2VLProcessor)

import torch
import clip
import jsonlines
import json

import os, sys

import faiss


from typing import List, Dict

def parse_mask_heads(head_specs: List[str]) -> Dict[int, List[int]]:
    """
    head_specs is something like ["l0_h4", "l0_h30", "l13_h27", ...].
    This will return a dict: {0: [4, 30], 13: [27], ...}
    """
    mask_dict = {}
    for spec in head_specs:
        layer_str, head_str = spec.split("_")
        layer_idx = int(layer_str.replace("l", ""))
        head_idx = int(head_str.replace("h", ""))
        if layer_idx not in mask_dict:
            mask_dict[layer_idx] = []
        mask_dict[layer_idx].append(head_idx)
    return mask_dict


def load_clip(args):

    if args.clip_type == "clip":
        model, preprocess = clip.load("ViT-L/14@336px", device="cuda", jit=False)
        tokenizer = None

    elif "internvl" in args.clip_type.lower():
        model = AutoModel.from_pretrained(
            "OpenGVLab/InternVL-14B-224px",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=0,
        ).eval()

        preprocess = CLIPImageProcessor.from_pretrained("OpenGVLab/InternVL-14B-224px")
        tokenizer = AutoTokenizer.from_pretrained(
            "OpenGVLab/InternVL-14B-224px", use_fast=False, add_eos_token=True
        )
        tokenizer.pad_token_id = 0  # set pad_token_id to 0

    return model, preprocess, tokenizer

def monkey_patch_attention_forward(attn_module, layer_idx, mask_dict):
    """
    attn_module: DeepseekV2Attention (또는 LlamaAttention) 인스턴스
    layer_idx: 몇 번째 layer인지
    mask_dict: {'l0_h0': 0.0, ...} 형태
    """
    original_forward = attn_module.forward  # 백업

    def patched_forward(
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        **kwargs
    ):
        attn_output, attn_weights, present_key_value = original_forward(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=True,
            use_cache=use_cache,
            **kwargs
        )
        num_heads = attn_weights.shape[1]
        _dim = int(hidden_states.shape[-1]/num_heads)
        for head_idx in range(num_heads):
            key = f"l{layer_idx}_h{head_idx}"
            if key in mask_dict:
                attn_output[:,:,head_idx*_dim : head_idx*(_dim+1)] = 0.0

        return (attn_output, attn_weights, present_key_value)

    attn_module.forward = patched_forward
    
def apply_monkey_patch(model, dictionary):
    # model.model.layers 각각에 대해 monkey patch
    for layer_idx, layer_module in enumerate(model.language.model.layers):
        attn_module = layer_module.self_attn
        monkey_patch_attention_forward(attn_module, layer_idx, dictionary)

    return model

def lvlm_loader(model_id, device, args = None):
    min_pixels = 256*28*28
    max_pixels = 256*28*28
    if 'qwen' in model_id.lower():
        print('a')
        if args.do_masking:
            
            if args.unmasking:
                from .models.qwenvl2_sink import Qwen2VLForConditionalGenerationMasked
                model = Qwen2VLForConditionalGenerationMasked.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    mask_dict = args.masking_dict,
                    beta = args.beta
                )
            else:   
                from .models.qwenvl2_mask import Qwen2VLForConditionalGenerationMasked
                model = Qwen2VLForConditionalGenerationMasked.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    mask_dict = args.masking_dict,
                )

        else:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
            )

        processor = AutoProcessor.from_pretrained(model_id, min_pixels=min_pixels, max_pixels=max_pixels)        
        
        return model, processor
    
    elif 'llava' in model_id.lower():
        print('b')
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            # low_cpu_mem_usage=True, 
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_id, min_pixels=min_pixels, max_pixels=max_pixels)  
        return model, processor
    
    elif 'OpenGVLab' in model_id:
        
        if args.do_masking:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            base_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

            
            if args.unmasking:
                from .models.internvl2_sink import MaskInternVLChatModel
                masked_model = MaskInternVLChatModel.from_pretrained(model_id,
                                                                    mask_dict = args.masking_dict, 
                                                                    beta = args.beta,
                                                                    torch_dtype=torch.float16, 
                                                                    use_flash_attn=False,
                                                                    low_cpu_mem_usage=False,
                                                                    trust_remote_code=True).eval().to(args.device)
                
            else:
                from .models.internvl2_mask import MaskInternVLChatModel
                
                masked_model = MaskInternVLChatModel.from_pretrained(model_id,
                                                                    mask_dict = args.masking_dict, 
                                                                    torch_dtype=torch.float16, 
                                                                    use_flash_attn=False,
                                                                    low_cpu_mem_usage=False,
                                                                    trust_remote_code=True).eval().to(args.device)
            masked_model.config.llm_config.use_cache = False
            return masked_model, tokenizer
            
        else:
            print('d')
            model = AutoModel.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                use_flash_attn=False,
                trust_remote_code=True).eval().cuda()
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
            return model, tokenizer
    
    else: raise ValueError("Check the model_id")
  
'''
vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
'''
def load_dataset(args):
    if args.datasets.lower() == "mmqa":
        ds_path = f"./dataset/{args.datasets.upper()}_{args.dev_or_test}_imageQ.json"
    elif args.datasets.lower() == "manyqa":
        ds_path = f"./dataset/ManyModalQAData/official_aaai_split_{args.dev_or_test}_data.json"
        
    meta_path = f"./dataset/metadata/{args.datasets.lower()}_imageQ_{args.dev_or_test}_metadata.json"
    
    with open(ds_path, "r") as f:
        dataset = json.load(f)
    with open(meta_path, "r") as f:
        metadata = json.load(f)
    
    return dataset, metadata

def dataset_loader(args):
    with open("./dataset/imgs.lineidx", "r") as fp_lineidx:
        lineidx = [int(i.strip()) for i in fp_lineidx.readlines()]
    if args.feature:
        index_path = f"./index/{args.datasets}_train_{args.clip_type}_index.index" # _v2
        index_to_id_path = f"./index/{args.datasets}_train_{args.clip_type}_index_to_id.json"# _v2
    else:
        index_path = f"./index/{args.datasets}_{args.dev_or_test}_{args.clip_type}_index.index" # _v2
        index_to_id_path = f"./index/{args.datasets}_{args.dev_or_test}_{args.clip_type}_index_to_id.json" # _v2
    if args.datasets == 'WebQA':
        if args.dev_or_test == 'train':
            path = f"./dataset/{args.datasets}_{args.dev_or_test}_image_{args.model_id.split('/')[1]}.json"
        else:
            path = f"./dataset/{args.datasets}_{args.dev_or_test}_image.json"
        if args.feature:
            path = f"./dataset/feature_{args.datasets}_{args.dev_or_test}_image_{args.model_id.split('/')[1]}.json"
        metadata = f"./dataset/{args.datasets}_image_metadata.json"
    else:
        if args.dev_or_test == 'train':
            path = f"./dataset/{args.datasets}_{args.dev_or_test}_imageQ_{args.model_id.split('/')[1]}.json"
        else:
            path = f"./dataset/{args.datasets}_{args.dev_or_test}_imageQ.json"
        if args.feature_all:
            path = f"./dataset/{args.datasets}_{args.dev_or_test}_imageQ.json"
        if args.feature:
            path = f"./dataset/feature_{args.datasets}_{args.dev_or_test}_imageQ_{args.model_id.split('/')[1]}.json"
            
        metadata_path = f"./dataset/{args.datasets}_imageQ_metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)  
    index = faiss.read_index(index_path)
    with open(index_to_id_path, 'r') as f:
        index_to_image_id = json.load(f)
    with open(path, 'r') as f:
        dataset = json.load(f)                                        

    if args.datasets == 'WebQA':
        with open(f"./dataset/{args.datasets}_caption_test.json", 'r') as f:
            caption1 = json.load(f)
        with open(f"./dataset/{args.datasets}_caption_train_dev.json", 'r') as f:
            caption = json.load(f)
        caption.update(caption1)
        return dataset, caption, index_to_image_id, lineidx, index, metadata
    else:
        return dataset, index_to_image_id, lineidx, index, metadata
    
def load_jsonl_path_to_dataset(path, config):
    datatset = []
    
    with jsonlines.open(path, mode='r') as reader:
        for obj in reader:
            if f"{config.data_format}" in obj.get('metadata', {}).get('modalities')[0]:
                datatset.append(obj)
    return datatset
