#%%
import argparse
import os
import json
from tqdm import tqdm


import torch
import numpy as np

from transformers import AutoTokenizer

from utils.loader import lvlm_loader
from utils.utils import generate, return_attention, compute_tile_weights, return_weights, valuable_weights
from utils.metrics import compute_score

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from transformers import GenerationConfig
from transformers import StoppingCriteria, StoppingCriteriaList

class StopOnToken(StoppingCriteria):
    def __init__(self, stop_token_id):
        self.stop_token_id = stop_token_id

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids[0][-1] == self.stop_token_id:
            return True
        return False

from glob import glob
def main(args):
# @dataclass
# class Config():
#     model_id = '' # Qwen/Qwen2-VL-2B-Instruct Qwen/Qwen2-VL-7B-Instruct deepseek-ai/deepseek-vl2-tiny deepseek-ai/deepseek-vl2-Small
# args=Config()
# args.model_id = 'Qwen/Qwen2-VL-7B-Instruct'
# args.bos_as_weight = False
# args.norm = False
# args.do_ablation = False
# args.fine_grained = True
    args.do_masking = False
    args.do_ablation = False
    ppp = ["img_dataset", "img_dataset_p"]

    def load_data(metadata, number, fg = False):
        path = f'{pp}/'
        img_paths = sorted(glob(path + f'{number}/*'))
        evidence_page = int(list(metadata[number][f'{str(number)}'].keys())[0])
        bbox = list(metadata[number][f'{str(number)}'].values())[0]
        answer = metadata[number]["answer"]
        try:
            question = metadata[number]["question"]
        except: question = ''
        return img_paths, evidence_page, bbox, answer, question

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, processor = lvlm_loader(args.model_id, device = device, args = args)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    args.processor = processor

    model.config.use_cache = False
    model.gradient_checkpointing_disable

    if 'intern' in args.model_id.lower():
        n_heads = model.config.llm_config.num_attention_heads
        n_layer = model.config.llm_config.num_hidden_layers
        q_s, q_i, q_e = 92544, 92546, 92545
    else:
        n_heads = model.config.num_attention_heads # 10
        n_layer = model.config.num_hidden_layers # 12
        q_s, q_i, q_e = 151652, 151655, 151653

    features, mean_features = {}, {}
    for layer in range(n_layer):
        for head in range(n_heads):
            features[f'l{layer}_h{head}'] = []
            mean_features[f'l{layer}_h{head}'] = []
            
    for _number, pp in enumerate(ppp):
        # if _number == 0:
        #     continue
        if args.fine_grained: metadata_path = f"{pp}/bbox_fg.json"
        else: metadata_path = f"{pp}/bbox.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        error = []
        for number in tqdm(range(len(metadata))):
            try:
                if _number == 0:
                    img_paths, evidence_page, bbox, answer, _ = load_data(metadata, number)
                else: img_paths, evidence_page, bbox, answer, question = load_data(metadata, number)
                label_img_path = img_paths[evidence_page]
                label_img=Image.open(label_img_path)
                label_array=np.array(label_img)
                args.label_array = label_array
                
                if args.fine_grained:
                    input_content = [{"type": "image", "image": f'''{label_img_path}'''}]
                    messages = [{"role": "user", "content": input_content}]
                    weights_list, new_weight_list = [], []
                    
                    for box_len in range(len(bbox)):
                        xmin, ymin, xmax, ymax = bbox[box_len]
                        weights=return_weights(args, messages, xmin, ymin, xmax, ymax)
                        weights_list.append(weights)
                        new_weight = valuable_weights(weights)
                        new_weight_list.append(new_weight)
                                
                else:
                    xmin, ymin, xmax, ymax = bbox

                    input_content = [{"type": "image", "image": f'''{label_img_path}'''}]
                    messages = [{"role": "user", "content": input_content}]
                    weights=return_weights(args, messages, xmin, ymin, xmax, ymax)
                    new_weight = valuable_weights(weights)
                    
                # generate + start_ids
                if 'qwen' in args.model_id.lower():
                    input_content = [{"type": "image", "image": f'''{path}'''} for path in img_paths]
                    # import pdb;pdb.set_trace()
                    if _number == 0:
                        input_content.append({"type": "text", "text": '''What is the pass key?'''}) # What is the pass key? The pass key is
                    else:
                        input_content.append({"type": "text", "text": f'''{question}'''})
                        
                    outputs, generated_ids_trimmed, output_text, with_sp_output_text, inputs = generate(args, model, tokenizer, processor, input_content=input_content)
                    start_ids = torch.nonzero(inputs.input_ids.squeeze().to('cpu') == q_s).squeeze() +1
                    end_ids = torch.nonzero(inputs.input_ids.squeeze().to('cpu') == q_e).squeeze() -1
                    # import pdb;pdb.set_trace()        
                    
                elif 'internvl' in args.model_id.lower():
                    # TODO 생성코드 정리. start지점 만들기 weigth확인
                    num_patches = len(img_paths)
                    num_image_token = model.num_image_token
                    # import pdb;pdb.set_trace()
                    IMG_START_TOKEN = "<img>"
                    IMG_END_TOKEN   = "</img>"
                    IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"

                    image_tokens = IMG_START_TOKEN + (IMG_CONTEXT_TOKEN * (num_image_token * num_patches)) + IMG_END_TOKEN
                    if _number == 0:
                        user_question = "What is the passkey?"
                    else: user_question = f"{question}"
                    
                    prompt = f"{image_tokens}\n{user_question}" 
                    model_inputs = tokenizer(prompt, return_tensors="pt")
                    input_ids = model_inputs["input_ids"].to("cuda")
                    attention_mask = model_inputs["attention_mask"].to("cuda")
                    
                    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)   
                    generation_config = GenerationConfig(
                        max_new_tokens=32,
                        do_sample=False,
                    )
                    from utils.utils import load_image
                    pixel_values = torch.cat([load_image(path, max_num = 1).to(torch.float16).cuda() for path in img_paths], dim = 0)
                    stop_token_id = 281  # 원하는 토큰 id로 변경하세요.
                    stopping_criteria = StoppingCriteriaList([StopOnToken(stop_token_id)])
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            pixel_values=pixel_values,      # (batch=1, 3, 448, 448)
                            input_ids=input_ids,            # (batch=1, seq_len)
                            attention_mask=attention_mask,  
                            return_dict_in_generate=True,
                            output_attentions=True,
                            generation_config=generation_config,
                            stopping_criteria=stopping_criteria
                        )
                    generated_ids_trimmed=[outputs.sequences.squeeze(dim = 0)]
                    inputs = input_ids
                        
                    n_of_imgs= int(torch.sum(torch.where(input_ids == 92546, 1, 0)).item() / 256)
                    start_num = torch.nonzero(input_ids.squeeze().to('cpu') == q_s).item() +1
                    start_ids = start_num + 256 * torch.arange(0, n_of_imgs)
                    

                attentions = outputs['attentions']
                
                pattern_list = tokenizer(str(answer))
                
                tmp, context_idx, ans_idx = [], [], []
                # 생성된 길이만 돌리자 
                for n, tok in enumerate(generated_ids_trimmed[0].to('cpu').tolist()):
                    if 'qwen' in args.model_id.lower():
                        if tok in pattern_list.input_ids:
                            ans_idx.append(inputs.input_ids.shape[1] + n)
                            
                    elif 'internvl' in args.model_id.lower():
                        if tok in pattern_list.input_ids[1:]:
                            ans_idx.append(inputs.shape[1] + n)
                            
                for layer in range(n_layer):
                    for head in range(n_heads):
                        
                        attn = return_attention(args.model_id, attentions, layer = layer, head_n = head, 
                                                norm = args.norm, bos_as_weight= args.bos_as_weight)
                        if 'qwen' in args.model_id.lower():
                            image_retrieval_score = compute_score(pattern_list.input_ids, outputs.sequences.squeeze().to('cpu').tolist())
                        elif 'internvl' in args.model_id.lower():
                            image_retrieval_score = compute_score(pattern_list.input_ids[1:], outputs.sequences.squeeze().to('cpu').tolist())
                        # if (image_retrieval_score != 1.0) and (image_retrieval_score != 0):
                        #     import pdb;pdb.set_trace()
                        if args.fine_grained:
                            # import pdb;pdb.set_trace()
                            for box_len in range(len(bbox)):
                                pos_attn_position = set([weight + start_ids[evidence_page].item() for weight in list(new_weight_list[box_len].keys())])
                                
                                score = 0
                                # for ans in ans_idx:
                                # import pdb;pdb.set_trace()
                                try:
                                    index = torch.argmax(attn[ans_idx[box_len],:-len(generated_ids_trimmed[0])]).item()            
                                except:
                                    continue
                                # if answer in output_text:
                                #     features[f"l{str(layer)}_h{str(head)}"].append(image_retrieval_score)
                                # else:
                                if index in pos_attn_position:
                                    features[f"l{str(layer)}_h{str(head)}"].append(image_retrieval_score)
                                    
                                else:
                                    features[f"l{str(layer)}_h{str(head)}"].append(0)
                                
                                    
                        else:
                            pos_attn_position = set([weight + start_ids[evidence_page].item() for weight in list(new_weight.keys())])
                            # import pdb;pdb.set_trace()
                            score = 0
                            for ans in ans_idx:
                                index = torch.argmax(attn[ans,:-len(generated_ids_trimmed[0])]).item()            
                                
                                # if answer in output_text:
                                #     features[f"l{str(layer)}_h{str(head)}"].append(image_retrieval_score)
                                # else:
                                if index in pos_attn_position:
                                    features[f"l{str(layer)}_h{str(head)}"].append(image_retrieval_score)
                                else:
                                    features[f"l{str(layer)}_h{str(head)}"].append(0)
            except:
                error.append(number)
                continue
            # if _number == 0:
            #     import pdb;pdb.set_trace()
            # import pdb;pdb.set_trace()
            # if number > 3:
            #     break
        for layer in range(n_layer):
            for head in range(n_heads):
                mean_features[f"l{str(layer)}_h{str(head)}"] = \
                    sum(features[f"l{str(layer)}_h{str(head)}"])/len(features[f"l{str(layer)}_h{str(head)}"])

    agg_feature_save_path = "retrieval_score"
    if os.path.exists(agg_feature_save_path) is False:
        os.mkdir(agg_feature_save_path)
        
    agg_feature_save_path += "/image"

    if os.path.exists(agg_feature_save_path) is False:
        os.mkdir(agg_feature_save_path)

    if args.fine_grained: save_name = agg_feature_save_path+f"/{args.model_id.split('/')[1]}"+"_fg"
    else: save_name = agg_feature_save_path+f"/{args.model_id.split('/')[1]}_cg"

    with open(save_name + "_2.json", 'w', encoding='utf-8') as f:
        json.dump(features, f, ensure_ascii=False, indent=4)
        
    with open(save_name + "_mean_2.json", 'w', encoding='utf-8') as f:
        json.dump(mean_features, f, ensure_ascii=False, indent=4)

    with open(save_name + "_errors_2.json", 'w', encoding='utf-8') as f:
        json.dump(error, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', default='', choices= ["Qwen/Qwen2-VL-2B-Instruct",
                                                            "Qwen/Qwen2-VL-7B-Instruct", 
                                                            "OpenGVLab/InternVL2-8B",])
    parser.add_argument('--norm', action='store_true')
    parser.add_argument('--bos_as_weight', action='store_true')
    parser.add_argument('--do_ablation', action='store_true')
    parser.add_argument('--fine_grained', action='store_true')
    
    args=parser.parse_args()
    main(args)
'''
_experiment1.sh
python step2_image_retrieval_head.py --model_id Qwen/Qwen2-VL-2B-Instruct --fine_grained
python step2_image_retrieval_head.py --model_id Qwen/Qwen2-VL-7B-Instruct --fine_grained
python step2_image_retrieval_head.py --model_id OpenGVLab/InternVL2-8B --fine_grained
'''
