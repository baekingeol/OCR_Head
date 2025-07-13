#%%
'''
lmms-lab/DocVQA, slideVQA
NQ, hotpotqa
passkey, passkey_img
'''
import torch
from typing import List, Text
from ast import literal_eval

from transformers import AutoTokenizer

import json
import os
os.environ["PYTORCH_USE_CUDA_DSA"] = "1"

import random
random.seed(22)

from utils.loader import lvlm_loader, load_dataset
from utils.prompt import prompt
from utils.utils import return_attention, generate, generate_w_masking

from dataclasses import dataclass

from tqdm import tqdm

from glob import glob
import jsonlines
from transformers import AutoConfig
from transformers import GenerationConfig
from transformers import StoppingCriteria, StoppingCriteriaList

from datasets import load_dataset

from utils.metrics import EmF1Metric, SupportEmF1Metric


class StopOnToken(StoppingCriteria):
    def __init__(self, stop_token_id):
        self.stop_token_id = stop_token_id

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids[0][-1] == self.stop_token_id:
            return True
        return False


def main(args):
    
    img_tmp_file_path = f'tmp{str(args.tmp)}/'
    if os.path.exists(img_tmp_file_path): pass
    else: os.mkdir(img_tmp_file_path)
    
    random.seed(args.seed)
    
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    
    if 'intern' in args.model_id.lower():
        n_heads = config.llm_config.num_attention_heads
        n_layer = config.llm_config.num_hidden_layers
    else:
        n_heads = config.num_attention_heads # 10
        n_layer = config.num_hidden_layers # 12
        
    total_layer_head = {}

    p = f"retrieval_score/{args.task}/{args.model_id.split('/')[1]}_"
    if args.task == 'image': p += "fg_2.json"
    else: p += "te_2.json"
    
    paths_ = [p]
    dataset_all= []
    for path in paths_:
        with open(path, 'r') as f:
            data = json.load(f)
        dataset_all.append(data)
    print(paths_)
        
    for model_id in [args.model_id]:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        # import pdb;pdb.set_trace()
        if "internlm" in model_id.lower():
            n_heads = config.num_attention_heads # 10
            n_layer = config.num_hidden_layers # 12
        elif 'intern' in model_id.lower():
            n_heads = config.llm_config.num_attention_heads
            n_layer = config.llm_config.num_hidden_layers
        else:
            n_heads = config.num_attention_heads # 10
            n_layer = config.num_hidden_layers # 12
        total_layer_head[f"{model_id}"] = [n_layer, n_heads]
            
    def retrun_mean_dataset(dataset_all, model_image, total_layer_head = total_layer_head):
        mean_list = []
        for num, data in tqdm(enumerate(dataset_all)):
            layers, heads = total_layer_head[model_image[num]]    
            default_dictionary = {}
            
            for layer in range(layers):
                for head in range(heads):
                    default_dictionary[f'l{layer}_h{head}'] = 0
                    
            for layer in range(layers):
                for head in range(heads):
                    _t = torch.tensor(dataset_all[num][f'l{layer}_h{head}'])
                    if sum(torch.where(_t != 0, 0, 1)) > 0.1 * len(data[f'l{layer}_h{head}']):
                        default_dictionary[f'l{layer}_h{head}'] = sum(data[f'l{layer}_h{head}'])/len(data[f'l{layer}_h{head}'])
                        
                    else:
                        default_dictionary[f'l{layer}_h{head}'] = 0
            mean_list.append(default_dictionary)
        return mean_list
    mean_head_list = retrun_mean_dataset(dataset_all, [f"{args.model_id}"])
    d = mean_head_list[0]
        
    top_k_dict = dict(sorted(d.items(), key=lambda x: x[1], reverse=True)[:args.num_of_masking_head]) 
    args.masking_dict = top_k_dict

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device
    
    model, processor = lvlm_loader(args.model_id, device = device, args = args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    
    #%% dataset load
    if args.datasets == 'doc': 
        dataset = load_dataset("lmms-lab/DocVQA", "DocVQA",split="validation")
    elif args.datasets == "mdocs":
        dataset = load_dataset("lmms-lab/MP-DocVQA", split="val")
    else:
        if args.datasets == 'hotpot': dataset_path = 'dataset/raw_data/hotpotqa/hotpot_dev_distractor_v1.json'
        elif args.datasets == 'nq': dataset_path = 'dataset/raw_data/nq/biencoder-nq-dev.json' 
        
        if 'passkey' in args.datasets:
            if args.fine_grained: dataset_path = "img_dataset_dev/bbox_fg.json"
            else: dataset_path = "img_dataset_dev/bbox.json"

        with open(dataset_path, 'r') as f:
            dataset = json.load(f)

    #%% 
    def load_data(dataset, number, datasets, fg = False):
        if "passkey" in datasets:
            path = 'img_dataset_dev/'
            img_paths = sorted(glob(path + f'{number}/*'))
            evidence_page = int(list(dataset[number][f'{str(number)}'].keys())[0])
            bbox = list(dataset[number][f'{str(number)}'].values())[0]
            answer = dataset[number]["answer"]
            prompt = dataset[number]["prompt"]
            return prompt, answer, img_paths, evidence_page
        elif datasets == "doc": 
            return dataset[number]['question'], dataset[number]['answers'], dataset[number]['image'], None
        elif datasets == 'mdocs':
            data_type = type(dataset[0]['image_1'])

            image_range=range(1,20)
            image_list = [f"image_{num}" for num in image_range if isinstance(dataset[number][f'image_{num}'], data_type)]

            q = dataset[number]['question']
            a = dataset[number]['answers']
            evidence = dataset[number]['answer_page_idx']
            return q, a, image_list, evidence
    n_iter = range(1200)
    
    if args.datasets == "hotpot":
        metric = SupportEmF1Metric()    
    else: 
        metric = EmF1Metric()
    

    pred_count = 0
    preds, golds = [], []
    for n, number in enumerate(n_iter):
        try:
            q, a, img_path, evi = load_data(dataset, number, args.datasets,fg = args.fine_grained)
            prompt = q
   
            if "qwen" in args.model_id.lower():
                input_img = []
                if (args.datasets == 'passkey_text'):
                    input_content = [{"type": "text", "text": f"Question: {q}\n"}] + \
                                    [{"type": "text", "text": f"\nAnswer: "}]
                elif args.datasets == 'passkey_image':
                    for path in img_path:
                        input_img.append({"type": "image", "image": f'''{path}'''})
                    input_content = input_img + \
                                    [{"type": "text", "text": f"What is the pass key? The pass key is "}]
                elif (args.datasets == 'nq') or (args.datasets == 'hotpot'):
                    
                    input_content = [{"type": "text", "text": f"Evidence {num+1}: {evidence}"} for num, evidence in enumerate(evi)] + \
                                    [{"type": "text", "text": f"Just answer the question and do not generate additional word\nQuestion: {q}\n"}] + \
                                    [{"type": "text", "text": f"\nAnswer: "}]

                elif args.datasets == 'doc':
                    img_path.save(f'tmp{str(args.tmp)}/tmp.jpg')
                    input_img.append({"type": "image", "image": f'tmp{str(args.tmp)}/tmp.jpg'})
                    input_content = input_img + \
                        [{"type": "text", "text": f"Question: {q}\n"}] +\
                        [{"type": "text", "text": f"Answer: "}]
                        # [{"type": "text", "text": f"{prompt}"}]
                
                elif args.datasets == 'mdocs':
                    if len(img_path) > 8:
                        continue
                    for num, img in enumerate(img_path):
                        dataset[number][img].save(f'tmp{str(args.tmp)}/tmp{num}.jpg')
                        input_img.append({"type": "image", "image": f'tmp{str(args.tmp)}/tmp{num}.jpg'})
                    input_content = input_img + \
                        [{"type": "text", "text": f"Question: {q}\n"}] +\
                        [{"type": "text", "text": f"Answer: "}]
                        # [{"type": "text", "text": f"{prompt}"}]

                        
                outputs, generated_ids_trimmed, output_text, with_sp_output_text, inputs=generate_w_masking(args, model, tokenizer, processor, input_content)
            else:
                num_image_token = 256
                IMG_START_TOKEN = "<img>"
                IMG_END_TOKEN   = "</img>"
                IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
                
                if (args.datasets == 'passkey_text'):
                    num_patches = 0
                    image_tokens = IMG_START_TOKEN + (IMG_CONTEXT_TOKEN * (num_image_token * num_patches)) + IMG_END_TOKEN
                    user_question = q
                    
                elif args.datasets == 'passkey_image':
                    num_patches = len(img_path)
                    image_tokens = IMG_START_TOKEN + (IMG_CONTEXT_TOKEN * (num_image_token * num_patches)) + IMG_END_TOKEN
                    user_question = "What is the pass key? The pass key is "
                    
                elif (args.datasets == 'nq') or (args.datasets == 'hotpot'):
                    num_patches = 0
                    image_tokens = IMG_START_TOKEN + (IMG_CONTEXT_TOKEN * (num_image_token * num_patches)) + IMG_END_TOKEN
                    user_question = ''.join([f"Evidence {num+1}: {evidence}" for num, evidence in enumerate(evi)]) + \
                        f"Just answer the question and do not generate additional word\nQuestion: {q}\nAnswer: "
                    
                elif args.datasets == 'slide':
                    num_patches = len(img_path[:8])
                    image_tokens = IMG_START_TOKEN + (IMG_CONTEXT_TOKEN * (num_image_token * num_patches)) + IMG_END_TOKEN
                    user_question = f"Question: {q}\nAnswer: "
                    
                elif (args.datasets == 'doc') or (args.datasets == 'textvqa'):
                    num_patches = 1
                    img_path.save(f'tmp{str(args.tmp)}/tmp.jpg')
                    img_path = f"tmp{str(args.tmp)}/tmp.jpg"
                    image_tokens = IMG_START_TOKEN + (IMG_CONTEXT_TOKEN * (num_image_token * num_patches)) + IMG_END_TOKEN
                    user_question = prompt
                    user_question = f"Question: {q}\nAnswer: "
                
                elif args.datasets == 'mdocs':
                    if len(img_path) > 8:
                        continue
                    num_patches = len(img_path)
                    img_path2 = []
                    for num, img in enumerate(img_path):
                        dataset[number][img].save(f'tmp{str(args.tmp)}/tmp{num}.jpg')
                        img_path2.append(f'tmp{str(args.tmp)}/tmp{num}.jpg')
                    image_tokens = IMG_START_TOKEN + (IMG_CONTEXT_TOKEN * (num_image_token * num_patches)) + IMG_END_TOKEN
                    # user_question = prompt
                    user_question = f"Question: {q}\nAnswer: "
                    
                prompt = f"{image_tokens}\n{user_question}" 
                model_inputs = tokenizer(prompt, return_tensors="pt")
                input_ids = model_inputs["input_ids"].to("cuda")
                attention_mask = model_inputs["attention_mask"].to("cuda")
                
                model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)   
                generation_config = GenerationConfig(
                    max_new_tokens=32,
                    do_sample=False,
                )         
                stop_token_id = 92542 
                stopping_criteria = StoppingCriteriaList([StopOnToken(stop_token_id)])
                
                if args.datasets == "doc" or args.datasets == 'mdocs':
                    from utils.utils import load_image
                    if args.datasets == 'doc':
                        pixel_values = load_image(img_path, max_num = 1).to(torch.float16).cuda()
                    elif args.datasets == 'mdocs':
                        pixel_values = torch.cat([load_image(path, max_num = 1).to(torch.float16).cuda() for path in img_path2], dim = 0)
                    else:
                        pixel_values = torch.cat([load_image(path, max_num = 1).to(torch.float16).cuda() for path in img_path], dim = 0)
                      
                    with torch.no_grad():
                        outputs = model.generate(
                            pixel_values=pixel_values,      
                            input_ids=input_ids,   
                            attention_mask=attention_mask,  
                            return_dict_in_generate=True,
                            output_attentions=True,
                            generation_config=generation_config,
                            stopping_criteria=stopping_criteria
                        )
                    
                else:
                    outputs = model.generate(input_ids = input_ids,
                                    attention_mask=attention_mask,  
                                    return_dict_in_generate=True,
                                    output_attentions=True,
                                    generation_config=generation_config,
                                    stopping_criteria=stopping_criteria
                                )
                    
                output_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            prepro_pred = output_text[0]
            if 'qwen' in args.model_id.lower():
                preds.append(output_text)    
            else:
                preds.append([output_text])

            golds.append(a)
            pred_count += 1
        except: continue
            
            
        if pred_count > 599:
            break
    if 'passkey' in args.datasets:
        accuracy = []
        for pred, gold in zip(preds, golds):
            if str(gold) in pred[0]:
                accuracy.append(1)
            else:
                accuracy.append(0)
        score = {"accuracy": sum(accuracy)/len(accuracy)}
    else:
        for pred, gold in zip(preds, golds):
            if args.datasets == "hotpot":
                metric(pred, [gold]) # [], []
            elif args.datasets == 'mdocs':
                metric(pred, literal_eval(gold))
            else:
                metric(pred, gold)
        score = metric.get_metric()
    
    
    # %%
    save_path = "results/"

    if os.path.exists(save_path): pass
    else: os.mkdir(save_path)

    
    if args.do_masking: 
        save_path += "sink_"
        
        if args.task == 'image':
            if args.fine_grained: save_path += "fg_"
            else: save_path += "cg_"
        else:
            save_path += 'te_'
            
    else: save_path += "no_"
    
    save_path += f"sd{str(args.seed)}_"
    
    if args.is_all_random: save_path += "all_random_"
    
    save_path += f"{str(args.num_of_masking_head)}_{args.task}_{args.datasets}_{args.model_id.split('/')[1]}.json"

    
    print(f"{args.model_id}, task: {args.task}, n_head: {str(args.num_of_masking_head)}, beta: {str(args.beta)}")
    print(score)
    # %%
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="", choices = ["doc", 'mdocs'])
    parser.add_argument("--task", default="text", choices=['image', 'text'])
    parser.add_argument("--model_id", default="", choices= ["OpenGVLab/InternVL2-8B", "Qwen/Qwen2-VL-7B-Instruct"])
    parser.add_argument("--do_masking", action='store_true')
    parser.add_argument("--bos_as_weight", action='store_true')
    parser.add_argument("--norm", action='store_true')
    parser.add_argument("--is_all_random", action='store_true')
    parser.add_argument("--no_intersection", action='store_true')
    parser.add_argument("--fine_grained", action='store_true')
    parser.add_argument("--unmasking", action='store_true')
    parser.add_argument("--dev_or_test", default="dev")
    
    parser.add_argument("--num_of_masking_head", default=10, type = int)
    parser.add_argument("--tmp", default=1, type = int)
    parser.add_argument("--beta", default=0.4, type = float)
    parser.add_argument("--seed", default=42, type = int, choices = [42, 22, 1234, 4321, 2424])
    
    args = parser.parse_args()
    main(args)
'''
python exp_sink.py --datasets doc --task image --tmp 1 --model_id Qwen/Qwen2-VL-7B-Instruct --num_of_masking_head 2 --do_masking --unmasking --beta 0.4
python exp_sink.py --datasets mdocs --task image --tmp 1 --model_id Qwen/Qwen2-VL-7B-Instruct --num_of_masking_head 2 --do_masking --unmasking --beta 0.4
python exp_sink.py --datasets doc --task image --tmp 1 --model_id OpenGVLab/InternVL2-8B --num_of_masking_head 2 --do_masking --unmasking --beta 0.4
python exp_sink.py --datasets mdocs --task image --tmp 1 --model_id OpenGVLab/InternVL2-8B --num_of_masking_head 2 --do_masking --unmasking --beta 0.4
'''