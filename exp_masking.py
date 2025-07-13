#%%
'''
lmms-lab/DocVQA, slideVQA
NQ, hotpotqa
passkey, passkey_img
'''
import torch
from typing import List, Text

from transformers import AutoTokenizer
from ast import literal_eval
import json
import os
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
# %%
class StopOnToken(StoppingCriteria):
    def __init__(self, stop_token_id):
        self.stop_token_id = stop_token_id

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids[0][-1] == self.stop_token_id:
            return True
        return False
    
# %%

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
        
    #%% head load
    if "qwen" in args.model_id.lower():
        model_size = args.model_id.split('-')[2].replace('B','')
    if args.do_masking:
        if args.is_all_random:
        ### original code
            if args.task == 'image':
                if args.fine_grained: head_path = f"retrieval_score/retrieval_head/{args.model_id.split('/')[1]}_fg_2.json" # {args.model_id.split('/')[1]}_fg.json
                else: head_path = f"retrieval_score/retrieval_head/{args.model_id.split('/')[1]}_cg2.json" # {args.model_id.split('/')[1]}_cg.json 
            elif args.task == 'text':
                head_path = f"retrieval_score/retrieval_head/{args.model_id.split('/')[1]}_te_2.json" # {model.split('/')[1]}_te.json
            with open(head_path, 'r') as f:
                attn_mask_dict = json.load(f)
            args.masking_dict = attn_mask_dict

            if args.is_all_random:
                all_random_mask_dict = {}
                for layer in range(n_layer):
                    for head in range(n_heads):
                        all_random_mask_dict[f'l{layer}_h{head}'] = 0
                
                if args.no_intersection:
                    no_int = list(set(all_random_mask_dict) - set(args.masking_dict))
                    masking_key_list = random.sample(no_int, args.num_of_masking_head)
                    
                else:
                    masking_key_list = random.sample(list(all_random_mask_dict), args.num_of_masking_head)
                    
            else:
                masking_key_list = random.sample(list(args.masking_dict), args.num_of_masking_head)

            inut_masking_dict = {}
            if args.do_masking:
                for key in masking_key_list:
                    inut_masking_dict[key] = 0
                
            args.masking_dict = inut_masking_dict
            
        else:

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
            top_k_dict_i = dict(sorted(d.items(), key=lambda x: x[1], reverse=True)[:args.num_of_masking_head])
            masking_key_list = []
            for key, val in top_k_dict_i.items():
                masking_key_list.append(key)
            
            inut_masking_dict = {}
            if args.do_masking:
                for key in masking_key_list:
                    inut_masking_dict[key] = 0
                
            args.masking_dict = inut_masking_dict
            
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    args.device = device
    
    model, processor = lvlm_loader(args.model_id, device = device, args = args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    
    if args.datasets == 'doc': 
        dataset = load_dataset("lmms-lab/DocVQA", "DocVQA",split="validation")
    elif args.datasets == "mdocs":
        dataset = load_dataset("lmms-lab/MP-DocVQA", split="val")
    elif args.datasets == 'textvqa':
        dataset = load_dataset("lmms-lab/textvqa", split="validation")
        
    else:
        if args.datasets == 'hotpot': dataset_path = 'dataset/raw_data/hotpotqa/hotpot_dev_distractor_v1.json' 
        elif args.datasets == 'nq': dataset_path = 'dataset/raw_data/nq/biencoder-nq-dev.json' 
        if args.datasets == 'slide': dataset_path = 'dataset/SlideVQA/annotations/qa/dev.jsonl'
        if 'passkey' in args.datasets:
            if args.fine_grained: dataset_path = "img_dataset_dev/bbox_fg.json"
            else: dataset_path = "img_dataset_dev/bbox.json"
        if args.datasets == 'slide':
            dataset = []
            with jsonlines.open(dataset_path, 'r') as f:
                for line in f.iter():
                    dataset.append(line)
        else:
            with open(dataset_path, 'r') as f:
                dataset = json.load(f)
    
    def extract_evidence_passage_nq(dataset, _number) -> List:
        evidence = [evi['text'] for evi in dataset[_number]['positive_ctxs'] if evi['title_score'] == 1]
        return evidence
    def extract_random_passage_nq(dataset, _number) -> List:
        random_num = random.sample(range(0, len(dataset[_number]['negative_ctxs'])-1), 5)
        neg_evi = [dataset[_number]['negative_ctxs'][ran]['text'] for ran in random_num]
        return neg_evi
    def extract_evidence_passage_hotpot(dataset, _number) -> List:
        evidences = []
        for content in dataset[_number]['context']:
            if content[0] in [facts[0] for facts in dataset[_number]['supporting_facts']]:
                evidences.append(''.join(content[1]))
        return evidences
    def extract_random_passage_hotpot(dataset) -> Text:
        rand_int = random.randint(0, len(dataset)-1)
        n_of_context = len(dataset[rand_int]['context'])
        rand_context_n=random.randint(0, n_of_context-1)
        return ''.join(dataset[rand_int]['context'][rand_context_n][1])
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
        elif datasets == "nq": 
            evidence = extract_evidence_passage_nq(dataset, number)
            neg_evi = extract_random_passage_nq(dataset, number)
            evidence.extend(neg_evi)
            random.shuffle(evidence)
            return dataset[number]['question'], dataset[number]['answers'], None, evidence
        
        elif datasets == "hotpot": 
            num_of_random_passages=random.randint(1,5)
            random_evidence = [extract_random_passage_hotpot(dataset) for _ in range(num_of_random_passages)]
            evidence_and_ran_evi=extract_evidence_passage_hotpot(dataset, number)
            evidence_and_ran_evi.extend(random_evidence)
            random.shuffle(evidence_and_ran_evi)
            return dataset[number]['question'], dataset[number]['answer'], None, evidence_and_ran_evi
        elif datasets == "slide": 
            paths = ["dataset/SlideVQA/images/dev/"+dataset[number]['deck_name'] + '/' +img.split('/')[-1] for img in dataset[number]['image_urls']]        
            return dataset[number]['question'], dataset[number]['answer'], paths, dataset[number]['evidence_pages'][0]-1
        elif (datasets == "doc") or (args.datasets == 'textvqa'): 
            return dataset[number]['question'], dataset[number]['answers'], dataset[number]['image'], None
        elif datasets == 'mdocs':
            data_type = type(dataset[0]['image_1'])

            image_range=range(1,20)
            image_list = [f"image_{num}" for num in image_range if isinstance(dataset[number][f'image_{num}'], data_type)]

            q = dataset[number]['question']
            a = dataset[number]['answers']
            evidence = dataset[number]['answer_page_idx']
            return q, a, image_list, evidence
    if args.datasets == 'slide':
        n_iter = []
        count = 0
        for num in range(len(dataset)):
            if dataset[num]['evidence_pages'][0]-1 > 7:
                continue
            else:
                n_iter.append(num)
                count += 1
            if count +1 > 600:
                break
    else: n_iter = range(1200)
    
    if (args.datasets == "hotpot") or (args.datasets == "slide"):
        metric = SupportEmF1Metric()    
    else: 
        metric = EmF1Metric()
    pred_count = 0
    preds, golds = [], []
    for n, number in enumerate(n_iter):
        try:
            q, a, img_path, evi = load_data(dataset, number, args.datasets,fg = args.fine_grained)
            
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
                elif args.datasets == 'slide':
                    for path in img_path[:8]:
                        input_img.append({"type": "image", "image": f'''{path}'''})
                    input_content = input_img + \
                        [{"type": "text", "text": f"Question: {q}\n"}] +\
                        [{"type": "text", "text": f"Answer: "}]
                elif (args.datasets == 'doc') or (args.datasets == 'textvqa'):
                    img_path.save(f'tmp{str(args.tmp)}/tmp.jpg')
                    input_img.append({"type": "image", "image": f'tmp{str(args.tmp)}/tmp.jpg'})
                    input_content = input_img + \
                        [{"type": "text", "text": f"Question: {q}\n"}] +\
                        [{"type": "text", "text": f"Answer: "}]
                        
                elif args.datasets == 'mdocs':
                    if len(img_path) > 8:
                        continue
                    for num, img in enumerate(img_path):
                        dataset[number][img].save(f'tmp{str(args.tmp)}/tmp{num}.jpg')
                        input_img.append({"type": "image", "image": f'tmp{str(args.tmp)}/tmp{num}.jpg'})
                    input_content = input_img + \
                        [{"type": "text", "text": f"Question: {q}\n"}] +\
                        [{"type": "text", "text": f"Answer: "}]
                    
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
                    img_path.save('tmp.jpg')
                    img_path = "tmp.jpg"
                    image_tokens = IMG_START_TOKEN + (IMG_CONTEXT_TOKEN * (num_image_token * num_patches)) + IMG_END_TOKEN
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
                    user_question = f"Question: {q}\nAnswer: "
                    
                prompt = f"{image_tokens}\n{user_question}" 
                model_inputs = tokenizer(prompt, return_tensors="pt")
                input_ids = model_inputs["input_ids"].to(device)
                attention_mask = model_inputs["attention_mask"].to(device)
                
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
                        pixel_values = load_image(img_path, max_num = 1).to(torch.float16).to(device)
                    
                    elif args.datasets == 'mdocs':
                        pixel_values = torch.cat([load_image(path, max_num = 1).to(torch.float16).to(device) for path in img_path2], dim = 0)
                    else:
                        pixel_values = torch.cat([load_image(path, max_num = 1).to(torch.float16).to(device) for path in img_path], dim = 0)
                        
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
            if "qwen" in args.model_id.lower():
                preds.append(output_text)
            else:
                preds.append([output_text])
            golds.append(a)
            pred_count += 1
        except: continue
        
        
        if pred_count == 1:
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
            if (args.datasets == "hotpot") or (args.datasets == "slide"):
                metric(pred, [gold]) # [], []
            elif args.datasets == 'mdocs':
                metric(pred, literal_eval(gold))
            else:
                metric(pred, gold)
            
        score = metric.get_metric()
    
    print(score)
    
    save_path = "results2/"

    if os.path.exists(save_path): pass
    else: os.mkdir(save_path)

    
    if args.do_masking: 
        save_path += "mask_"
        
        if args.task == 'image':
            if args.fine_grained: save_path += "fg_"
            else: save_path += "cg_"
        else:
            save_path += 'te_'
            
    else: save_path += "no_"
    
    save_path += f"sd{str(args.seed)}_"
    
    if args.is_all_random: save_path += "all_random_"
    
    print(f"{args.model_id}, {str(args.num_of_masking_head)}")

    
    save_path += f"{str(args.num_of_masking_head)}_{args.task}_{args.datasets}_{args.model_id.split('/')[1]}.json"
    print(save_path)
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="", choices = ["passkey_text", "passkey_image", "nq", "hotpot", "doc", "mdocs"])
    parser.add_argument("--task", default="text", choices=['image', 'text'])
    parser.add_argument("--model_id", default="", choices= ["OpenGVLab/InternVL2-8B", "Qwen/Qwen2-VL-7B-Instruct", "Qwen/Qwen2-VL-2B-Instruct"])
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
    parser.add_argument("--seed", default=1234, type = int, choices = [42, 48, 4242, 22, 1234])
    
    args = parser.parse_args()
    main(args)
'''
python exp_masking.py --datasets passkey_text --model_id OpenGVLab/InternVL2-8B
python exp_masking.py --datasets passkey_text --model_id Qwen/Qwen2-VL-7B-Instruct
python exp_masking.py --datasets passkey_image --model_id OpenGVLab/InternVL2-8B
python exp_masking.py --datasets passkey_image --model_id Qwen/Qwen2-VL-7B-Instruct
python exp_masking.py --datasets nq --model_id OpenGVLab/InternVL2-8B
python exp_masking.py --datasets nq --model_id Qwen/Qwen2-VL-7B-Instruct
python exp_masking.py --datasets hotpot --model_id OpenGVLab/InternVL2-8B
python exp_masking.py --datasets hotpot --model_id Qwen/Qwen2-VL-7B-Instruct
python exp_masking.py --datasets slide --model_id OpenGVLab/InternVL2-8B
python exp_masking.py --datasets slide --model_id Qwen/Qwen2-VL-7B-Instruct
python exp_masking.py --datasets doc --model_id Qwen/Qwen2-VL-7B-Instruct
python exp_masking.py --datasets doc --model_id OpenGVLab/InternVL2-8B

python exp_masking.py --datasets mdocs --task image --model_id Qwen/Qwen2-VL-7B-Instruct --num_of_masking_head 10 --do_masking --fine_grained
python exp_masking.py --datasets mdocs --task image --model_id OpenGVLab/InternVL2-8B --num_of_masking_head 10 --do_masking --fine_grained
python exp_masking.py --datasets mdocs --task image --model_id Qwen/Qwen2-VL-2B-Instruct --num_of_masking_head 10 --do_masking --fine_grained
python exp_masking.py --datasets doc --task image --model_id Qwen/Qwen2-VL-7B-Instruct --num_of_masking_head 10 --do_masking --fine_grained
python exp_masking.py --datasets doc --task image --model_id OpenGVLab/InternVL2-8B --num_of_masking_head 10 --do_masking --fine_grained
python exp_masking.py --datasets doc --task image --model_id Qwen/Qwen2-VL-2B-Instruct --num_of_masking_head 10 --do_masking --fine_grained




python exp_masking.py --datasets nq --model_id OpenGVLab/InternVL2-8B
python exp_masking.py --datasets hotpot --model_id OpenGVLab/InternVL2-8B
python exp_masking.py --datasets mdocs --model_id OpenGVLab/InternVL2-8B
python exp_masking.py --datasets doc --model_id OpenGVLab/InternVL2-8B

python exp_masking.py --datasets doc --task image --model_id Qwen/Qwen2-VL-7B-Instruct --num_of_masking_head 10 --do_masking --fine_grained
python exp_masking.py --datasets slide --task image --model_id Qwen/Qwen2-VL-7B-Instruct --num_of_masking_head 10 --do_masking --fine_grained
python exp_masking.py --datasets slide --task image --model_id Qwen/Qwen2-VL-7B-Instruct --num_of_masking_head 20 --do_masking --fine_grained
python exp_masking.py --datasets doc --task image --model_id Qwen/Qwen2-VL-7B-Instruct --num_of_masking_head 20 --do_masking --fine_grained

python exp_masking.py --datasets doc --task image --model_id Qwen/Qwen2-VL-2B-Instruct --num_of_masking_head 10 --do_masking --fine_grained
python exp_masking.py --datasets doc --task image --model_id OpenGVLab/InternVL2-8B --num_of_masking_head 10 --do_masking --fine_grained

python exp_masking.py --datasets nq --task text --model_id OpenGVLab/InternVL2-8B --num_of_masking_head 10 --do_masking --fine_grained
python exp_masking.py --datasets nq --task text --model_id OpenGVLab/InternVL2-8B --num_of_masking_head 10 --do_masking --is_all_random
python exp_masking.py --datasets nq --task text --model_id OpenGVLab/InternVL2-8B --num_of_masking_head 20 --do_masking --fine_grained
python exp_masking.py --datasets nq --task text --model_id OpenGVLab/InternVL2-8B --num_of_masking_head 20 --do_masking --is_all_random
python exp_masking.py --datasets nq --task image --model_id OpenGVLab/InternVL2-8B --num_of_masking_head 10 --do_masking --fine_grained
python exp_masking.py --datasets nq --task image --model_id OpenGVLab/InternVL2-8B --num_of_masking_head 10 --do_masking --is_all_random
python exp_masking.py --datasets nq --task image --model_id OpenGVLab/InternVL2-8B --num_of_masking_head 20 --do_masking --fine_grained
python exp_masking.py --datasets nq --task image --model_id OpenGVLab/InternVL2-8B --num_of_masking_head 20 --do_masking --is_all_random
python exp_masking.py --datasets nq --task text --model_id Qwen/Qwen2-VL-2B-Instruct --num_of_masking_head 10 --do_masking --fine_grained
python exp_masking.py --datasets nq --task text --model_id Qwen/Qwen2-VL-2B-Instruct --num_of_masking_head 10 --do_masking --is_all_random
python exp_masking.py --datasets nq --task text --model_id Qwen/Qwen2-VL-2B-Instruct --num_of_masking_head 20 --do_masking --fine_grained
python exp_masking.py --datasets nq --task text --model_id Qwen/Qwen2-VL-2B-Instruct --num_of_masking_head 20 --do_masking --is_all_random
python exp_masking.py --datasets nq --task image --model_id Qwen/Qwen2-VL-2B-Instruct --num_of_masking_head 10 --do_masking --fine_grained
python exp_masking.py --datasets nq --task image --model_id Qwen/Qwen2-VL-2B-Instruct --num_of_masking_head 10 --do_masking --is_all_random
python exp_masking.py --datasets nq --task image --model_id Qwen/Qwen2-VL-2B-Instruct --num_of_masking_head 20 --do_masking --fine_grained
python exp_masking.py --datasets nq --task image --model_id Qwen/Qwen2-VL-2B-Instruct --num_of_masking_head 20 --do_masking --is_all_random
python exp_masking.py --datasets nq --task text --model_id Qwen/Qwen2-VL-7B-Instruct --num_of_masking_head 10 --do_masking --fine_grained
python exp_masking.py --datasets nq --task text --model_id Qwen/Qwen2-VL-7B-Instruct --num_of_masking_head 10 --do_masking --is_all_random
python exp_masking.py --datasets nq --task text --model_id Qwen/Qwen2-VL-7B-Instruct --num_of_masking_head 20 --do_masking --fine_grained
python exp_masking.py --datasets nq --task text --model_id Qwen/Qwen2-VL-7B-Instruct --num_of_masking_head 20 --do_masking --is_all_random
python exp_masking.py --datasets nq --task image --model_id Qwen/Qwen2-VL-7B-Instruct --num_of_masking_head 10 --do_masking --fine_grained
python exp_masking.py --datasets nq --task image --model_id Qwen/Qwen2-VL-7B-Instruct --num_of_masking_head 10 --do_masking --is_all_random
python exp_masking.py --datasets nq --task image --model_id Qwen/Qwen2-VL-7B-Instruct --num_of_masking_head 20 --do_masking --fine_grained
python exp_masking.py --datasets nq --task image --model_id Qwen/Qwen2-VL-7B-Instruct --num_of_masking_head 20 --do_masking --is_all_random

python exp_masking.py --datasets hotpot --task text --model_id OpenGVLab/InternVL2-8B --num_of_masking_head 10 --do_masking --fine_grained
python exp_masking.py --datasets hotpot --task text --model_id OpenGVLab/InternVL2-8B --num_of_masking_head 10 --do_masking --is_all_random
python exp_masking.py --datasets hotpot --task text --model_id OpenGVLab/InternVL2-8B --num_of_masking_head 20 --do_masking --fine_grained
python exp_masking.py --datasets hotpot --task text --model_id OpenGVLab/InternVL2-8B --num_of_masking_head 20 --do_masking --is_all_random
python exp_masking.py --datasets hotpot --task image --model_id OpenGVLab/InternVL2-8B --num_of_masking_head 10 --do_masking --fine_grained
python exp_masking.py --datasets hotpot --task image --model_id OpenGVLab/InternVL2-8B --num_of_masking_head 10 --do_masking --is_all_random
python exp_masking.py --datasets hotpot --task image --model_id OpenGVLab/InternVL2-8B --num_of_masking_head 20 --do_masking --fine_grained
python exp_masking.py --datasets hotpot --task image --model_id OpenGVLab/InternVL2-8B --num_of_masking_head 20 --do_masking --is_all_random
python exp_masking.py --datasets hotpot --task text --model_id Qwen/Qwen2-VL-2B-Instruct --num_of_masking_head 10 --do_masking --fine_grained
python exp_masking.py --datasets hotpot --task text --model_id Qwen/Qwen2-VL-2B-Instruct --num_of_masking_head 10 --do_masking --is_all_random
python exp_masking.py --datasets hotpot --task text --model_id Qwen/Qwen2-VL-2B-Instruct --num_of_masking_head 20 --do_masking --fine_grained
python exp_masking.py --datasets hotpot --task text --model_id Qwen/Qwen2-VL-2B-Instruct --num_of_masking_head 20 --do_masking --is_all_random
python exp_masking.py --datasets hotpot --task image --model_id Qwen/Qwen2-VL-2B-Instruct --num_of_masking_head 10 --do_masking --fine_grained
python exp_masking.py --datasets hotpot --task image --model_id Qwen/Qwen2-VL-2B-Instruct --num_of_masking_head 10 --do_masking --is_all_random
python exp_masking.py --datasets hotpot --task image --model_id Qwen/Qwen2-VL-2B-Instruct --num_of_masking_head 20 --do_masking --fine_grained
python exp_masking.py --datasets hotpot --task image --model_id Qwen/Qwen2-VL-2B-Instruct --num_of_masking_head 20 --do_masking --is_all_random
python exp_masking.py --datasets hotpot --task text --model_id Qwen/Qwen2-VL-7B-Instruct --num_of_masking_head 10 --do_masking --fine_grained
python exp_masking.py --datasets hotpot --task text --model_id Qwen/Qwen2-VL-7B-Instruct --num_of_masking_head 10 --do_masking --is_all_random
python exp_masking.py --datasets hotpot --task text --model_id Qwen/Qwen2-VL-7B-Instruct --num_of_masking_head 20 --do_masking --fine_grained
python exp_masking.py --datasets hotpot --task text --model_id Qwen/Qwen2-VL-7B-Instruct --num_of_masking_head 20 --do_masking --is_all_random
python exp_masking.py --datasets hotpot --task image --model_id Qwen/Qwen2-VL-7B-Instruct --num_of_masking_head 10 --do_masking --fine_grained
python exp_masking.py --datasets hotpot --task image --model_id Qwen/Qwen2-VL-7B-Instruct --num_of_masking_head 10 --do_masking --is_all_random
python exp_masking.py --datasets hotpot --task image --model_id Qwen/Qwen2-VL-7B-Instruct --num_of_masking_head 20 --do_masking --fine_grained
python exp_masking.py --datasets hotpot --task image --model_id Qwen/Qwen2-VL-7B-Instruct --num_of_masking_head 20 --do_masking --is_all_random

python exp_masking.py --datasets slide --task text --model_id OpenGVLab/InternVL2-8B --num_of_masking_head 10 --do_masking --fine_grained
python exp_masking.py --datasets slide --task text --model_id OpenGVLab/InternVL2-8B --num_of_masking_head 10 --do_masking --is_all_random
python exp_masking.py --datasets slide --task text --model_id OpenGVLab/InternVL2-8B --num_of_masking_head 20 --do_masking --fine_grained
python exp_masking.py --datasets slide --task text --model_id OpenGVLab/InternVL2-8B --num_of_masking_head 20 --do_masking --is_all_random
python exp_masking.py --datasets slide --task image --model_id OpenGVLab/InternVL2-8B --num_of_masking_head 10 --do_masking --fine_grained
python exp_masking.py --datasets slide --task image --model_id OpenGVLab/InternVL2-8B --num_of_masking_head 10 --do_masking --is_all_random
python exp_masking.py --datasets slide --task image --model_id OpenGVLab/InternVL2-8B --num_of_masking_head 20 --do_masking --fine_grained
python exp_masking.py --datasets slide --task image --model_id OpenGVLab/InternVL2-8B --num_of_masking_head 20 --do_masking --is_all_random
python exp_masking.py --datasets slide --task text --model_id Qwen/Qwen2-VL-2B-Instruct --num_of_masking_head 10 --do_masking --fine_grained
python exp_masking.py --datasets slide --task text --model_id Qwen/Qwen2-VL-2B-Instruct --num_of_masking_head 10 --do_masking --is_all_random
python exp_masking.py --datasets slide --task text --model_id Qwen/Qwen2-VL-2B-Instruct --num_of_masking_head 20 --do_masking --fine_grained
python exp_masking.py --datasets slide --task text --model_id Qwen/Qwen2-VL-2B-Instruct --num_of_masking_head 20 --do_masking --is_all_random
python exp_masking.py --datasets slide --task image --model_id Qwen/Qwen2-VL-2B-Instruct --num_of_masking_head 10 --do_masking --fine_grained
python exp_masking.py --datasets slide --task image --model_id Qwen/Qwen2-VL-2B-Instruct --num_of_masking_head 10 --do_masking --is_all_random
python exp_masking.py --datasets slide --task image --model_id Qwen/Qwen2-VL-2B-Instruct --num_of_masking_head 20 --do_masking --fine_grained
python exp_masking.py --datasets slide --task image --model_id Qwen/Qwen2-VL-2B-Instruct --num_of_masking_head 20 --do_masking --is_all_random
python exp_masking.py --datasets slide --task text --model_id Qwen/Qwen2-VL-7B-Instruct --num_of_masking_head 10 --do_masking --fine_grained
python exp_masking.py --datasets slide --task text --model_id Qwen/Qwen2-VL-7B-Instruct --num_of_masking_head 10 --do_masking --is_all_random
python exp_masking.py --datasets slide --task text --model_id Qwen/Qwen2-VL-7B-Instruct --num_of_masking_head 20 --do_masking --fine_grained
python exp_masking.py --datasets slide --task text --model_id Qwen/Qwen2-VL-7B-Instruct --num_of_masking_head 20 --do_masking --is_all_random
python exp_masking.py --datasets slide --task image --model_id Qwen/Qwen2-VL-7B-Instruct --num_of_masking_head 10 --do_masking --fine_grained
python exp_masking.py --datasets slide --task image --model_id Qwen/Qwen2-VL-7B-Instruct --num_of_masking_head 10 --do_masking --is_all_random
python exp_masking.py --datasets slide --task image --model_id Qwen/Qwen2-VL-7B-Instruct --num_of_masking_head 20 --do_masking --fine_grained
python exp_masking.py --datasets slide --task image --model_id Qwen/Qwen2-VL-7B-Instruct --num_of_masking_head 20 --do_masking --is_all_random

python exp_masking.py --datasets doc --task text --model_id OpenGVLab/InternVL2-8B --num_of_masking_head 10 --do_masking --fine_grained
python exp_masking.py --datasets doc --task text --model_id OpenGVLab/InternVL2-8B --num_of_masking_head 10 --do_masking --is_all_random
python exp_masking.py --datasets doc --task text --model_id OpenGVLab/InternVL2-8B --num_of_masking_head 20 --do_masking --fine_grained
python exp_masking.py --datasets doc --task text --model_id OpenGVLab/InternVL2-8B --num_of_masking_head 20 --do_masking --is_all_random
python exp_masking.py --datasets doc --task image --model_id OpenGVLab/InternVL2-8B --num_of_masking_head 10 --do_masking --fine_grained
python exp_masking.py --datasets doc --task image --model_id OpenGVLab/InternVL2-8B --num_of_masking_head 10 --do_masking --is_all_random
python exp_masking.py --datasets doc --task image --model_id OpenGVLab/InternVL2-8B --num_of_masking_head 20 --do_masking --fine_grained
python exp_masking.py --datasets doc --task image --model_id OpenGVLab/InternVL2-8B --num_of_masking_head 20 --do_masking --is_all_random
python exp_masking.py --datasets doc --task text --model_id Qwen/Qwen2-VL-2B-Instruct --num_of_masking_head 10 --do_masking --fine_grained
python exp_masking.py --datasets doc --task text --model_id Qwen/Qwen2-VL-2B-Instruct --num_of_masking_head 10 --do_masking --is_all_random
python exp_masking.py --datasets doc --task text --model_id Qwen/Qwen2-VL-2B-Instruct --num_of_masking_head 20 --do_masking --fine_grained
python exp_masking.py --datasets doc --task text --model_id Qwen/Qwen2-VL-2B-Instruct --num_of_masking_head 20 --do_masking --is_all_random
python exp_masking.py --datasets doc --task image --model_id Qwen/Qwen2-VL-2B-Instruct --num_of_masking_head 10 --do_masking --fine_grained
python exp_masking.py --datasets doc --task image --model_id Qwen/Qwen2-VL-2B-Instruct --num_of_masking_head 10 --do_masking --is_all_random
python exp_masking.py --datasets doc --task image --model_id Qwen/Qwen2-VL-2B-Instruct --num_of_masking_head 20 --do_masking --fine_grained
python exp_masking.py --datasets doc --task image --model_id Qwen/Qwen2-VL-2B-Instruct --num_of_masking_head 20 --do_masking --is_all_random
python exp_masking.py --datasets doc --task text --model_id Qwen/Qwen2-VL-7B-Instruct --num_of_masking_head 10 --do_masking --fine_grained
python exp_masking.py --datasets doc --task text --model_id Qwen/Qwen2-VL-7B-Instruct --num_of_masking_head 10 --do_masking --is_all_random
python exp_masking.py --datasets doc --task text --model_id Qwen/Qwen2-VL-7B-Instruct --num_of_masking_head 20 --do_masking --fine_grained
python exp_masking.py --datasets doc --task text --model_id Qwen/Qwen2-VL-7B-Instruct --num_of_masking_head 20 --do_masking --is_all_random
python exp_masking.py --datasets doc --task image --model_id Qwen/Qwen2-VL-7B-Instruct --num_of_masking_head 10 --do_masking --fine_grained
python exp_masking.py --datasets doc --task image --model_id Qwen/Qwen2-VL-7B-Instruct --num_of_masking_head 10 --do_masking --is_all_random
python exp_masking.py --datasets doc --task image --model_id Qwen/Qwen2-VL-7B-Instruct --num_of_masking_head 20 --do_masking --fine_grained
python exp_masking.py --datasets doc --task image --model_id Qwen/Qwen2-VL-7B-Instruct --num_of_masking_head 20 --do_masking --is_all_random

'''
#%%