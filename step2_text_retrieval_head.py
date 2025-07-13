#%%
import argparse
import os
import json
from pprint import pprint

from dataclasses import dataclass
from datasets import load_dataset, concatenate_datasets

import torch

from transformers import AutoTokenizer, AutoConfig
from datasets import load_dataset

from utils.loader import lvlm_loader
from utils.utils import generate, return_attention
from transformers import GenerationConfig
from transformers import StoppingCriteria, StoppingCriteriaList

class StopOnToken(StoppingCriteria):
    def __init__(self, stop_token_id):
        self.stop_token_id = stop_token_id

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids[0][-1] == self.stop_token_id:
            return True
        return False
#%%

#%%

def main(args):
    args.do_masking = False
    args.do_ablation = False
    ds_path = "img_dataset_p/bbox_fg.json"
    with open(ds_path, 'r') as f:
        data = json.load(f)
        
    dataset=load_dataset("nanotron/simple_needle_in_a_hay_stack")
    # %% 1k - 32k, cate: 6 // 150개씩 가져와서 inference 진행 후

    context_lengths = [1024, 2048, 4096, 8192]#, 16384, 32768]

    sampled_subsets = []
    for cl in context_lengths:
        subset = dataset['train'].filter(lambda x: x['context_length'] == cl)
        
        sampled = subset.shuffle(seed=42).select(range(150))
        sampled_subsets.append(sampled)

    f_dataset = concatenate_datasets(sampled_subsets)
    #%%
    # @dataclass
    # class Config():
    #     model_id = '' # Qwen/Qwen2-VL-2B-Instruct Qwen/Qwen2-VL-7B-Instruct deepseek-ai/deepseek-vl2-tiny deepseek-ai/deepseek-vl2-Small
    # args=Config()
    # args.model_id = 'Qwen/Qwen2-VL-7B-Instruct'
    # args.bos_as_weight = False
    # args.norm = False
    # args.do_ablation = False
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, processor = lvlm_loader(args.model_id, device = device, args = args)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    # model.config.use_cache = False
    #%%
    if 'intern' in args.model_id.lower():
        n_heads = model.config.llm_config.num_attention_heads
        n_layer = model.config.llm_config.num_hidden_layers
        q_s, q_i, q_e = 92544, 92546, 92545
    else:
        n_heads = model.config.num_attention_heads # 10
        n_layer = model.config.num_hidden_layers # 12
        q_s, q_i, q_e = 151652, 151655, 151653
    #%%
    features = []
    # number = 485 # 8k 까지만 가능. 
    from tqdm import tqdm
    for nn, final_dataset in enumerate([f_dataset, data]):
        
        for number in tqdm(range(len(final_dataset))):
            try:
                if nn == 0:
                    input_text = final_dataset[number]['prompt'].replace(f"{str(final_dataset[number]['answer'])} is the pass key.", '')    
                else: input_text = final_dataset[number]['prompt'].replace('\n','') + '\n Question: ' + final_dataset[number]['question']
                
                if "qwen" in args.model_id.lower():
                    input_content = [{"type": "text", "text": f"{input_text}"}]
                
                with torch.no_grad():
                    if 'qwen' in args.model_id.lower():
                        outputs, generated_ids_trimmed, output_text, with_sp_output_text, inputs=generate(args, model, tokenizer, processor, input_content)    

                    else: 
                        num_patches = 0
                        num_image_token = model.num_image_token
                        IMG_START_TOKEN = "<img>"
                        IMG_END_TOKEN   = "</img>"
                        IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"

                        image_tokens = IMG_START_TOKEN + (IMG_CONTEXT_TOKEN * (num_image_token * num_patches)) + IMG_END_TOKEN

                        if nn == 0: user_question = final_dataset[number]['prompt'].replace(f"{str(final_dataset[number]['answer'])} is the pass key.", '') + "What is the passkey?"
                        else: user_question = final_dataset[number]['prompt'].replace('\n','') + '\n Question: ' + final_dataset[number]['question']
                        
                        prompt = f"{image_tokens}\n{user_question}" 
                        model_inputs = tokenizer(prompt, return_tensors="pt")
                        input_ids = model_inputs["input_ids"].to("cuda")
                        attention_mask = model_inputs["attention_mask"].to("cuda")

                        model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)   

                        stop_token_id = 281  # 원하는 토큰 id로 변경하세요.
                        stopping_criteria = StoppingCriteriaList([StopOnToken(stop_token_id)])
                        generation_config = GenerationConfig(
                            max_new_tokens=32,
                            do_sample=False,
                        )
                        outputs = model.generate(input_ids = input_ids,
                                        attention_mask=attention_mask,  
                                        return_dict_in_generate=True,
                                        output_attentions=True,
                                        generation_config=generation_config,
                                        stopping_criteria=stopping_criteria
                                    )
                        generated_ids_trimmed=[outputs.sequences.squeeze(dim = 0)]
                        inputs = input_ids
                    # import pdb;pdb.set_trace()
                    
                features.append(final_dataset[number])
                # import pdb;pdb.set_trace()
                # if nn == 1:
                #     number = number + len(features)
                features[-1]['head_value'] = {}
                pattern_list = tokenizer(str(final_dataset[number]['answer']))

                # tmp, context_idx, ans_idx = [], [], []
                # for n, tok in enumerate(outputs.sequences.to('cpu').tolist()[0]):
                #     if tok in pattern_list.input_ids:
                #         if tok in tmp:
                #             ans_idx.append(n)
                #         else:
                #             tmp.append(tok)
                #             context_idx.append(n)
                tmp, context_idx, ans_idx = [], [], []
                for n, tok in enumerate(outputs.sequences.to('cpu').tolist()[0]):
                    if tok in pattern_list.input_ids:
                        ans_idx.append(n)
                        
                
                attentions = outputs['attentions']
                for layer in range(n_layer):
                    for head in range(n_heads):
                        attn = return_attention(args.model_id, attentions, layer = layer, head_n = head, 
                                    norm = args.norm, bos_as_weight= args.bos_as_weight)
                        score = 0
                        for ans in ans_idx:
                            # import pdb;pdb.set_trace()
                            index = torch.argmax(attn[ans,:-len(generated_ids_trimmed[0])]).item()
                            # import pdb;pdb.set_trace()
                            # if (index in context_idx) and (outputs.sequences[0][ans].item() == outputs.sequences[0][index].item()):
                            if outputs.sequences[0][ans].item() == outputs.sequences[0][index].item():
                                score += 1
                        if len(ans_idx) == 0:
                            final_score = 0
                        else:    
                            final_score = score / len(ans_idx)
                        if final_score > 1:
                            continue
                            # import pdb;pdb.set_trace()
                        features[-1]['head_value'][f"l{str(layer)}_h{str(head)}"] = final_score
            except:
                continue
            # import pdb;pdb.set_trace()
            # if number == 10:
            #     break
    #%%
    agg_feature, mean_feature = {}, {}
    for layer in range(n_layer):
        for head in range(n_heads):
            agg_feature[f"l{str(layer)}_h{str(head)}"] = []
            # mean_feature[f"l{str(layer)}_h{str(head)}"] = []
            
    for number in range(len(features)):
        for layer in range(n_layer):
            for head in range(n_heads):
                try:
                    agg_feature[f"l{str(layer)}_h{str(head)}"].append(features[number]['head_value'][f"l{str(layer)}_h{str(head)}"])
                except:
                    continue
    # import pdb;pdb.set_trace()
    agg_feature_save_path = "retrieval_score"

    if os.path.exists(agg_feature_save_path) is False:
        os.mkdir(agg_feature_save_path)

    agg_feature_save_path += "/text"

    if os.path.exists(agg_feature_save_path) is False:
        os.mkdir(agg_feature_save_path)

    with open(agg_feature_save_path+f"/{args.model_id.split('/')[1]}_te_2.json", 'w', encoding='utf-8') as f:
        json.dump(agg_feature, f, ensure_ascii=False, indent=4)

    for layer in range(n_layer):
        for head in range(n_heads):
            mean_feature[f"l{str(layer)}_h{str(head)}"] = sum(agg_feature[f"l{str(layer)}_h{str(head)}"])/len(agg_feature[f"l{str(layer)}_h{str(head)}"])
            
    with open(agg_feature_save_path+f"/{args.model_id.split('/')[1]}_mean_te_2.json", 'w', encoding='utf-8') as f:
        json.dump(mean_feature, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', default='', choices= ["Qwen/Qwen2-VL-2B-Instruct",
                                                            "Qwen/Qwen2-VL-7B-Instruct",
                                                            "OpenGVLab/InternVL2-8B",])
    parser.add_argument('--norm', action='store_true')
    parser.add_argument('--bos_as_weight', action='store_true')
    parser.add_argument('--do_ablation', action='store_true')
    args=parser.parse_args()
    main(args)
'''
_experiment1.sh
python step2_text_retrieval_head.py --model_id Qwen/Qwen2-VL-2B-Instruct
python step2_text_retrieval_head.py --model_id Qwen/Qwen2-VL-7B-Instruct
python step2_text_retrieval_head.py --model_id OpenGVLab/InternVL2-8B
'''
