from tqdm import *
import json

from transformers import pipeline
from transformers import AutoTokenizer
import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", default='your_path/Meta-Llama-3.1-8B-Instruct', help="Where to load LLM.")
    parser.add_argument("--save_dir", required=True, default='cot/STAR_filter.json', help="Where to save the json.")
    parser.add_argument("--prompt_path", default='prompts/cot_filter.prompt', help="Where to load prompt.")
    parser.add_argument("--cot_path", required=True, default='your_path/STAR_transfer.json', help="Where to load untransfered CoT.")

    args = parser.parse_args()


    return args


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    pipe = pipeline("text-generation", model=args.model_path, tokenizer=tokenizer, max_new_tokens=2048, device_map="auto", torch_dtype=torch.bfloat16, eos_token_id=tokenizer.eos_token_id, pad_token_id = tokenizer.eos_token_id)


    terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
    
    cot_datas = json.load(open(args.cot_path))

    with open(args.prompt_path) as f:
        filter_prompt = f.read().strip()
    
    data_list = []
    for idx, cot in enumerate(tqdm(cot_datas)):
        question = cot['question']
        rationale = cot['rationale']

        prompt = filter_prompt.replace("INSERT_QUESTION_HERE", question).replace("INSERT_RATIONALE_HERE", rationale)

        extended_prompt = [
            {"role": "system", "content": "You are a helpful assitant."},
            {"role": "user", "content": prompt},
        ]
        inputs = tokenizer.apply_chat_template(extended_prompt, add_generation_prompt=True, tokenize=False)
        outputs = pipe(inputs,return_full_text=False, eos_token_id=terminators)
        response = outputs[-1]['generated_text']

        if response == 'True':
            data_list.append(cot)
        
    with open(args.save_dir, 'w') as f:
        json.dump(data_list, f)

if __name__ == "__main__":
    args = parse_args()
    main(args)