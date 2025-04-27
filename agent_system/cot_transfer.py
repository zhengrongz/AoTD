import os

from transformers import pipeline
from transformers import AutoTokenizer
import torch
import re
from tqdm import *
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", default='your_path/Meta-Llama-3.1-8B-Instruct', help="Where to load LLM.")
    parser.add_argument("--save_dir", required=True, default='cot/STAR_transfer.json', help="Where to save the json.")
    parser.add_argument("--prompt_path", default='prompts/cot_transfer.prompt', help="Where to load prompt.")
    parser.add_argument("--cot_path", required=True, default='cot/STAR', help="Where to load untransfered CoT.")

    args = parser.parse_args()


    return args

def parse_input(input_str):
    # 使用正则表达式提取各部分内容
    question_id = re.search(r'id:\s*(.*)', input_str).group(1).strip()
    video_id = re.search(r'video id:\s*(.*)', input_str).group(1).strip()
    question = re.search(r'question:\s*(.*)', input_str).group(1).strip()
    answer = re.search(r'answer:\s*(.*)', input_str).group(1).strip()
    code = re.search(r'def execute_command(.*?return (answer|summary))', input_str, re.DOTALL)
    if code:
        code = code.group(0).strip()
    else:
        code = None
    possible_answers = re.search(r'possible answers:\s*(\[.*\])', input_str).group(1).strip()
    execute_trace = re.search(r'call(.*?)program output:', input_str, re.DOTALL)
    if execute_trace:
        execute_trace = execute_trace.group(1).strip()
    else:
        execute_trace = None

    program_output = re.search(r'program output:\s*(.*)', input_str)
    if program_output:
        program_output = program_output.group(1).strip().strip("'")
    else:
        program_output = None

    # 构造字典
    parsed_dict = {
        'question_id': question_id,
        'video_id': video_id,
        'question': question,
        'answer': answer,
        'code': code,
        'possible_answers': possible_answers,
        'execute_trace': execute_trace,
        'program_output': program_output,
    }
    
    return parsed_dict



def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    pipe = pipeline("text-generation", model=args.model_path, tokenizer=tokenizer, max_new_tokens=2048, device_map="auto", torch_dtype=torch.bfloat16, eos_token_id=tokenizer.eos_token_id, pad_token_id = tokenizer.eos_token_id)

    terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    
    with open(args.prompt_path) as f:
        transfer_prompt = f.read().strip()

    cot_files = sorted(os.listdir(args.cot_path))
    content_paragraphs = []
    for cot_file in cot_files:
        cot_dir = os.path.join(args.cot_path, cot_file)
        with open(cot_dir) as f:
            content = f.read().strip()
        paragraphs = content.split('\n\n')
        paragraphs = [paragraph.strip() for paragraph in paragraphs if paragraph.strip()]
        content_paragraphs.extend(paragraphs)
    

    cot_list = []
    for paragraph in tqdm(content_paragraphs):
        paragraph_dict = parse_input(paragraph)
        if paragraph_dict['program_output'] is None or paragraph_dict['answer'] == 'both right' or paragraph_dict['answer'] == 'both false':
            continue
        if paragraph_dict['answer'] != paragraph_dict['program_output']:
            continue

        prompt = transfer_prompt.replace("INSERT_QUESTION_HERE", paragraph_dict['question']).replace("INSERT_PROGRAM_HERE", paragraph_dict['code']).replace("INSERT_EXECUTION_TRACE_HERE", paragraph_dict['execute_trace'])

        extended_prompt = [
            {"role": "system", "content": "You are a helpful assitant."},
            {"role": "user", "content": prompt},
            ]
        inputs = tokenizer.apply_chat_template(extended_prompt, add_generation_prompt=True, tokenize=False)
        outputs = pipe(inputs,return_full_text=False, eos_token_id=terminators)
        response = outputs[-1]['generated_text']

        match = re.search(r'rationale:\n\n(.*)', response, re.DOTALL)
        # match = re.search(pattern, response)
        if match:
            rationale = match.group(1)
        else:
            rationale = response
        # import ipdb; ipdb.set_trace()
        paragraph_dict['rationale'] = rationale
        cot_list.append(paragraph_dict)

    with open(args.save_dir, 'w') as fp:
        json.dump(cot_list, fp, indent=True)

if __name__ == "__main__":
    args = parse_args()
    main(args)