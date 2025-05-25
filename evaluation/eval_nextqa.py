from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
import numpy as np
import torch
from decord import VideoReader, gpu, cpu
from tqdm import *
from PIL import Image
import os
import json
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import cv2
import imageio
from torch.utils.data import Dataset
import csv
import codecs

import argparse

def get_video(video_path):
    vr = VideoReader(uri=video_path, ctx=cpu(0)) # you need to install from source to use gpu ctx
    vlen = len(vr)
    indices = np.linspace(0, vlen, 32, endpoint=False).astype(int)
    frames = vr.get_batch(indices).asnumpy()
    return frames

def transfer_answer(result, answer_num):
    result_num = 'random'
    flag = False
    if result[0][0:1] == '(':
        if result[0][1:2] == 'A':
            result_num = '0'
        elif result[0][1:2] == 'B':
            result_num = '1'
        elif result[0][1:2] == 'C':
            result_num = '2'
        elif result[0][1:2] == 'D':
            result_num = '3'
        elif result[0][1:2] == 'E':
            result_num = '4'
    else:
        if result[0][0:1] == 'A':
            result_num = '0'
        elif result[0][0:1] == 'B':
            result_num = '1'
        elif result[0][0:1] == 'C':
            result_num = '2'
        elif result[0][0:1] == 'D':
            result_num = '3'
        elif result[0][0:1] == 'E':
            result_num = '4'
    if result_num == 'random':
        if "A" in result:
            result_num = '0'
        elif "B" in result:
            result_num = '1'
        elif "C" in result:
            result_num = '2'
        elif "D" in result:
            result_num = '3'
        elif "E" in result:
            result_num = '4'
        else:
            import random
            flag = True
            ans_id = random.randint(0,answer_num)
            result_num = str(ans_id)
    
    return result_num

def parse_args():
    parser = argparse.ArgumentParser(description="eval")

    parser.add_argument("--model_path", required=True, default='ft', help="model version to evaluate.")
    parser.add_argument("--data_dir", required=True, default='your_path/val.csv', help="where to load data.")
    parser.add_argument("--video_dir", required=True, default='your_path/NExTVideo', help="where to load video.")
    parser.add_argument("--save_dir", required=True, default='your_path/cot_1epoch.json', help="file name to save the eval result.")
    

    args = parser.parse_args()


    return args


def infer_nextqa(processor, model, example):
    video = example['video']
    T, H, W, C = video.shape
    video_input =video

    question = example['question']
    possible_answers = example['possible_answers']
    question_prompt = "Question: {question}\nOptions:\n(A) {a0}\n(B) {a1}\n(C) {a2}\n(D) {a3}\n(E) {a4}\nAnswer with the option\'s letter from the given choices directly and only give the best option.".format(question=question, a0=possible_answers[0], a1=possible_answers[1], a2=possible_answers[2], a3=possible_answers[3], a4=possible_answers[4])
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question_prompt},
                {"type": "video"},
                ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=prompt, videos=video_input, padding=True, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[-1]
    # video_input = batch['pixel_values_videos'].to(model.device)
    with torch.inference_mode():
        generate_kwargs = {"max_new_tokens": 1024, "do_sample": False, "temperature": 0.01}
        out = model.generate(**inputs, **generate_kwargs)
        output_stripped = out[:, input_length: ]
    output = processor.batch_decode(output_stripped, skip_special_tokens=True)[0]

    result = transfer_answer(output, 4)
    return result



class NEXTQA_dataset(Dataset):
    def __init__(self, data_dir, video_dir):
        self.qa_datas = []
        self.video_dir = video_dir
        with codecs.open(data_dir, encoding='utf-8-sig') as f:
            for i,row in enumerate(csv.DictReader(f, skipinitialspace=True)):
                self.qa_datas.append(row)
        f.close()

        
    def __len__(self):
        return len(self.qa_datas)
    

    def __getitem__(self, idx):
        qa_data = self.qa_datas[idx]

        video_id = qa_data['video']
        video_dir = os.path.join(self.video_dir, f"{video_id}.mp4")
        video_input = get_video(video_dir)

        question = qa_data['question']
        answer = qa_data['answer']
        possible_answers = [qa_data['a0'], qa_data['a1'], qa_data['a2'], qa_data['a3'], qa_data['a4']]
            
        return {
            'video': video_input, 
            'question': question, 
            'answer': answer,
            'possible_answers': possible_answers,
        }

def main(args):
    # import ipdb; ipdb.set_trace()
    processor = LlavaNextVideoProcessor.from_pretrained(args.model_path)
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    dataset = NEXTQA_dataset(args.data_dir, args.video_dir)


    save_path = args.save_dir
    correct = 0
    total = 0
    res_list = []

    for example in tqdm(dataset):

        pred = infer_nextqa(processor, model, example)
        gt = example['answer']
        res_list.append({
            'pred': pred,
            'gt': gt
        })
        if pred == gt:
            correct += 1

        total += 1

    print(f"Total Acc: {correct / total * 100 :.2f}%")


    with open(save_path, "w") as f:
        json.dump({
            "res_list": res_list
        }, f)




if __name__ == "__main__":
    args = parse_args()
    main(args)