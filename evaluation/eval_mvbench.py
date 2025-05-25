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

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="eval")

    parser.add_argument("--model_path", required=True, default='ft', help="model version to evaluate.")
    parser.add_argument("--data_dir", required=True, default='your_path/mvbench', help="where to load data.")
    parser.add_argument("--save_dir", required=True, default='your_path/cot_1epoch.json', help="file name to save the eval result.")
    

    args = parser.parse_args()


    return args


def infer_mvbench(processor, model, example):
    video = example['video']
    T, H, W, C = video.shape
    video_input =video
    # video_input = video.reshape(TC//3, H, W, 3).to("cuda:0")
    question_prompt = example['question'] + '\nAnswer with the option\'s letter from the given choices directly and only give the best option.'
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
    return output

def check_ans(pred, gt):
    flag = False
    
    pred_list = pred.lower().split(' ')
    pred_option, pred_content = pred_list[0], ' '.join(pred_list[1:])
    gt_list = gt.lower().split(' ')
    gt_option, gt_content = gt_list[0], ' '.join(gt_list[1:])
    if gt_content[-1] == '.':
        gt_content = gt_content[:-1]
    
    if pred_option.replace('.', '') in gt_option:
        flag = True
    elif gt_option in pred_option:
        flag = True
        
    return flag



class MVBench_dataset(Dataset):
    def __init__(self, data_dir, data_list, num_segments=8, resolution=224):
        self.data_list = []
        for k, v in data_list.items():
            with open(os.path.join(data_dir, v[0]), 'r') as f:
                json_data = json.load(f)
            for data in json_data:
                self.data_list.append({
                    'task_type': k,
                    'prefix': v[1],
                    'data_type': v[2],
                    'bound': v[3],
                    'data': data
                })
        
        self.decord_method = {
            'video': self.read_video,
            'gif': self.read_gif,
            'frame': self.read_frame,
        }
        
        self.num_segments = num_segments
        
        # transform
        # crop_size = resolution
        # scale_size = resolution
        # input_mean = [0.48145466, 0.4578275, 0.40821073]
        # input_std = [0.26862954, 0.26130258, 0.27577711]
        # self.transform = T.Compose([
        #     GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
        #     GroupCenterCrop(crop_size),
        #     Stack(),
        #     ToTorchFormatTensor(),
        #     GroupNormalize(input_mean, input_std) 
        # ])
    
    def __str__(self):
        len_list = {}
        option_list = {}
        for data in self.data_list:
            if data['task_type'] not in len_list:
                len_list[data['task_type']] = 0
            len_list[data['task_type']] += 1
            if data['task_type'] not in option_list:
                option_list[data['task_type']] = 0
            option_list[data['task_type']] += len(data['data']['candidates'])
        
        correct = 0
        total = 0
        res = f"There are {len(self.data_list)} videos as follow:\n"
        for k, v in len_list.items():
            correct += len_list[k]
            total += option_list[k]
            res += f"{v} for {k} ({option_list[k]} options => {len_list[k]/option_list[k]*100:.2f}%)\n"
            correct = correct + 1 / option_list[k]
        res += f"Total random accuracy: {correct/total*100:.2f}%"
        return res.rstrip()
        
    def __len__(self):
        return len(self.data_list)
    
    def get_index(self, bound, fps, max_frame, first_idx=0):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / self.num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(self.num_segments)
        ])
        return frame_indices
    
    def read_video(self, video_path, bound=None):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        
        # images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0) 
        frames = vr.get_batch(frame_indices).asnumpy()
        frames = torch.tensor(frames)
        # for frame_index in frame_indices:
        #     img = Image.fromarray(vr[frame_index].asnumpy())
        #     images_group.append(img)
        # torch_imgs = self.transform(images_group)
        return frames
    
    def read_gif(self, video_path, bound=None, fps=25):
        gif = imageio.get_reader(video_path)
        max_frame = len(gif) - 1
        
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0) 
        for index, frame in enumerate(gif):
            if index in frame_indices:
                img = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                # img = Image.fromarray(img)
                images_group.append(img)
        # torch_imgs = self.transform(images_group)
        frames = torch.stack(images_group, dim=0)
        return frames #TODO: 读取方式
    
    def read_frame(self, video_path, bound=None, fps=3):
        max_frame = len(os.listdir(video_path))
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=1) # frame_idx starts from 1
        for frame_index in frame_indices:
            img = Image.open(os.path.join(video_path, f"{frame_index:05d}.jpg"))
            img = np.array(img)
            img = torch.tensor(img)
            images_group.append(img)
        # torch_imgs = self.transform(images_group)
        frames = torch.stack(images_group, dim=0)
        return frames #TODO: 读取方式

    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        question += "Options:\n"
        answer = data['answer']
        answer_idx = -1
        for idx, c in enumerate(data['candidates']):
            question += f"({chr(ord('A') + idx)}) {c}\n"
            if c == answer:
                answer_idx = idx
        question = question.rstrip()
        answer = f"({chr(ord('A') + answer_idx)}) {answer}"
        return question, answer

    def __getitem__(self, idx):
        decord_method = self.decord_method[self.data_list[idx]['data_type']]
        bound = None
        if self.data_list[idx]['bound']:
            bound = (
                self.data_list[idx]['data']['start'],
                self.data_list[idx]['data']['end'],
            )
        video_path = os.path.join(self.data_list[idx]['prefix'], self.data_list[idx]['data']['video'])
        torch_imgs = decord_method(video_path, bound)
        question, answer = self.qa_template(self.data_list[idx]['data'])
            
        return {
            'video': torch_imgs, 
            'question': question, 
            'answer': answer,
            'task_type': self.data_list[idx]['task_type']
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

    num_frame = 32
    resolution = 336


    data_list = {
        "Action Sequence": ("json_action_sequence.json", f"{args.data_dir}/videos/star/Charades_v1_480/", "video", True), # has start & end
        "Action Prediction": ("json_action_prediction.json", f"{args.data_dir}/videos/star/Charades_v1_480/", "video", True), # has start & end
        "Action Antonym": ("json_action_antonym.json", f"{args.data_dir}/videos/ssv2_video/", "video", False),
        "Fine-grained Action": ("json_fine_grained_action.json", f"{args.data_dir}/videos/Moments_in_Time_Raw/videos/", "video", False),
        "Unexpected Action": ("json_unexpected_action.json", f"{args.data_dir}/videos/FunQA_test/test/", "video", False),
        "Object Existence": ("json_object_existence.json", f"{args.data_dir}/videos/clevrer/video_validation/", "video", False),
        "Object Interaction": ("json_object_interaction.json", f"{args.data_dir}/videos/star/Charades_v1_480/", "video", True), # has start & end
        "Object Shuffle": ("json_object_shuffle.json", f"{args.data_dir}/videos/perception/videos/", "video", False),
        "Moving Direction": ("json_moving_direction.json", f"{args.data_dir}/videos/clevrer/video_validation/", "video", False),
        "Action Localization": ("json_action_localization.json", f"{args.data_dir}/videos/sta/sta_video/", "video", True),  # has start & end
        "Scene Transition": ("json_scene_transition.json", f"{args.data_dir}/videos/scene_qa/video/", "video", False),
        "Action Count": ("json_action_count.json", f"{args.data_dir}/videos/perception/videos/", "video", False),
        "Moving Count": ("json_moving_count.json", f"{args.data_dir}/videos/clevrer/video_validation/", "video", False),
        "Moving Attribute": ("json_moving_attribute.json", f"{args.data_dir}/videos/clevrer/video_validation/", "video", False),
        "State Change": ("json_state_change.json", f"{args.data_dir}/videos/perception/videos/", "video", False),
        "Fine-grained Pose": ("json_fine_grained_pose.json", f"{args.data_dir}/videos/nturgbd/", "video", False),
        "Character Order": ("json_character_order.json", f"{args.data_dir}/videos/perception/videos/", "video", False),
        "Egocentric Navigation": ("json_egocentric_navigation.json", f"{args.data_dir}/videos/vlnqa/", "video", False),
        "Episodic Reasoning": ("json_episodic_reasoning.json", f"{args.data_dir}/videos/tvqa/frames_fps3_hq/", "frame", True),  # has start & end, read frame
        "Counterfactual Inference": ("json_counterfactual_inference.json", f"{args.data_dir}/videos/clevrer/video_validation/", "video", False),
    }

    data_dir = f"{args.data_dir}/json"
    dataset = MVBench_dataset(data_dir, data_list, num_segments=num_frame, resolution=resolution)


    save_path = args.save_dir
    correct = 0
    total = 0
    res_list = []
    acc_dict = {}

    for example in tqdm(dataset):
        video = example['video']
        # import ipdb; ipdb.set_trace()
        task_type = example['task_type']
        if task_type not in acc_dict:
            acc_dict[task_type] = [0, 0] # correct, total
        acc_dict[task_type][1] += 1
        total += 1
        pred = infer_mvbench(processor, model, example)
        gt = example['answer']
        res_list.append({
            'pred': pred,
            'gt': gt
        })
        if check_ans(pred=pred, gt=gt):
            acc_dict[task_type][0] += 1
            correct += 1
        # print(f"Part  Acc: {acc_dict[task_type][0] / acc_dict[task_type][1] * 100 :.2f}%")
    print(f"Total Acc: {correct / total * 100 :.2f}%")
        # print('-' * 30, task_type, '-' * 30)

    with open(f"{save_path}.json", "w") as f:
        json.dump({
            "acc_dict": acc_dict,
            "res_list": res_list
        }, f)




if __name__ == "__main__":
    args = parse_args()
    main(args)