import os
import torch
import numpy as np
import random
import decord
import numpy as np
from decord import cpu, gpu
from tqdm import *
import json
from datasets.star_dataset import STARDataset
from datasets.next_dataset import NEXTDataset
from torch.utils.data import Dataset, DataLoader
from main import *



def get_video(video_path):
    video_reader = decord.VideoReader(str(video_path), num_threads=1, ctx=cpu(0))
    decord.bridge.set_bridge('torch')
    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    frame_idxs = np.linspace(0, vlen, 32, endpoint=False).astype(int)
    video = video_reader.get_batch(frame_idxs).byte()
    clip_len = vlen / (32 * fps)
    video = VideoClip(video, fps=1/clip_len, clip_len=clip_len)

    return video

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def collate_fn(batch):
    question_id = [x['question_id'] for x in batch]
    question = [x['question'] for x in batch]
    video_id = [x['video_id'] for x in batch]
    possible_answers = [x['possible_answers'] for x in batch]
    answer = [x['answer'] for x in batch]

    return question_id, question, video_id, possible_answers, answer


def main():
    setup_seed(20)

    if config.cot['dataset_name'] == 'STAR':
        star_dataset = STARDataset(config.cot['dataset_dir'])
        dataloader = DataLoader(star_dataset, shuffle=False, batch_size=1, num_workers=8, collate_fn=collate_fn)
    elif config.cot['dataset_name'] == 'NEXT_QA':
        next_dataset = NEXTDataset(config.cot['dataset_dir'])
        dataloader = DataLoader(next_dataset, shuffle=False, batch_size=1, num_workers=8, collate_fn=collate_fn)
    else:
        raise ValueError("Wrong dataset name!")

    num_correct = 0
    num_code_wrong = 0

    cot_name = config.cot['save_cot_dir']

    for idx, (question_ids, questions, video_ids, possible_answerss, answers) in enumerate(tqdm(dataloader)):
        question_id = question_ids[0]
        question = questions[0]
        video_id = video_ids[0]
        possible_answers = possible_answerss[0]
        answer = answers[0]
        
        video_path = os.path.join(config.cot['video_dir'], f"{video_id}.mp4")
        video = get_video(video_path)

        code = get_code(question, possible_answers, input_type='video_clip, query, possible_answers')

        if cot_name != "":
            with open(cot_name, "a") as f:
                f.write("\n")
                f.write(f"question id: {question_id}\n")
                f.write(f"video id: {video_id}\n")
                f.write(f"question: {question}\n")
                f.write(f"possible answers: {possible_answers}\n")
                f.write(f"answer: {answer}\n")
                f.write(code[0] + '\n')

        try:
            result = execute_code(code, video, question, possible_answers, show_intermediate_steps=False)
        except:
            num_code_wrong += 1
            continue


    if result == answer:
        num_correct += 1



if __name__ == "__main__":
    main()