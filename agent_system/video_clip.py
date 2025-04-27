from video_process import forward
import numpy as np
import torch
from configs import config
from torchvision import transforms
import cv2
import json
from collections import Counter
import os

cot_name = config.cot['save_cot_dir']

class VideoClip:
    def __init__(self, video, start=None, end=None, parent_start=None,fps=30, clip_len=1, trimmed=False):
        if start is None and end is None:
            self.video = video
            self.start = 0
            self.end = video.shape[0]  # duration
        else:
            if start == end:
                self.video = video
                start = 0
                end = video.shape[0]
            else:
                if trimmed:
                    self.video = video[start:end]
                else:
                    self.video = video
                if start is None:
                    start = 0
                if end is None:
                    end = video.shape[0]
            self.start = start + parent_start
            self.end = end + parent_start

        self.num_frames = self.video.shape[0]
        self.fps = fps
        self.clip_len = clip_len

#-------------UNIFY MODULE--------------
def forward_(model_name, *args, **kwargs):
    return forward(model_name, *args, **kwargs)

def Count(video_clip, obj):
    clip = video_clip.video
    result = 0
    if obj == 'people':
        obj = 'person'
    if obj == 'children':
        obj = 'child'
    if cot_name != "":
        with open(cot_name, "a") as f:
            f.write("call function Count\n")
            f.write(f"counting {obj}\n")
    num_objs = 0
    for i in range(clip.shape[0]):
        frame = clip[i]
        frame = frame.permute(2,0,1)
        obj_bboxs = forward_("owlv2", frame, obj, threshold=0.2)
        num_objs = max(num_objs, len(obj_bboxs))
        if cot_name != "":
            with open(cot_name, "a") as f:
                if len(obj_bboxs) != 0:
                    f.write(f"counting {len(obj_bboxs)} {obj} at {obj_bboxs} in frame {i+video_clip.start}\n")
    
    result = num_objs
    
    return result

def Video_query(video_clip, query, possible_answers=None):
    clip = video_clip.video

    answer = forward_('llava', clip, query=query, possible_answers=possible_answers, task='qa')

    if cot_name != "":
        with open(cot_name, "a") as f:
            f.write("call function Video_query\n")
            f.write(f"Question: {query}\n")
            f.write(f"Answer: {answer}\n") 
    return answer


def Query_Actions(video_clip, obj=None, possible_answers=None):
    clip = video_clip.video

    if possible_answers is not None:
        if obj is not None:
            query = f"What did the person do to {obj}?"
            action = forward_("llava", clip, query, possible_answers, task="qa")
        else:
            query = "What did the person do?"
            action = forward_("llava", clip, query, task="qa")
    else:
        if obj is not None:
            query = f"What did the person do to {obj}?"
            action = forward_("llava", clip, query, task="qa")
        else:
            action = forward_("llava", clip, task="ar")

    if cot_name != "":
        with open(cot_name, "a") as f:
            f.write("call function Query_Actions\n")
            if obj is not None:
                f.write(f"Query {obj}\n")
            f.write(f"Answer: {action}\n")
    return action


def Query_Objs(video_clip, obj, possible_answers=None):
    clip = video_clip.video
    query = f"What is {obj}?"

    obj_classes = forward_('llava', clip, query, possible_answers, task='qa')

    if cot_name != "":
        with open(cot_name, "a") as f:
            f.write("call function Query_Objs\n")
            f.write(f"Query {obj}\n")
            f.write(f"Answer: {obj_classes}\n")
    return obj_classes


def Filter_frames_with_act(video_clip, action):
    if isinstance(action, list):
        new_clip = video_clip
        for act in action:
            filtered_start, filtered_end = forward_('univtg', new_clip.video, new_clip.fps, act, new_clip.clip_len)
            filtered_start = max(0, filtered_start)
            filtered_end = min(filtered_end, new_clip.num_frames)
            new_clip = VideoClip(new_clip.video, start=filtered_start, end=filtered_end, parent_start=new_clip.start, fps=new_clip.fps, clip_len=new_clip.clip_len, trimmed=True)
            if cot_name != "":
                with open(cot_name, "a") as f:
                    f.write(f"filter action {act}\n")
                    f.write(f"find action from frame {new_clip.start} to frame {new_clip.end}\n")
    else:
        clip = video_clip.video
        filtered_start, filtered_end = forward_('univtg', clip, video_clip.fps, action, video_clip.clip_len)
        filtered_start = max(0, filtered_start)
        filtered_end = min(filtered_end, video_clip.num_frames)
        new_clip = VideoClip(clip, start=filtered_start, end=filtered_end, parent_start=video_clip.start, fps=video_clip.fps, clip_len=video_clip.clip_len, trimmed=True)
        if cot_name != "":
            with open(cot_name, "a") as f:
                f.write(f"filter action {action}\n")
                f.write(f"find action from frame {new_clip.start} to frame {new_clip.end}\n")
    return new_clip

def Filter_frames_with_obj(video_clip, obj):
    clip = video_clip.video
    frame_list = []
    new_start = None
    new_end = None
    if obj == "people":
        obj = "person"
    if obj == "children":
        obj = "child"
        
    if cot_name != "":
        with open(cot_name, "a") as f:
            f.write("call function Filter_frames_with_obj\n")
            f.write(f"filtered with {obj}\n")
    for i in range(clip.shape[0]):
        frame = clip[i]
        frame = frame.permute(2,0,1)
        obj_bboxes = forward_('owlv2', frame, obj)
        if len(obj_bboxes) == 0:
            continue
        else:
            frame_list.append(frame.permute(1,2,0))
            if len(frame_list) == 0:
                new_start = i
            new_end = i
    if frame_list == []:
        result = video_clip
    else:
        new_clip = torch.stack(frame_list, dim=0)
        result = VideoClip(new_clip, start=new_start, end=new_end, parent_start=video_clip.start, fps=video_clip.fps, clip_len=video_clip.clip_len)
    if cot_name != "":
        with open(cot_name, "a") as f:
            f.write(f"filtered from frame {result.start} to frame {result.end}, {result.num_frames} totally\n")
    return result

def trim(video_clip, start=None, end=None): 
    if start == video_clip.end or end == video_clip.start:
        result = video_clip
    else:
        if start is not None:
            start = max(start - video_clip.start, 0)
        if end is not None:
            end = min(end - video_clip.start, video_clip.num_frames)
        result = VideoClip(video_clip.video, start, end, video_clip.start, video_clip.fps, clip_len=video_clip.clip_len, trimmed=True)
    if cot_name != "":
        with open(cot_name, "a") as f:
            f.write("call function trim\n")
            f.write(f"trimmed video from frame {result.start} to frame {result.end}\n")

    return result

def Filter_else_Actions(actions, current_action):
    if len(actions) <= 1:
        return actions
    else_actions = actions.remove(current_action)

    return else_actions

def Find(video_clip, obj):
    clip = video_clip.video
    frame_list = []
    if obj == "people":
        obj = "person"
    if obj == "children":
        obj = "child"
    if cot_name != "":
        with open(cot_name, "a") as f:
            f.write("call function Find\n")
            f.write(f"finding {obj}\n")
    for i in range(clip.shape[0]):
        frame = clip[i]
        frame = frame.permute(2,0,1)
        obj_bboxs = forward_("owlv2", frame, obj)
        if len(obj_bboxs) == 0:
            continue
        else:
            obj_bboxs = [obj_bboxs[0]]
            if cot_name != "":
                with open(cot_name, 'a') as f:
                    f.write(f"find {obj} at {obj_bboxs} in frame {i+video_clip.start}")
                    f.write('\n')
            cropped_frame = [crop(frame, coord) for coord in obj_bboxs]
            for img in cropped_frame:
                frame_list.append(img.permute(1,2,0))
    
    if frame_list == []:
        if cot_name != "":
            with open(cot_name, 'a') as f:
                f.write(f"Don't find {obj}\n")
        return video_clip
    new_clip = torch.stack(frame_list, dim=0)
    
    new_clip = VideoClip(new_clip, fps=video_clip.fps, clip_len=video_clip.clip_len)

    return new_clip

def select_answer(query, info, possible_answers):
    def format_dict(x):
        if isinstance(x, dict):
            x = ''.join([f'\n\t- {k}: {format_dict(v)}' for k, v in x.items()])
        return x
    with open(config.select_answer_prompt_dir) as f:
        prompt = f.read().strip()

    info_formatting = '\n'.join([f"- {k}: {format_dict(v)}" for k, v in info.items()])
    possible_answers_options = ""
    for idx, c in enumerate(possible_answers):
        possible_answers_options += f"({chr(ord('A') + idx)}) {c}\n"
    prompt = prompt.format(info=info_formatting, question=query, possible_answers=possible_answers_options)

    answer = forward_('llama', prompt, task='qa')
    answer = answer.strip()
    if cot_name != "":
        with open(cot_name, "a") as f:
            f.write("call function select_answer\n")
            f.write(f"the information used: {info_formatting}\n")
            f.write(f"select: {answer}\n")
    return answer

def exist(video_clip, query):
    clip = video_clip.video
    frame_list = []
    if cot_name != "":
        with open(cot_name, "a") as f:
            f.write("call function exist\n")
            f.write(f"finding {query}\n")
    for i in range(clip.shape[0]):
        frame = clip[i]
        frame = frame.permute(2,0,1)
        obj_bboxs = forward_("owlv2", frame, query)
        if len(obj_bboxs) == 0:
            continue
        else:
            obj_bboxs = [obj_bboxs[0]]
            cropped_frame = [crop(frame, coord) for coord in obj_bboxs]
            for img in cropped_frame:
                frame_list.append(img.permute(1,2,0))
    
    if frame_list == []:
        if cot_name != "":
            with open(cot_name, 'a') as f:
                f.write(f"{query} doesn't exist\n")
        return False
    new_clip = torch.stack(frame_list, dim=0)


    question = f"Is this {query}?"
    qa_result = forward_('llava', new_clip, question, task="qa")


    if "yes" in qa_result or "Yes" in qa_result:
        if cot_name != "":
            with open(cot_name, 'a') as f:
                f.write(f"{query} exists\n")
            Find(video_clip, query)
        return True
    else:
        if cot_name != "":
            with open(cot_name, 'a') as f:
                f.write(f"{query} doesn't exist\n")
        return False

def Video_summary(video_clip, query=None):
    clip = video_clip.video
    summary = forward_('llava', clip, query=query, task='summary')
    if cot_name != "":
        with open(cot_name, 'a') as f:
            f.write('call function Video_summary\n')
            f.write(f'summary result: {summary}\n')
    return summary

def Video_caption(video_clip):
    clip = video_clip.video
    if cot_name != "":
        with open(cot_name, 'a') as f:
            f.write('call function Video_caption\n')
    caption_list = []
    for i in range(clip.shape[0]):
        frame = clip[i]
        caption = forward_('llavanext', frame, task='caption')
        if cot_name != "":
            with open(cot_name, 'a') as f:
                f.write(f"frame {i+video_clip.start}: {caption}")
        caption_list.append(f"frame {i}: {caption}")

    
    return caption_list

def Compare(item1, item2):
    answer = forward_('llama', [item1, item2], task='compare')
    return answer.lower()


#-------------SUPPORT MODULE--------------
def remove_duplicates_preserve_order(input_list):
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def crop(frame: torch.Tensor, coordinate: torch.Tensor) -> torch.Tensor:
    left, lower, right, upper = coordinate[:4]
    left = int(left)
    lower = int(lower)
    right = int(right)
    upper = int(upper)

    height = frame.shape[1]
    width = frame.shape[2]

    if config.crop_larger_margin:
        left = max(0, left - 10)
        lower = max(0, lower - 10)
        right = min(width, right + 10)
        upper = min(height, upper + 10)
    cropped_frame = frame[:, frame.shape[1]-upper:frame.shape[1]-lower, left:right]
    resize_transform = transforms.Resize(size=(224, 224))
    cropped_frame = resize_transform(cropped_frame)

    return cropped_frame
