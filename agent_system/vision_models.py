import argparse
import torch
import os
import math
from tqdm import tqdm
import json
from decord import VideoReader, cpu
import numpy as np
from transformers import AutoConfig
import abc
from configs import config
from typing import List, Union
import contextlib
import timeit
import time
import re
from util import HiddenPrints
import torch.backends.cudnn as cudnn
from omegaconf import OmegaConf
import cv2
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import copy
from argparse import ArgumentParser
import torch.nn.functional as F
import torchvision
from collections import Counter
import string
import random



device = "cuda" if torch.cuda.is_available() else "cpu"



class BaseModel(abc.ABC):
    requires_gpu = True
    num_gpus = 1
    to_batch = False
    load_order = 0

    def __init__(self, gpu_number):
        self.dev = f'cuda:{gpu_number}' if device == 'cuda' else device
    
    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @classmethod
    @abc.abstractmethod
    def name(cls):
        pass

    @classmethod
    def list_processes(cls):
        return [cls.name]
    

class Owlv2Model(BaseModel):
    name = 'owlv2'
    
    def __init__(self, gpu_number=0, threshold=config.detect_thresholds.owlv2):
        super().__init__(gpu_number)
        from transformers import Owlv2Processor, Owlv2ForObjectDetection
        if config.owlv2['model_path']:
            model_path = config.owlv2['model_path']
        else:
            model_path = "google/owlv2-base-patch16-ensemble"
        self.processor = Owlv2Processor.from_pretrained(model_path)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_path).cuda()
        self.model.eval()
        self.model.requires_grad_(False)
        self.model = self.model.to(self.dev)
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, image: torch.Tensor, obj: Union[str, list], threshold=0.2):
        if isinstance(obj, str):
            obj = [obj]
        text = ['a photo of a ' + t for t in obj]
        inputs = self.processor(text=text, images=image, return_tensors='pt')
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        target_sizes = torch.tensor([image.shape[1:]]).to(self.model.device)
        results = self.processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=threshold)
        boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]
        # import ipdb; ipdb.set_trace()
        if scores.numel() == 0:
            return []
        else:
            height, width = image.shape[1:]
            indexs = torch.argsort(scores, descending=True)
            boxes = boxes[indexs]
            left = torch.clamp(torch.round(boxes[:, 0]), min=0)
            upper = torch.clamp(torch.round(boxes[:, 1]), min=0)
            right = torch.clamp(torch.round(boxes[:, 2]), max=width)
            lower = torch.clamp(torch.round(boxes[:, 3]), max=height)
            boxes = torch.stack([left, height-lower, right, height-upper], -1).to(boxes.device)
            # boxes = torch.stack(boxes, -1)
            return boxes.cpu()

    
class DeepSeekModel(BaseModel):
    name = 'deepseek'
    requires_gpu = True

    def __init__(self, gpu_number=0):
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        if config.deepseek['model_path']:
            model_path = config.deepseek['model_path']
        else:
            model_path = "deepseek-ai/deepseek-coder-6.7b-instruct"
        with open(config.deepseek['prompt_path']) as f:
            self.base_prompt = f.read().strip()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.pipeline = pipeline("text-generation", model=model_path, tokenizer=self.tokenizer, max_new_tokens=512, device_map="auto", torch_dtype=torch.bfloat16)
        self.tokenizer.pad_token = self.tokenizer.eos_token
            
    @torch.no_grad()
    def forward(self, prompt, possible_answers=None, input_type='image'):
        base_prompt = self.base_prompt

        if isinstance(prompt, str):
            extended_prompt = base_prompt.replace("INSERT_QUERY_HERE", prompt).\
                                        replace('INSERT_POSSIBLE_ANSWERS_HERE', str(possible_answers))
        result = self.forward_(extended_prompt, input_type=input_type)

        return result
    
    def forward_(self, extended_prompt, input_type="image"):
        messages = [
            {"role": "user", "content": "Only answer with a function starting def execute_command!!"},
            {"role": "user", "content": extended_prompt}
        ]
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        outputs = self.pipeline(inputs,return_full_text=False)
        response = outputs[0]['generated_text']
        response = response.replace(f"execute_command({input_type})", f"execute_command({input_type}, time_wait_between_lines, syntax)")
        return response

class UniVTGModel(BaseModel):
    name = 'univtg'

    def __init__(self, gpu_number=0):
        from UniVTG.main.config import TestOptions, setup_model
        from UniVTG.utils.basic_utils import l2_normalize_np_array
        from UniVTG.run_on_video import clip 
        self.clip = clip
        self.l2_normalize_np_array = l2_normalize_np_array
        def load_model():
            opt = TestOptions().parse(config.univtg)
            # pdb.set_trace()
            cudnn.benchmark = True
            cudnn.deterministic = False

            if opt.lr_warmup > 0:
                total_steps = opt.n_epoch
                warmup_steps = opt.lr_warmup if opt.lr_warmup > 1 else int(opt.lr_warmup * total_steps)
                opt.lr_warmup = [warmup_steps, total_steps]

            model, criterion, _, _ = setup_model(opt)
            return model
        model_version = "ViT-B/32"
        clip_len = 1
        self.clip_model, _ = clip.load(model_version, device=gpu_number, jit=False)
        self.vtg_model = load_model()
        self.clip_device = next(self.clip_model.parameters()).device
        self.vtg_device = next(self.vtg_model.parameters()).device

    def vid2clip(self, model, video_clip, output_feat_size=512, half_precision=False):
        from UniVTG.run_on_video.preprocessing import Preprocessing
        preprocess = Preprocessing()
        video = preprocess(video_clip)
        n_chunk = len(video)
        vid_features = torch.zeros(
            (n_chunk, output_feat_size),  
            dtype=torch.float16,  
            device=self.clip_device  
        )
        n_iter = int(math.ceil(n_chunk))
        for i in range(n_iter):
            min_ind = i
            max_ind = (i + 1)
            video_batch = video[min_ind:max_ind].to(self.clip_device)
            batch_features = model.encode_image(video_batch)
            vid_features[min_ind:max_ind] = batch_features
        vid_features = vid_features.cpu().numpy()
        if half_precision:
            vid_features = vid_features.astype('float16')
        return vid_features
    
    def txt2clip(self, model, query):
        encoded_texts = self.clip.tokenize(query).to(self.clip_device)
        text_feature = model.encode_text(encoded_texts)['last_hidden_state']
        valid_lengths = (encoded_texts != 0).sum(1).tolist()[0]
        text_feature = text_feature[0, :valid_lengths].detach().cpu().numpy()
    
        return text_feature
    
    def load_data(self, vid_features, txt_features, clip_len=1):
        vid = vid_features.astype(np.float32)
        txt = txt_features.astype(np.float32)

        vid = torch.from_numpy(self.l2_normalize_np_array(vid))
        txt = torch.from_numpy(self.l2_normalize_np_array(txt))
        clip_len = clip_len * 2
        ctx_l = vid.shape[0]

        timestamp =  ( (torch.arange(0, ctx_l) + clip_len / 2) / ctx_l).unsqueeze(1).repeat(1, 2)

        if True:
            tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
            tef_ed = tef_st + 1.0 / ctx_l
            tef = torch.stack([tef_st, tef_ed], dim=1)  # (Lv, 2)
            vid = torch.cat([vid, tef], dim=1)  # (Lv, Dv+2)

        src_vid = vid.unsqueeze(0).to(self.vtg_device)
        src_txt = txt.unsqueeze(0).to(self.vtg_device)
        src_vid_mask = torch.ones(src_vid.shape[0], src_vid.shape[1]).to(self.vtg_device)
        src_txt_mask = torch.ones(src_txt.shape[0], src_txt.shape[1]).to(self.vtg_device)

        return src_vid, src_txt, src_vid_mask, src_txt_mask, timestamp, ctx_l
    
    def convert_to_hms(self, seconds):
        if seconds < 0:
            seconds = 0
        return time.strftime('%H:%M:%S', time.gmtime(seconds))

    @torch.no_grad()
    def forward(self, video_clip, fps, query, clip_len=1):
        vid_features = self.vid2clip(self.clip_model, video_clip)

        txt_features = self.txt2clip(self.clip_model, query)

        answer = self.forward_(self.vtg_model, fps, vid_features, txt_features, clip_len)
        return answer
    
    @torch.no_grad()
    def forward_(self, model, fps, vid_features, txt_features, clip_len=1):
        src_vid, src_txt, src_vid_mask, src_txt_mask, timestamp, ctx_l = self.load_data(vid_features, txt_features, clip_len)
        src_vid = src_vid.to(self.vtg_device)
        src_txt = src_txt.to(self.vtg_device)
        src_vid_mask = src_vid_mask.to(self.vtg_device)
        src_txt_mask = src_txt_mask.to(self.vtg_device)

        model.eval()
        with torch.no_grad():
            output = model(src_vid=src_vid, src_txt=src_txt, src_vid_mask=src_vid_mask, src_txt_mask=src_txt_mask)
        
        # prepare the model prediction
        pred_logits = output['pred_logits'][0].cpu()
        pred_spans = output['pred_spans'][0].cpu()
        pred_saliency = output['saliency_scores'].cpu()

        # prepare the model prediction
        pred_windows = (pred_spans + timestamp) * ctx_l * clip_len
        pred_confidence = pred_logits
        
        # grounding
        top1_window = pred_windows[torch.argmax(pred_confidence)].tolist()

        mr_res =  " - ".join([self.convert_to_hms(int(i)) for i in top1_window])
        mr_response = f"The Top-1 interval is: {mr_res}"
        time_pattern = r"\d{2}:\d{2}:\d{2}"
        matched_times = re.findall(time_pattern, mr_response) 


        def time_to_frame(time_str, fps):
            h, m, s = map(int, time_str.split(':'))
            second = h * 3600 + m * 60 + s
            frame = int(np.round(second * fps))
            return frame

        pred_start = time_to_frame(matched_times[0], fps)
        pred_end = time_to_frame(matched_times[1], fps)
        
        return [pred_start, pred_end]


class LLaVAVideoModel(BaseModel):
    name = 'llava'

    def __init__(self, gpu_number=0):
        from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
        # import ipdb; ipdb.set_trace()
        if config.llava['model_path']:
            model_path = config.llava['model_path']
        else:
            model_path = "llava-hf/LLaVA-NeXT-Video-7B-DPO-hf"
        self.processor = LlavaNextVideoProcessor.from_pretrained(model_path)

        self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            attn_implementation="flash_attention_2",
        )

    @torch.no_grad()
    def forward(self, video_clip, query=None, possible_answers=None, task='qa'):
        # import ipdb; ipdb.set_trace()
        if video_clip.shape[0] > 32:
            sample_inds = np.linspace(0, video_clip.shape[0]-1, 32, dtype=int)
            clip = video_clip[sample_inds]
        else:
            clip = video_clip

        if task == "summary":
            response = self.summary(clip)
        elif task == "qa":
            response = self.qa(clip, query, possible_answers)
        elif task == "ar":
            response = self.action_recognition(clip)
        else:
            raise ValueError("Wrong task type in LLaVA model!")
        
        return response
    
    @torch.no_grad()
    def inference(self, video_clip, query):
        conversation = [
            {

                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {"type": "video"},
                    ],
            },
        ]

        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs_video = self.processor(text=prompt, videos=video_clip, padding=True, return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(**inputs_video, max_new_tokens=100, do_sample=False, temperature=0.2)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs_video.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        output = output_text[0]

        return output
    @torch.no_grad()
    def qa(self, video_clip, question, possible_answers):
        if possible_answers is not None:
            prompt = f"Question: {question}\nOptions:\n"
            for idx, c in enumerate(possible_answers):
                prompt += f"({chr(ord('A') + idx)}) {c}\n"
            prompt += "Answer with the option\'s letter from the given choices directly and only give the best option."
            output = self.inference(video_clip, prompt)
            num_options = len(possible_answers)
            valid_letters = string.ascii_uppercase[:num_options]
            regex = f"[{valid_letters}]"
            uppercase_letters = re.findall(regex, output)
            if not uppercase_letters:
                answer = random.choice(possible_answers)
            else:
                answer_letter = uppercase_letters[0]
                index = valid_letters.index(answer_letter)
                answer = possible_answers[index]
        else:
            prompt = f"Question: {question}\nAnswer in one word or phrase."
            output = self.inference(video_clip, prompt)
            answer = output


        return answer
        
    
    @torch.no_grad()
    def summary(self, video_clip):
        prompt = "Please provide a one sentence description of the video, focusing on the main subjects, their actions, the background scenes."

        video_summary = self.inference(video_clip, prompt)

        return video_summary
    
    
    @torch.no_grad()
    def action_recognition(self, video_clip):

        with open(config.llava['action_prompt_path']) as f:
            prompt = f.read().strip()

        actions = self.inference(video_clip, prompt)

        return actions
    

class LLaMAModel(BaseModel):
    name = 'llama'
    def __init__(self, gpu_number=0):
        from transformers import pipeline
        from transformers import AutoTokenizer
        if config.llama['model_path']:
            model_path = config.llama['model_path']
        else:
            model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.pipe = pipeline("text-generation", model=model_path, tokenizer=self.tokenizer, max_new_tokens=500, device_map="auto", torch_dtype=torch.bfloat16, eos_token_id=self.tokenizer.eos_token_id, pad_token_id = self.tokenizer.eos_token_id)

        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        self.messages = [
        {"role": "system", "content": "You are a helpful assitant."},
        {"role": "user", "content": "{}"},
        ]

    @torch.no_grad()
    def forward(self, query, task='compare'):
        if task == 'compare':
            response = self.compare(query)
        elif task == 'qa':
            response = self.qa(query)
        
        return response
    
    @torch.no_grad()
    def compare(self, query):
        with open('/remote-home/share/yudishi/foundation-gpt/prompts/compare.prompt') as f:
            verify_prompt = f.read().strip()
        prompt = verify_prompt.replace("INSERT_ANSWERS_HERE", query[0]).replace("INSERT_CANDIDATE_HERE", query[1])
        extended_prompt = [
            {"role": "system", "content": "You are a helpful assitant."},
            {"role": "user", "content": prompt},
        ]
        inputs = self.tokenizer.apply_chat_template(extended_prompt, add_generation_prompt=True, tokenize=False)
        outputs = self.pipe(inputs,return_full_text=False, eos_token_id=self.terminators)
        response = outputs[-1]['generated_text']
        return response
    
    @torch.no_grad()
    def qa(self, query):
        extended_prompt = [
            {"role": "system", "content": "You are a helpful assitant."},
            {"role": "user", "content": query},
        ]
        inputs = self.tokenizer.apply_chat_template(extended_prompt, add_generation_prompt=True, tokenize=False)
        outputs = self.pipe(inputs,return_full_text=False, eos_token_id=self.terminators)
        response = outputs[-1]['generated_text']
        return response
    


class LLaVANexT(BaseModel):
    name = 'llavanext'

    def __init__(self, gpu_number=0):
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
        if config.llavanext['model_path']:
            model_path = config.llavanext['model_path']
        else:
            model_path = "llava-hf/llama3-llava-next-8b-hf"
        self.processor = LlavaNextProcessor.from_pretrained(model_path)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2") 

    @torch.no_grad()
    def forward(self, image, query=None, possible_answers=None, task='qa'):
        img = image.cpu().numpy()
        pil_img = Image.fromarray(img)
        if task == 'caption':
            response = self.caption(pil_img)
        else:
            raise ValueError("LLaVA-NeXT task type wrong!")
        return response

    @torch.no_grad()
    def caption(self, image):
        prompt = "Please provide a one sentence description of the image, focusing on the main subjects, their actions, the background scenes."

        text_output = self.inference(image, prompt)
  

        return text_output
    
    @torch.no_grad()
    def inference(self, image, prompt):
        conversation = [
            {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image"},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.model.device)

        # autoregressively complete prompt
        output = self.model.generate(**inputs, max_new_tokens=100, temperature=0.2)

        response = self.processor.decode(output[0], skip_special_tokens=True)

        return response



    
    

