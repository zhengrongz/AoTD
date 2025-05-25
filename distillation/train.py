import numpy as np
from transformers import Trainer, TrainingArguments, Seq2SeqTrainingArguments, DataCollatorForLanguageModeling
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaNextVideoForConditionalGeneration, LlavaNextVideoConfig


import torch
from datasets import load_dataset, concatenate_datasets
from tqdm import *
from decord import VideoReader, gpu, cpu
import yaml



MAX_LENGTH = 8192
BATCH_SIZE = 4
OUTPUT_DIR = "./outputs" # path where to save the checkpoints
MODEL_ID = "your_path/LLaVA-NeXT-Video-7B-hf"
REPO_ID = "AoTD_train" # Change to your hf-hub repo
DATA_PATH = "datasets/train_datasets.yaml"


def get_video(video_path):
    vr = VideoReader(uri=video_path, ctx=cpu(0), num_threads=1) # you need to install from source to use gpu ctx
    vlen = len(vr)
    indices = np.linspace(0, vlen, 32, endpoint=False).astype(int)
    frames = vr.get_batch(indices).asnumpy()
    return frames


class LlavaNextVideoDataCollatorWithPadding:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        batches = []
        for feature in features:
            conversation = feature['conversation']
            conversation = self.processor.apply_chat_template(conversation, add_generation_prompt=False)

            video = get_video(feature['video_path'])
            # import ipdb; ipdb.set_trace()
            batch = self.processor(
                text=conversation,
                videos=video,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            )

            batch['input_ids'][0][-1] = self.processor.tokenizer.eos_token_id
            batches.append(batch)
        padded_inputs = self.processor.tokenizer.pad(
            {   
                "input_ids": [batch['input_ids'][0] for batch in batches],
                "attention_mask": [batch['attention_mask'][0] for batch in batches],
            },
            padding=True,
            return_tensors="pt",
        )
        labels = padded_inputs["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        padded_inputs["labels"] = labels
        padded_inputs["pixel_values_videos"] = torch.cat([batch['pixel_values_videos'] for batch in batches], dim=0)

        return padded_inputs

def build_dataset(data_path):
    dataset_combined = []

    with open(data_path, "r") as file:
        yaml_data = yaml.safe_load(file)
        datasets = yaml_data.get("datasets")

    for dataset in datasets:
        json_path = dataset.get("json_path")
        dataset = load_dataset("json", data_files=json_path)
        dataset_combined.append(dataset['train'])
    
    dataset_processed = concatenate_datasets(dataset_combined)
    dataset_processed = dataset_processed.shuffle(seed=42)

    train_dataset = dataset_processed.with_format("torch")

    return train_dataset


def parse_args():
    args = TrainingArguments(

        # args related to training
        output_dir = OUTPUT_DIR,
        # dataloader_drop_last=True,
        do_eval=False,
        evaluation_strategy="no",
        eval_strategy = 'steps',
        eval_steps=20,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size = BATCH_SIZE,
        gradient_accumulation_steps = 1,
        learning_rate = 4e-05,
        num_train_epochs=1, # adjust this depending on your dataset size
        lr_scheduler_type = 'cosine',
        warmup_ratio = 0.03,
        remove_unused_columns=False,

        # args related to eval/save
        logging_steps = 1,
        save_strategy = 'epoch',
        # save_steps=1000,
        save_total_limit = 1,
        fp16 = False, # we have the model train and eval with fp16 precision
        bf16=True,
        tf32=True,
        fp16_full_eval = True,
        optim = 'adamw_bnb_8bit', # adam in lower-bits to save memory, consider changing to 'adamw_torch' if model is not converging
        report_to = "none", # install wand to use this
        hub_model_id = REPO_ID,
        push_to_hub = False, # wel'll push the model to hub after each epoch
        label_names=["labels"],
        dataloader_num_workers=2, # let's get more workers since iterating on video datasets might be slower in general
        gradient_checkpointing=True,
        neftune_noise_alpha=0.1
    )

    return args





def main():
    args = parse_args()
    processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=False)
    processor.tokenizer.padding_side = "right"

    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False
    for k, v in model.named_parameters():
        if "vision_tower" in k or 'image_newline' in k:
            v.requires_grad = False

    train_dataset = build_dataset(DATA_PATH)


    trainer = Trainer(
        model = model,
        tokenizer = processor,
        data_collator = LlavaNextVideoDataCollatorWithPadding(processor=processor),
        train_dataset = train_dataset,
        args=args,
    )

    trainer.train()


if __name__ == "__main__":

    main()


