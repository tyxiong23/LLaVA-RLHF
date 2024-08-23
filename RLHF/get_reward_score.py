import argparse
from dataclasses import dataclass, field
import json
import os
import sys
from typing import Optional, List, Dict, Sequence
import logging
from argparse import Namespace
from os.path import join, exists
from llava import conversation as conversation_lib
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token

import copy

import torch
import transformers

from finetune_lora_ppo import DataArguments, ModelArguments, TrainingArguments
from models.reward_model import RewardConfig, RewardModel

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, PeftModel, PeftModelForCausalLM, prepare_model_for_kbit_training
from peft.tuners.lora import LoraLayer
from transformers import BitsAndBytesConfig
from tqdm import tqdm
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from PIL import Image
import random

from llava.model import LlavaLlamaForCausalLM
import math


FACTUAL_PROMPT = "Specifically, the AI's response should be fully supported by the combination of the following captions:\n"

@dataclass
class ExtraArguments:
    num_chunks: int = field(
        default=1,
    )
    chunk_idx: int = field(
        default=0
    )
    answer_file: str = field(
        default=None
    )

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def load_jsonl(save_path):
    with open(save_path, "r") as f:
        data = [json.loads(line) for line in f.readlines()]
    return data


def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def load_data(data_path: str):
    if data_path.endswith("jsonl"):
        data_list = load_jsonl(data_path)
    else:
        data_list = load_json(data_path)
    return data_list


class DisableLogger:
    def __enter__(self):
        logging.disable(logging.CRITICAL)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        logging.disable(logging.NOTSET)


def load_reward_model_for_inference(
    checkpoint_dir: str,
    vision_tower: str = None,
    lora_modules: list = None,
    image_aspect_ratio: str = "square",
    image_grid_pinpoints: int = None,
    bits: int = 4,
    fp16: bool = False,
    bf16: bool = False,
    double_quant: bool = True,
    quant_type: str = "nf4",
    gradient_checkpointing: bool = False,
    adapter_name="lora_default",
    is_trainable=True,
    reuse_base_model=False,
    trust_remote_code=False,
):
    # Load the model.
    lora_checkpoint_dir = checkpoint_dir
    if os.path.exists(os.path.join(lora_checkpoint_dir, "adapter_model")):
        lora_checkpoint_dir = os.path.join(lora_checkpoint_dir, "adapter_model")
    if os.path.exists(os.path.join(lora_checkpoint_dir, "lora_default")):
        lora_checkpoint_dir = os.path.join(lora_checkpoint_dir, "lora_default")

    lora_config = LoraConfig.from_pretrained(lora_checkpoint_dir)
    config = RewardConfig(
        backbone_model_name_or_path=lora_config.base_model_name_or_path
    )

    args = Namespace(
        model_name_or_path=config.backbone_model_name_or_path,
        vision_tower=vision_tower,
        lora_modules=lora_modules,
        image_aspect_ratio=image_aspect_ratio,
        image_grid_pinpoints=image_grid_pinpoints,
        bits=bits,
        fp16=fp16,
        bf16=bf16,
        double_quant=double_quant,
        quant_type=quant_type,
        trust_remote_code=trust_remote_code,
        full_finetune=False,
        gradient_checkpointing=gradient_checkpointing,
    )

    print("bf16", args.bf16, "fp16", args.fp16)

    model = RewardModel(
        args,
        config,
        checkpoint_dir=checkpoint_dir,
        qlora=True,
        adapter_name=adapter_name,
        is_trainable=is_trainable,
        reuse_base_model=reuse_base_model,
    )
    return model


def make_reward_model(
        adapter_name, is_trainable, args, reuse_base_model=False, num_new_tokens: int = 0
    ):
        model = load_reward_model_for_inference(
            checkpoint_dir=args.reward_model_name_or_path,
            vision_tower=args.vision_tower,
            image_aspect_ratio=args.image_aspect_ratio,
            image_grid_pinpoints=args.image_grid_pinpoints,
            # bits=16,
            bits=16,
            fp16=args.fp16,
            bf16=args.bf16,
            gradient_checkpointing=args.gradient_checkpointing,
            adapter_name=adapter_name,
            is_trainable=is_trainable,
            reuse_base_model=reuse_base_model,
            trust_remote_code=args.trust_remote_code,
        )
        # smart_tokenizer_and_embedding_resize(
        #     num_new_tokens, tokenizer, model.backbone_model
        # )
        
        vision_tower = model.backbone_model.get_vision_tower()
        print("vision_tower loaded", vision_tower.is_loaded)
        vision_tower.to(device="cuda", dtype=torch.bfloat16)
        mm_projector = model.backbone_model.get_model().mm_projector
        mm_projector.to(device="cuda", dtype=torch.bfloat16)
        for name, module in model.named_modules():
            if hasattr(module, "weight"):
                if args.bf16 and module.weight.dtype == torch.float:
                    print('not bf16', name)
                    module.to(dtype=torch.bfloat16)
        # model.backbone_model.merge_and_unload()
        # model.backbone_model.to(dtype=torch.bfloat16)
        
        
        return model


def prepare_llava_image(image_path, image_processor, image_aspect_ratio) -> torch.Tensor:
    def expand2square(pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(
                pil_img.mode, (width, width), background_color
            )
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(
                pil_img.mode, (height, height), background_color
            )
            result.paste(pil_img, ((height - width) // 2, 0))
            return result
    image = Image.open(image_path).convert("RGB")

    if image_aspect_ratio == "pad":   
        # print("pad", image.size)
        image = expand2square(
            image, tuple(int(x * 255) for x in image_processor.image_mean)
        )
        # print("pad_new", image.size, image_processor.preprocess(image, return_tensors="pt"))
        
    image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
    return image_tensor



def single_reward(model: RewardModel, question: str, response: str, reward_model_prompt: str, image_tensor: torch.Tensor, version: str, tokenizer, captions=None):
    conv = copy.deepcopy(conv_templates[version])
    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], response)

    reward_model_prompt_per_example = reward_model_prompt

    if (
            captions is not None
            and r"{factual_prompt}" in reward_model_prompt_per_example
    ):
        factual_prompt = FACTUAL_PROMPT
        for caption in captions:
            factual_prompt = factual_prompt + f"  - {caption}\n"
        reward_model_prompt_per_example = reward_model_prompt_per_example.format(
            factual_prompt=factual_prompt
        )

    if reward_model_prompt_per_example is not None:
        prompt_question = conv.get_prompt() + reward_model_prompt_per_example + "</s>"
    else:
        prompt_question = conv.get_prompt()

    # print(f'[{prompt_question}]')
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
    input_ids = input_ids[:, :-1]
    # print("input_ids", input_ids.shape)
    with torch.inference_mode():
        reward_score = model(
            input_ids=input_ids,
            images = image_tensor.to(torch.bfloat16).cuda(),
            return_dict=True
        ).rewards
        # print(reward_score.shape)
    return reward_score





def main():
    # initialize parameters
    hfparser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, ExtraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        extra_args,
        extra_args1
    ) = hfparser.parse_args_into_dataclasses(return_remaining_strings=True)

    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args), **vars(extra_args)
    )
    training_args.data_config = data_args
    tokenizer_model_name = args.base_model_name
    TokenizerClass = AutoTokenizer

    # prepare input and output
    absolute_data_path = os.path.expanduser(args.dataset_path)
    dataset_all = load_data(absolute_data_path)
    dataset = get_chunk(dataset_all, args.num_chunks, args.chunk_idx)
    print(f"load {len(dataset)} of {len(dataset_all)} for chunk {args.chunk_idx}")

    caption_dict = None
    if args.image_to_caption_file:
        caption_dict = load_data(args.image_to_caption_file)
        print(f"load {len(caption_dict)} caption dicts")
    
    answer_file = os.path.expanduser(args.answer_file)
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    ans_file = open(answer_file, "w")


    # Tokenizer
    tokenizer = TokenizerClass.from_pretrained(
        tokenizer_model_name,
        cache_dir=args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        truncation_side="right",
        use_fast=False,
    )

    tokenizer.pad_token = tokenizer.unk_token


    with DisableLogger():
        base_model = LlavaLlamaForCausalLM.from_pretrained(
            model_args.base_model_name,
            cache_dir=training_args.cache_dir,
        )

        vision_tower = base_model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()

        image_processor = vision_tower.image_processor
        del base_model

    # load reward model
    print("loading reward model.....")
    reward_model = make_reward_model(
        adapter_name="lora_default",
        is_trainable=False,
        args=args,
        reuse_base_model=False
    )
    print("Reward Model loaded!", reward_model.device)
    reward_model.backbone_model.config.use_cache = False
    print("Reward head", reward_model.reward_head.weight.device)

    reward_model_prompt = None
    if args.reward_prompt_file is not None:
        print("read reward model prompt from", args.reward_prompt_file)
        with open(args.reward_prompt_file, "r") as f:
            reward_model_prompt = " " + f.read().strip()
        print(reward_model_prompt)

    for idx, line in enumerate(tqdm(dataset)):
        prompt = line['prompt']
        prompt = prompt.replace("<image>", "").strip()
        question = DEFAULT_IMAGE_TOKEN + '\n' + prompt

        # prepare image
        image_file = line['image']
        image_path = os.path.join(args.image_folder, image_file)
        image_tensor = prepare_llava_image(image_path, image_processor=image_processor, image_aspect_ratio=args.image_aspect_ratio)
        # print(image_tensor.shape)

        captions = None
        if caption_dict:
            captions = caption_dict[image_file.split('/')[-1]]
            random.shuffle(captions)

        
        for k in ['chosen', 'rejected']:
            response = line[k]
            reward_score = single_reward(
                model=reward_model, question=question,
                response=response, reward_model_prompt=reward_model_prompt,
                image_tensor=image_tensor, version=args.version,
                tokenizer=tokenizer, captions=captions
            )
            k_score = f'{k}_score'
            line[k_score] = reward_score.item()
        

        line['pred_rewards'] = []
        for pred_idx, pred in enumerate(line['pred']):
            response = pred
            reward_score = single_reward(
                model=reward_model, question=question,
                response=response, reward_model_prompt=reward_model_prompt,
                image_tensor=image_tensor, version=args.version,
                tokenizer=tokenizer, captions=captions
            )
            line['pred_rewards'].append(reward_score.item())

        print('chosen-rejected', line['chosen_score'], line['rejected_score'], line['pred_rewards'])

        ans_file.write(json.dumps(line) + "\n")
        ans_file.flush()

        # if idx > 10:
        #     break


if __name__ == "__main__":
    main()
    