#!/bin/bash

cd /mnt/bn/vl-research/workspace/txiong23/projects/ai_feedback/repos/LLaVA-RLHF/RLHF

IMAGE_DIR="/mnt/bn/vl-research/workspace/txiong23/datasets"
MODEL_DIR="/mnt/bn/vl-research/workspace/txiong23/projects/ai_feedback/repos/LLaVA-RLHF/checkpoints/LLaVA-RLHF-13b-v1.5-336"
RM_LORA="rm_lora_adapter_model"

for BASE_DIR in "/mnt/bn/vl-research/workspace/txiong23/projects/ai_feedback/llava_next/project_checkpoints/one_vision/itersplit3/dpo/llava-onevision_Qwen2-7b-ov_dpo_llava-rlhf-data-generate-fact_itersplit3-1_beta0.1_epoch1" "/mnt/bn/vl-research/workspace/txiong23/projects/ai_feedback/llava_next/project_checkpoints/one_vision/itersplit3/sppo/llava-onevision_Qwen2-7b-ov_sppo_llava-rlhf-data-generate-fact_itersplit3-1_beta0.001_epoch1" "/mnt/bn/vl-research/workspace/txiong23/projects/ai_feedback/llava_next/project_checkpoints/one_vision/itersplit3/sppo-hard/llava-onevision_Qwen2-7b-ov_sppo-hard_llava-rlhf-data-generate-fact_itersplit3-1_beta0.001_epoch1"
do

JSON_PATH="${BASE_DIR}/generated_responses/all_merge.jsonl"
OUTPUT_DIR="${BASE_DIR}/get_rewards"

NUM_CHUNKS=8

output_file_merge="${OUTPUT_DIR}/all_merge_rewards.jsonl"
echo $output_file_merge

for chunk in $(seq 0 $((NUM_CHUNKS-1))); do
    echo $chunk
    CUDA_VISIBLE_DEVICES=${chunk} python3 get_reward_score.py \
        --vision_tower openai/clip-vit-large-patch14-336 \
        --num_chunks $NUM_CHUNKS \
        --chunk_idx $chunk \
        --bf16 True \
        --answer_file "${OUTPUT_DIR}/split_rewards/${NUM_CHUNKS}_${chunk}.jsonl" \
        --dataset_path $JSON_PATH \
        --image_folder $IMAGE_DIR \
        --image_aspect_ratio 'pad' \
        --reward_model_name_or_path "${MODEL_DIR}/${RM_LORA}" \
        --base_model_name "${MODEL_DIR}/sft_model" \
        --model_max_length 2048 \
        --query_len 1280 \
        --response_len 768 \
        --reward_prompt_file "/mnt/bn/vl-research/workspace/txiong23/projects/ai_feedback/repos/LLaVA-RLHF/RLHF/prompts/fact_rlhf_reward_prompt.txt" \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_to_caption_file "/mnt/bn/vl-research/workspace/txiong23/datasets/dpo_data/llava-rlhf/image_to_caption.json" \
        --version "v1" &
done

wait

# Clear out the output file if it exists.
> "$output_file_merge"

# Loop through the indices and concatenate each file.
for chunk in $(seq 0 $((NUM_CHUNKS-1))); do
    cat "${OUTPUT_DIR}/split_rewards/${NUM_CHUNKS}_${chunk}.jsonl" >> "$output_file_merge"
done

pip install matplotlib

python3 compute_sppo_probs.py \
    --in_file $output_file_merge

done