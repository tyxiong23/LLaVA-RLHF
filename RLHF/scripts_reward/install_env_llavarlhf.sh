pip install --upgrade pip

cd /mnt/bn/vl-research/workspace/txiong23/projects/ai_feedback/repos/LLaVA-RLHF/LLaVA

# git reset --hard 6cea223
# git apply < /mnt/bn/vl-research/workspace/txiong23/projects/ai_feedback/repos/LLaVA-RLHF/llava_setup/fix_llava_padding.patch

pip install -e .
pip install -e ".[train]"

# pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
# pip install deepspeed==0.9.3
# pip install peft==0.4.0
# pip install transformers==4.31.0
# pip install bitsandbytes==0.41.0
pip install datasets

# cd /mnt/bn/vl-research/workspace/txiong23/projects/lmms-eval/
# pip install -e .

pip install flash-attn --no-build-isolation