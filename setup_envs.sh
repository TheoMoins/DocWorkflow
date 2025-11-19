#!/bin/bash

echo "Setting up DocWorkflow environments..."

# Create main environment
echo "Creating main environment..."
virtualenv -p python3.10 envs/main
source envs/main/bin/activate
pip install -e ".[dev]"
deactivate

# Create VLM training environment
echo "Creating VLM training environment..."
virtualenv -p python3.10 envs/vlm-training
source envs/vlm-training/bin/activate

# Install unsloth and dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install unsloth qwen-vl-utils
pip install --no-deps trl peft accelerate bitsandbytes
pip install transformers datasets pandas pyyaml lxml pillow 
pip install matplotlib click tabulate jiwer

pip install -e . --no-deps 

deactivate

echo "Setup complete!"