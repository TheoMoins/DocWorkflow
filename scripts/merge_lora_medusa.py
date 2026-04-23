"""
Merge LoRA weights from outputs/T1-2_9B/Qwen3.5-9B-line-finetuned-line-finetuned
on top of outputs/T1-2_9B/merged_base into a full 16-bit model at outputs/Medusa0.1Line-9B.

Reproduces the merging logic from VLMLineHTRTask.train().
"""
import os
from pathlib import Path

BASE_PATH = "outputs/T1-2_9B/merged_base"
LORA_PATH = "outputs/T1-2_9B/Qwen3.5-9B-line-finetuned-line-finetuned"
OUTPUT_PATH = "outputs/Medusa0.1Line-9B"

def main():
    from unsloth import FastVisionModel
    from peft import PeftModel

    base_path = Path(BASE_PATH)
    lora_path = Path(LORA_PATH)
    output_path = Path(OUTPUT_PATH)

    if not base_path.exists():
        raise FileNotFoundError(f"Base model not found: {base_path}")
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA adapter not found: {lora_path}")

    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model from {base_path} ...")
    model, tokenizer = FastVisionModel.from_pretrained(
        str(base_path),
        load_in_4bit=False,
        load_in_8bit=False,
        use_gradient_checkpointing="unsloth",
    )

    print(f"Applying LoRA adapter from {lora_path} ...")
    model = PeftModel.from_pretrained(model, str(lora_path))

    print(f"Saving merged 16-bit model to {output_path} ...")
    model.save_pretrained_merged(str(output_path), tokenizer, save_method="merged_16bit")

    print(f"Done. Merged model saved to {output_path}")


if __name__ == "__main__":
    # Run from the project root:
    #   python scripts/merge_lora_medusa.py
    os.chdir(Path(__file__).resolve().parents[1])
    main()
