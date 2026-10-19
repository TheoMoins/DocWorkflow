import os
import sys
from pathlib import Path

BASE_PATH = "outputs/orig_outputs_T1-2/merged_base"
LORA_PATH = "outputs/orig_outputs_T1-2/Qwen3.5-4B-line-finetuned-line-finetuned"
OUTPUT_PATH = "outputs/Medusa0.1Line-4B"

def main(base=BASE_PATH, lora=LORA_PATH, output=OUTPUT_PATH):
    from unsloth import FastVisionModel
    from peft import PeftModel

    base_path = Path(base)
    lora_path = Path(lora)
    output_path = Path(output)

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

    print("Merging weights ...")
    model = model.merge_and_unload()

    print(f"Saving merged 16-bit model to {output_path} ...")
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    print(f"Done. Merged model saved to {output_path}")


if __name__ == "__main__":
    # Run from the project root:
    #   python scripts/merge_lora_medusa.py [BASE_PATH LORA_PATH OUTPUT_PATH]
    os.chdir(Path(__file__).resolve().parents[1])
    if len(sys.argv) not in (1, 4):
        sys.exit("usage: python scripts/merge_lora_medusa.py [BASE_PATH LORA_PATH OUTPUT_PATH]")
    main(*sys.argv[1:])
