from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModel, AutoConfig
import re


def get_supported_vision_models():
    """
    Get list of model types supported by AutoModelForImageTextToText.
    
    Returns:
        Set of supported model type strings
    """
    try:
        if hasattr(AutoModelForImageTextToText, '_model_mapping'):
            mapping = AutoModelForImageTextToText._model_mapping
            if hasattr(mapping, '_model_mapping'):
                return set(mapping._model_mapping.keys())
        
        return {
            'aria', 'aya_vision', 'blip', 'blip-2', 'chameleon', 'cohere2vision',
            'deepseek-vl', 'deepseek-vl-hybrid', 'emu3', 'evolla', 'florence2',
            'fuyu', 'gemma3', 'gemma3n', 'git', 'glm4v', 'glm4v-moe', 'gotocr2',
            'idefics', 'idefics2', 'idefics3', 'instructblip', 'internvl',
            'janus', 'kosmos-2', 'kosmos-2.5', 'lfm2vl', 'llama4', 'llava',
            'llava-next', 'llava-next-video', 'llava-onevision', 'mistral3',
            'mllama', 'ovis2', 'paligemma', 'perception-lm', 'pix2struct',
            'pixtral', 'qwen2.5-vl', 'qwen2-vl', 'qwen3-vl', 'qwen3-vl-moe',
            'shieldgemma2', 'smolvlm', 'udop', 'vipllava', 'vision-encoder-decoder'
        }
    except Exception:
        return {
            'aria', 'aya_vision', 'blip', 'blip-2', 'chameleon', 'cohere2vision',
            'deepseek-vl', 'deepseek-vl-hybrid', 'emu3', 'evolla', 'florence2',
            'fuyu', 'gemma3', 'gemma3n', 'git', 'glm4v', 'glm4v-moe', 'gotocr2',
            'idefics', 'idefics2', 'idefics3', 'instructblip', 'internvl',
            'janus', 'kosmos-2', 'kosmos-2.5', 'lfm2vl', 'llama4', 'llava',
            'llava-next', 'llava-next-video', 'llava-onevision', 'mistral3',
            'mllama', 'ovis2', 'paligemma', 'perception-lm', 'pix2struct',
            'pixtral', 'qwen2.5-vl', 'qwen2-vl', 'qwen3-vl', 'qwen3-vl-moe',
            'shieldgemma2', 'smolvlm', 'udop', 'vipllava', 'vision-encoder-decoder'
        }


def normalize_model_type(model_type):
    """
    Normalize model type string for comparison.
    Handles variations like qwen2_5_vl_text -> qwen2.5-vl
    
    Args:
        model_type: Raw model type string
        
    Returns:
        Normalized model type string
    """
    if not model_type:
        return None
    
    normalized = model_type
    normalized = re.sub(r'(\d)_(\d)', r'\1.\2', normalized)
    
    # Remplacer les _ restants par -
    normalized = normalized.replace('_', '-')
    
    # Enlever les suffixes connus
    for suffix in ['-text', '-vision', '-chat', '-instruct', '-model']:
        if normalized.endswith(suffix):
            normalized = normalized[:-len(suffix)]
            break
    
    return normalized


def is_supported_by_auto_image_text(model_name_or_path):
    """
    Check if a model is supported by AutoModelForImageTextToText.
    
    Args:
        model_name_or_path: Model name or path
        
    Returns:
        Boolean indicating if model is supported
    """
    try:
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
        
        model_type = getattr(config, 'model_type', None)
        
        if not model_type:
            return False
        
        print(f"Detected model_type: '{model_type}'")
        
        supported_models = get_supported_vision_models()
        
        if model_type in supported_models:
            print(f"  -> Direct match in supported models")
            return True
        
        normalized_type = normalize_model_type(model_type)
        print(f"  -> Normalized to: '{normalized_type}'")
        
        normalized_supported = {normalize_model_type(m) for m in supported_models}
        
        if normalized_type in normalized_supported:
            print(f"  -> Match after normalization!")
            return True
        
        print(f"  -> No match found")
        return False
        
    except Exception as e:
        print(f"Could not determine model type: {e}")
        return False