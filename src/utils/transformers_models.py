from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModel, AutoConfig
import torch
import gc


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
        
        if model_type:
            supported_models = get_supported_vision_models()
            
            # Debug: afficher le model_type détecté
            print(f"Detected model_type: '{model_type}'")
            
            # Vérification directe
            if model_type in supported_models:
                return True
            
            # Vérification avec normalisation (remplacer _ par - et vice versa)
            model_type_normalized = model_type.replace('_', '-')
            if model_type_normalized in supported_models:
                print(f"Matched with normalized name: '{model_type_normalized}'")
                return True
            
            model_type_with_underscore = model_type.replace('-', '_')
            if model_type_with_underscore in supported_models:
                print(f"Matched with underscore name: '{model_type_with_underscore}'")
                return True
        
        return False
        
    except Exception as e:
        print(f"Could not determine model type: {e}")
        import traceback
        traceback.print_exc()
        return False