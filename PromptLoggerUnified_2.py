import os
import json
import hashlib
from datetime import datetime
import pytz  # Ensure pytz is installed in your environment
import torch
import comfy.samplers

class PromptLoggerUnified:
    def extract_model_metadata(self, model):
        """Extract metadata from a ComfyUI MODEL object"""
        if model is None:
            return None

        metadata = {}

        try:
            # Try to get model type from the model configuration
            if hasattr(model, 'model') and hasattr(model.model, 'config'):
                config = model.model.config
                if 'model_type' in config:
                    metadata['model_type'] = config['model_type']
                elif 'in_channels' in config:
                    # Infer model type from config
                    if config.get('in_channels') == 9:
                        metadata['model_type'] = 'SD_Inpainting'
                    elif config.get('model_channels') == 320 and config.get('in_channels') == 4:
                        metadata['model_type'] = 'SDXL'
                    else:
                        metadata['model_type'] = 'SD1.5'

            # Try to get model hash if available
            if hasattr(model, 'model') and hasattr(model.model, 'state_dict'):
                try:
                    # Get a sample of model weights for hashing
                    state_dict = model.model.state_dict()
                    if state_dict:
                        # Use first weight tensor for hash
                        first_key = next(iter(state_dict.keys()))
                        weights = state_dict[first_key]
                        if isinstance(weights, torch.Tensor):
                            # Convert to bytes and hash
                            weights_bytes = weights.detach().cpu().numpy().tobytes()
                            metadata['model_hash'] = hashlib.sha256(weights_bytes).hexdigest()[:16]
                except Exception as e:
                    print(f"[PromptLoggerUnified] Could not generate model hash: {e}")

        except Exception as e:
            print(f"[PromptLoggerUnified] Error extracting model metadata: {e}")

        return metadata if metadata else None

    def parse_lora_info(self, lora_info_str):
        """Parse LoRA information from string format: name:strength_model:strength_clip"""
        if not lora_info_str or not lora_info_str.strip():
            return []

        loras = []
        lines = [line.strip() for line in lora_info_str.split('\n') if line.strip()]

        for line in lines:
            if ':' in line:
                parts = line.split(':')
                if len(parts) >= 2:
                    name = parts[0].strip()
                    try:
                        strength_model = float(parts[1]) if len(parts) > 1 else 1.0
                        strength_clip = float(parts[2]) if len(parts) > 2 else strength_model

                        loras.append({
                            'name': name,
                            'strength_model': strength_model,
                            'strength_clip': strength_clip
                        })
                    except ValueError:
                        # If parsing fails, just store the name
                        loras.append({
                            'name': name,
                            'strength_model': 1.0,
                            'strength_clip': 1.0
                        })
            else:
                # Just a name, use default strengths
                loras.append({
                    'name': line,
                    'strength_model': 1.0,
                    'strength_clip': 1.0
                })

        return loras

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": ("STRING", {"default": "output/projectX"}),
                "base_name": ("STRING", {"default": "logger"}),

                "sampler": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "normal"}),

                "steps": ("INT", {"default": 20}),
                "cfg": ("FLOAT", {"default": 7.5}),
                "seed": ("INT", {"default": 2025}),
                "control_after_generate": ("BOOLEAN", {"default": False}),
                "denoise": ("FLOAT", {"default": 1.0}),
                "use_timestamp": ("BOOLEAN", {"default": True}),
                "timestamp_format": ("STRING", {"default": "%d%b%Y_%H%M"}),
                "prompt": ("STRING", {"multiline": True, "lines": 200})
            },
            "optional": {
                "model": ("MODEL", {"tooltip": "Diffusion model for metadata extraction"}),
                "checkpoint_name": ("STRING", {"default": "", "tooltip": "Name of the checkpoint model used"}),
                "lora_info": ("STRING", {"default": "", "multiline": True, "tooltip": "LoRA information (name:strength_model:strength_clip format)"}),
                "vae_name": ("STRING", {"default": "", "tooltip": "Name of the VAE model used"})
            }
        }

    RETURN_TYPES = (
        "STRING", "STRING", "STRING", "STRING", "INT", "FLOAT", "INT", "BOOLEAN", "FLOAT")
    RETURN_NAMES = ("prompt", "filename", "sampler_name", "scheduler", "steps", "cfg", "seed", "control_after_generate", "denoise")
    FUNCTION = "log_and_generate"
    CATEGORY = "logging"

    def log_and_generate(self, prompt, folder, base_name, sampler, scheduler, denoise, steps, cfg, seed, use_timestamp, timestamp_format, control_after_generate, model=None, checkpoint_name="", lora_info="", vae_name=""):
        os.makedirs(folder, exist_ok=True)

        pacific = pytz.timezone("US/Pacific")
        now = datetime.now(pacific)
        stamp = now.strftime(timestamp_format) if use_timestamp else ""
        filename_base = f"{base_name}_{stamp}" if stamp else base_name
        relative_folder = folder.replace("output/", "").lstrip("/")
        image_filename = os.path.join(relative_folder, f"{filename_base}.png")
        json_filename = os.path.join(folder, f"{filename_base}.json")

        # Extract model metadata
        model_metadata = self.extract_model_metadata(model)
        lora_list = self.parse_lora_info(lora_info)

        entry = {
            "filename": os.path.basename(image_filename),
            "timestamp": now.isoformat(),
            "prompt": prompt,
            "folder": folder,
            "base_name": base_name,
            "sampler": sampler,
            "scheduler": scheduler,
            "steps": steps,
            "cfg": cfg,
            "seed": seed,
            "control_after_generate": control_after_generate,
            "denoise": denoise,
            "use_timestamp": use_timestamp,
            "timestamp_format": timestamp_format,
            "ksampler": {
                "sampler": sampler,
                "scheduler": scheduler,
                "steps": steps,
                "cfg": cfg,
                "seed": seed
            },
            "model": model_metadata if model else None,
            "checkpoint_name": checkpoint_name if checkpoint_name else None,
            "lora_info": lora_list if lora_info else None,
            "vae_name": vae_name if vae_name else None
        }

        # Add model information if available
        models_info = {}
        if checkpoint_name:
            models_info["checkpoint"] = checkpoint_name
        if model_metadata:
            models_info.update(model_metadata)
        if lora_list:
            models_info["loras"] = lora_list
        if vae_name:
            models_info["vae"] = vae_name

        if models_info:
            entry["models"] = models_info

        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2, ensure_ascii=False)

        # Print model information summary
        if models_info:
            print(f"[PromptLoggerUnified] Model info logged: {len(models_info)} components")
            if checkpoint_name:
                print(f"[PromptLoggerUnified] Checkpoint: {checkpoint_name}")
            if lora_list:
                print(f"[PromptLoggerUnified] LoRAs: {len(lora_list)} applied")
            if vae_name:
                print(f"[PromptLoggerUnified] VAE: {vae_name}")

        print(f"[PromptLoggerUnified] Saved image path: {image_filename}")
        print(f"[PromptLoggerUnified] Logged metadata to: {json_filename}")
        return (prompt, image_filename, sampler, scheduler, steps, cfg, seed, denoise, control_after_generate)

NODE_CLASS_MAPPINGS = {"PromptLoggerUnified": PromptLoggerUnified}
NODE_DISPLAY_NAME_MAPPINGS = {"Prompt Logger Unified v2": "Prompt Logger Unified v2"}
