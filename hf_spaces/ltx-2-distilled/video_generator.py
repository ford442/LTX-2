import sys
from pathlib import Path
import random
import torch
import numpy as np
from typing import Optional, List
from huggingface_hub import hf_hub_download
from gradio_client import Client, handle_file
from ltx_pipelines.distilled import DistilledPipeline
from ltx_core.tiling import TilingConfig
from ltx_core.loader.primitives import LoraPathStrengthAndSDOps
from ltx_core.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP
from ltx_pipelines.constants import (
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    DEFAULT_LORA_STRENGTH,
)
from moviepy.editor import VideoFileClip, concatenate_videoclips
import tempfile
import os

# Add packages to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / "packages" / "ltx-pipelines" / "src"))
sys.path.insert(0, str(current_dir / "packages" / "ltx-core" / "src"))

MAX_SEED = np.iinfo(np.int32).max

# HuggingFace Hub defaults
DEFAULT_REPO_ID = "Lightricks/LTX-2"
DEFAULT_CHECKPOINT_FILENAME = "ltx-2-19b-dev-fp8.safetensors"
DEFAULT_DISTILLED_LORA_FILENAME = "ltx-2-19b-distilled-lora-384.safetensors"
DEFAULT_SPATIAL_UPSAMPLER_FILENAME = "ltx-2-spatial-upscaler-x2-1.0.safetensors"

# Text encoder space URL
TEXT_ENCODER_SPACE = "linoyts/gemma-text-encoder"

def get_hub_or_local_checkpoint(repo_id: Optional[str] = None, filename: Optional[str] = None):
    """Download from HuggingFace Hub or use local checkpoint."""
    if repo_id is None and filename is None:
        raise ValueError("Please supply at least one of `repo_id` or `filename`")

    if repo_id is not None:
        if filename is None:
            raise ValueError("If repo_id is specified, filename must also be specified.")
        print(f"Downloading {filename} from {repo_id}...")
        ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)
        print(f"Downloaded to {ckpt_path}")
    else:
        ckpt_path = filename

    return ckpt_path

# Initialize pipeline at startup
print("=" * 80)
print("Loading LTX-2 Distilled pipeline...")
print("=" * 80)

checkpoint_path = get_hub_or_local_checkpoint(DEFAULT_REPO_ID, DEFAULT_CHECKPOINT_FILENAME)
distilled_lora_path = get_hub_or_local_checkpoint(DEFAULT_REPO_ID, DEFAULT_DISTILLED_LORA_FILENAME)
spatial_upsampler_path = get_hub_or_local_checkpoint(DEFAULT_REPO_ID, DEFAULT_SPATIAL_UPSAMPLER_FILENAME)

print(f"Initializing pipeline with:")
print(f"  checkpoint_path={checkpoint_path}")
print(f"  distilled_lora_path={distilled_lora_path}")
print(f"  spatial_upsampler_path={spatial_upsampler_path}")
print(f"  text_encoder_space={TEXT_ENCODER_SPACE}")

# Load distilled LoRA as a regular LoRA
loras = [
    LoraPathStrengthAndSDOps(
        path=distilled_lora_path,
        strength=DEFAULT_LORA_STRENGTH,
        sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
    )
]

# Initialize pipeline WITHOUT text encoder (gemma_root=None)
# Text encoding will be done by external space
pipeline = DistilledPipeline(
    checkpoint_path=checkpoint_path,
    spatial_upsampler_path=spatial_upsampler_path,
    gemma_root=None,  # No text encoder in this space
    loras=loras,
    fp8transformer=True,
    local_files_only=False,
)

# Initialize text encoder client
print(f"Connecting to text encoder space: {TEXT_ENCODER_SPACE}")
try:
    text_encoder_client = Client(TEXT_ENCODER_SPACE)
    print("✓ Text encoder client connected!")
except Exception as e:
    print(f"⚠ Warning: Could not connect to text encoder space: {e}")
    text_encoder_client = None

print("=" * 80)
print("Pipeline fully loaded and ready!")
print("=" * 80)

def get_embeddings(prompt: str, enhance_prompt: bool, input_image: Optional[str], seed: int):
    """Get embeddings from the external text encoder space."""
    print(f"Encoding prompt: {prompt}")
    
    if text_encoder_client is None:
        raise RuntimeError(
            f"Text encoder client not connected. Please ensure the text encoder space "
            f"({TEXT_ENCODER_SPACE}) is running and accessible."
        )
    
    try:
        # Prepare image for upload if it exists
        image_input = None
        if input_image is not None:
            image_input = handle_file(str(input_image))
        
        result = text_encoder_client.predict(
            prompt=prompt,
            enhance_prompt=enhance_prompt,
            input_image=image_input,
            seed=seed,
            negative_prompt="",
            api_name="/encode_prompt"
        )
        embedding_path = result[0]  # Path to .pt file
        print(f"Embeddings received from: {embedding_path}")

        # Load embeddings
        embeddings = torch.load(embedding_path)
        video_context = embeddings['video_context']
        audio_context = embeddings['audio_context']
        print("✓ Embeddings loaded successfully")
        return video_context, audio_context
    except Exception as e:
        raise RuntimeError(
            f"Failed to get embeddings from text encoder space: {e}\n"
            f"Please ensure {TEXT_ENCODER_SPACE} is running properly."
        )

def generate_single_clip(
    input_image,
    prompt: str,
    duration: float,
    enhance_prompt: bool = True,
    seed: int = 42,
    randomize_seed: bool = True,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
):
    """Generate a single video clip."""
    try:
        # Randomize seed if checkbox is enabled
        current_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)

        # Calculate num_frames from duration (using fixed 24 fps)
        frame_rate = 24.0
        num_frames = int(duration * frame_rate) + 1  # +1 to ensure we meet the duration

        # Create output directory if it doesn't exist
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"video_{current_seed}.mp4"

        # Handle image input
        images = []
        temp_image_path = None  # Initialize to None
        
        if input_image is not None:
            # Save uploaded image temporarily
            temp_image_path = output_dir / f"temp_input_{current_seed}.jpg"
            if hasattr(input_image, 'save'):
                input_image.save(temp_image_path)
            else:
                # If it's a file path already
                temp_image_path = Path(input_image)
            # Format: (image_path, frame_idx, strength)
            images = [(str(temp_image_path), 0, 1.0)]
        
        video_context, audio_context = get_embeddings(prompt, enhance_prompt, temp_image_path, current_seed)

        # Run inference
        pipeline(
            prompt=prompt,
            output_path=str(output_path),
            seed=current_seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=images,
            tiling_config=TilingConfig.default(),
            video_context=video_context,
            audio_context=audio_context,
        )

        return str(output_path), current_seed

    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return None, None

def stitch_videos(clips_list: List[str]) -> str:
    """Stitch a list of video clips together."""
    if not clips_list or len(clips_list) < 2:
        raise ValueError("You need at least two clips to stitch them together!")
    
    print(f"Stitching {len(clips_list)} clips...")
    try:
        video_clips = [VideoFileClip(clip_path) for clip_path in clips_list]
        final_clip = concatenate_videoclips(video_clips, method="compose")
        
        final_output_path = os.path.join(tempfile.mkdtemp(), f"stitched_video_{random.randint(10000,99999)}.mp4")
        
        high_quality_params = ['-crf', '0', '-preset', 'veryslow']
        
        final_clip.write_videofile(
            final_output_path, 
            codec="libx264", 
            audio=False, 
            threads=4, 
            ffmpeg_params=high_quality_params
        )
        
        for clip in video_clips:
            clip.close()
            
        return final_output_path
    except Exception as e:
        raise RuntimeError(f"Failed to stitch videos: {e}")
