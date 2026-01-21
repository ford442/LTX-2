import sys
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

# Add packages to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / "packages" / "ltx-pipelines" / "src"))
sys.path.insert(0, str(current_dir / "packages" / "ltx-core" / "src"))

from video_generator import generate_single_clip, stitch_videos
from ltx_pipelines.constants import (
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
)

app = FastAPI()

class GenerationRequest(BaseModel):
    prompt: str
    input_image: Optional[str] = None
    duration: float = 3.0
    enhance_prompt: bool = True
    seed: int = 42
    randomize_seed: bool = True
    height: int = DEFAULT_HEIGHT
    width: int = DEFAULT_WIDTH

class Clip(BaseModel):
    path: str
    # Gemini-comment: This model can be extended with more custom variables per clip,
    # for example:
    # duration: Optional[float] = None
    # fade_in: bool = False
    # fade_out: bool = False

class StitchRequest(BaseModel):
    clips: List[Clip]

@app.post("/generate/")
async def generate(request: GenerationRequest):
    """Generate a single video clip."""
    output_path, seed = generate_single_clip(
        input_image=request.input_image,
        prompt=request.prompt,
        duration=request.duration,
        enhance_prompt=request.enhance_prompt,
        seed=request.seed,
        randomize_seed=request.randomize_seed,
        height=request.height,
        width=request.width,
    )
    if output_path:
        return {"output_path": output_path, "seed": seed}
    else:
        return {"error": "Failed to generate video"}

@app.post("/stitch/")
async def stitch(request: StitchRequest):
    """Stitch a list of video clips together."""
    try:
        clip_paths = [clip.path for clip in request.clips]
        final_video_path = stitch_videos(clip_paths)
        return {"stitched_video_path": final_video_path}
    except (ValueError, RuntimeError) as e:
        return {"error": str(e)}

# Gemini-comment: New Features Integration
# The following endpoints are placeholders for integrating other Hugging Face Spaces and services.
# These integrations would allow "Jules, private web app" to orchestrate a full creative pipeline.

# @app.post("/generate_image/")
# async def generate_image_proxy(prompt: str):
#     """
#     Call an external Image Generation Space (e.g., FLUX, SDXL).
#     This allows creating the initial frame for video generation from text.
#     """
#     # Implementation: Use `gradio_client` to call the external space.
#     pass

# @app.post("/interpolate/")
# async def interpolate_video(video_path: str, interpolation_factor: int = 2):
#     """
#     Call an external RIFE Interpolation Space.
#     This smoothens the video by generating intermediate frames.
#     """
#     # Implementation: Call RIFE space API.
#     pass

# @app.post("/upload_ftp/")
# async def upload_to_ftp(file_path: str):
#     """
#     Upload a generated video to an FTP server.
#     Useful for archiving or transferring to other systems.
#     """
#     # Implementation: Use `paramiko` (as seen in `ltx-video-distilled-tester-dev/app.py`).
#     pass

# @app.post("/store_notes/")
# async def store_notes(note_content: str, related_media_path: Optional[str] = None):
#     """
#     Store notes or metadata about a generation in a Notes Space or database.
#     This could include prompts, seeds, and user feedback.
#     """
#     # Implementation: Call Notes space API or database.
#     pass

# Gemini-comment: Storage API & Tensor Persistence
# Question for the user: "Since we need to store 'last frame data' kind of tensors for video continuation,
# should we use a specific Storage API (like an S3 bucket, a dedicated HF Dataset, or a custom database)
# to keep these heavy tensor files? Or should we rely on the FTP solution mentioned above?"
#
# Keeping tensor data allows for perfect "lossless" continuation, whereas using the video frame
# introduces re-encoding artifacts.

@app.get("/")
async def root():
    return {"message": "LTX-2 Distilled API is running."}
