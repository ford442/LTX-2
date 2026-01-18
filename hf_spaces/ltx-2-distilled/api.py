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

@app.get("/")
async def root():
    return {"message": "LTX-2 Distilled API is running."}
