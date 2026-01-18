# LTX-2 Distilled with Stitching and API

This directory contains the code for the LTX-2 Distilled Hugging Face Space, with added functionality for video stitching and an API for programmatic access.

## New Structure

The codebase has been refactored for better modularity:

-   `app.py`: The main Gradio application for interactive use.
-   `video_generator.py`: Contains the core logic for generating and stitching video clips. All pipeline initialization and inference code resides here.
-   `api.py`: A FastAPI application that exposes the video generation and stitching functionality through a web API.
-   `requirements.txt`: The Python dependencies, updated to include `fastapi`, `uvicorn`, and `moviepy`.
-   `ltx2_two_stage.py`: An example script for using a two-stage pipeline (unchanged).

## Running the API

To run the API server, use the following command:

```bash
uvicorn hf_spaces.ltx-2-distilled.api:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

## API Endpoints

### `POST /generate/`

Generates a single video clip.

**Request Body:**

```json
{
  "prompt": "a cinematic shot of a cat wearing a cowboy hat",
  "input_image": null,
  "duration": 3.0,
  "enhance_prompt": true,
  "seed": 42,
  "randomize_seed": true,
  "height": 512,
  "width": 512
}
```

**Response:**

```json
{
  "output_path": "outputs/video_12345.mp4",
  "seed": 12345
}
```

### `POST /stitch/`

Stitches a list of video clips together.

**Request Body:**

The `clips` field takes a list of objects, where each object has a `path` to the video clip. This has been designed to be extensible with more custom variables per clip in the future.

```json
{
  "clips": [
    { "path": "outputs/video_12345.mp4" },
    { "path": "outputs/video_67890.mp4" }
  ]
}
```

**Response:**

```json
{
  "stitched_video_path": "/tmp/tmpXXXXX/stitched_video_54321.mp4"
}
```
