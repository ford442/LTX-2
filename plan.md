# Project Plan

This repository contains the source code for several Hugging Face Spaces related to video generation. The goal is to experiment with and improve upon the existing models and applications.

## Repository Structure

The repository is organized into several Hugging Face Spaces and a shared `packages` directory.

*   **`packages`**: This directory contains the core libraries (`ltx-core`, `ltx-pipelines`, `ltx-trainer`) that are shared across all the Hugging Face Spaces. Changes made in these packages will affect all the applications.

*   **Hugging Face Spaces (`hf_spaces`)**:
    *   **`ltx-2-distilled-original`**: The original, unmodified version of the LTX-2 distilled space.
    *   **`ltx-2-distilled`**: A refactored version of the LTX-2 distilled space, which now includes stitching capabilities and an API. This serves as an example of a "tester" application.
    *   **`ltx-2-distilled-tester-a` / `ltx-2-distilled-tester-b`**: Copies of the original application for implementing and testing new features.
    *   **`ltx-video-distilled-tester-original`**: The original, unmodified version of the LTX-Video distilled tester space.
    *   **`ltx-video-distilled-tester-dev`**: A development version of the LTX-Video distilled tester.
    *   **`ltx-video-distilled-tester`**: The space that was initially analyzed.

## Development Roadmap

### 1. Feature Porting: `ltx-video-distilled-tester` to `ltx-2-distilled`

The primary goal is to merge features from the `ltx-video-distilled-tester-dev` space into the `ltx-2-distilled` space. This includes, but is not limited to:

*   **Stitching and Video Continuation**: Implementing the advanced stitching and video continuation logic.
*   **Advanced Configuration**: Porting the highly tuned performance and quality settings.

### 2. Enhancements for `ltx-1-dev`

We will also focus on improving the `ltx-1-dev` model/space. The specific enhancements will be determined after an initial analysis of the existing codebase and its limitations.

### 3. General Improvements

*   **Code Quality:** Refactor and improve the code quality of the new tester applications.
*   **Documentation:** Update documentation to reflect the changes and new features.

### 4. Checkpoint and Pipeline Configuration Management

*   Implement a mechanism to switch between different checkpoints and their corresponding pipeline configurations. The supported checkpoints will be: `dev`, `dev-fp4`, `dev-fp8`, `distilled`, and `distilled-fp8`.

## Progress

### `ltx-video-distilled-tester-dev` Analysis (Completed)

*   Conducted a deep-dive analysis of the `ltx-video-distilled-tester-dev` codebase.
*   Added extensive comments to the following files to document performance and quality-related settings, with a focus on video stitching and continuation:
    *   `inference.py`
    *   `app.py`
    *   `configs/ltxv-13b-0.9.8-distilled.yaml`
    *   `ltx_video/pipelines/pipeline_ltx_video.py`
    *   `ltx_video/pipelines/crf_compressor.py`
*   Created `review_summary.md` to summarize the findings of the analysis.

### `ltx-2-distilled` Refactoring and API Preparation (Completed)

*   Refactored the `ltx-2-distilled` space to prepare for stitching and API capabilities.
*   **Modularization**: Decoupled the core video generation logic from the Gradio UI by creating `video_generator.py`.
*   **Stitching**: Integrated video stitching functionality using `moviepy`.
*   **API Development**:
    *   Created `api.py` using FastAPI to expose the video generation and stitching logic.
    *   Implemented `/generate` and `/stitch` endpoints.
    *   Designed the `/stitch` endpoint to be extensible for handling custom variables per clip.
*   **Documentation**: Created `README_API.md` to document the new structure and provide instructions on how to use the API.
