# Project Plan

This repository contains the source code for several Hugging Face Spaces related to video generation. The goal is to experiment with and improve upon the existing models and applications.

## Repository Structure

The repository is organized to maintain the original codebases while allowing for parallel development and testing of new features.

*   **Original Code:** The original Python code for the Hugging Face Spaces will be kept in their respective directories (e.g., `hf_spaces/ltx-2-distilled`) and will remain unmodified. This serves as a stable baseline.
*   **Experimental Testers:** For development, we will create copies of the original applications, named `tester-a`, `tester-b`, and so on. These copies will be used to implement and test new features.

## Development Roadmap

### 1. Feature Porting: `ltx-2-distilled` to `ltx-2`

The primary goal is to merge features from the `ltx-2-distilled` space into the main `ltx-2` space. This includes, but is not limited to:

*   **Stitching:** Implement the video stitching functionality from `ltx-2-distilled` into the `ltx-2` pipeline.

### 2. Enhancements for `ltx-1-dev`

We will also focus on improving the `ltx-1-dev` model/space. The specific enhancements will be determined after an initial analysis of the existing codebase and its limitations.

### 3. General Improvements

*   **Code Quality:** Refactor and improve the code quality of the new tester applications.
*   **Documentation:** Update documentation to reflect the changes and new features.

## Progress

### `ltx-video-distilled-tester` Analysis (Completed)

*   Conducted a deep-dive analysis of the `ltx-video-distilled-tester` codebase.
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
