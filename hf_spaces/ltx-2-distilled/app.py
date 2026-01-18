import sys
from pathlib import Path

# Add packages to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / "packages" / "ltx-pipelines" / "src"))
sys.path.insert(0, str(current_dir / "packages" / "ltx-core" / "src"))

import spaces
import gradio as gr
import numpy as np
from ltx_pipelines.constants import (
    DEFAULT_SEED,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
)
from video_generator import generate_single_clip, stitch_videos

MAX_SEED = np.iinfo(np.int32).max
# Default prompt from docstring example
DEFAULT_PROMPT = "An astronaut hatches from a fragile egg on the surface of the Moon, the shell cracking and peeling apart in gentle low-gravity motion. Fine lunar dust lifts and drifts outward with each movement, floating in slow arcs before settling back onto the ground. The astronaut pushes free in a deliberate, weightless motion, small fragments of the egg tumbling and spinning through the air. In the background, the deep darkness of space subtly shifts as stars glide with the camera's movement, emphasizing vast depth and scale. The camera performs a smooth, cinematic slow push-in, with natural parallax between the foreground dust, the astronaut, and the distant starfield. Ultra-realistic detail, physically accurate low-gravity motion, cinematic lighting, and a breath-taking, movie-like shot."

@spaces.GPU(duration=300)
def generate_video_for_gradio(
    input_image,
    prompt: str,
    duration: float,
    enhance_prompt: bool = True,
    seed: int = 42,
    randomize_seed: bool = True,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    clips_list: list = [],
    progress=gr.Progress(track_tqdm=True)
):
    """Generate a video based on the given parameters and updates the UI."""
    output_path, seed_val = generate_single_clip(
        input_image=input_image,
        prompt=prompt,
        duration=duration,
        enhance_prompt=enhance_prompt,
        seed=seed,
        randomize_seed=randomize_seed,
        height=height,
        width=width,
    )
    
    if output_path:
        updated_clips_list = clips_list + [output_path]
        return output_path, seed_val, updated_clips_list, f"Clips created: {len(updated_clips_list)}"
    else:
        return None, seed, clips_list, f"Clips created: {len(clips_list)}"

def stitch_videos_for_gradio(clips_list):
    if not clips_list or len(clips_list) < 2:
        gr.Warning("You need at least two clips to stitch them together!")
        return None
    
    final_video_path = stitch_videos(clips_list)
    return final_video_path

# Create Gradio interface
with gr.Blocks(title="LTX-2 Video Distilled 🎥🔈") as demo:
    clips_state = gr.State([])

    gr.Markdown("# LTX-2 Distilled 🎥🔈: The First Open Source Audio-Video Model")
    gr.Markdown("Fast, state-of-the-art video & audio generation with [Lightricks LTX-2 TI2V model](https://huggingface.co/Lightricks/LTX-2) and [distillation LoRA](https://huggingface.co/Lightricks/LTX-2/blob/main/ltx-2-19b-distilled-lora-384.safetensors) for accelerated inference. Read more: [[model]](https://huggingface.co/Lightricks/LTX-2), [[code]](https://github.com/Lightricks/LTX-2)")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="Input Image (Optional)",
                type="pil"
            )

            prompt = gr.Textbox(
                label="Prompt",
                info="for best results - make it as elaborate as possible",
                value="Make this image come alive with cinematic motion, smooth animation",
                lines=3,
                placeholder="Describe the motion and animation you want..."
            )
            with gr.Row():
                duration = gr.Slider(
                    label="Duration (seconds)",
                    minimum=1.0,
                    maximum=10.0,
                    value=3.0,
                    step=0.1
                )
                enhance_prompt = gr.Checkbox(
                        label="Enhance Prompt",
                        value=True
                    )

            generate_btn = gr.Button("Generate Video", variant="primary", size="lg")
            
            with gr.Accordion("Stitching", open=True):
                clip_counter_display = gr.Markdown("Clips created: 0")
                stitch_btn = gr.Button("Stitch Clips", variant="secondary")
                clear_clips_btn = gr.Button("Clear Clips", variant="stop")


            with gr.Accordion("Advanced Settings", open=False):
                seed = gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=MAX_SEED,
                    value=DEFAULT_SEED,
                    step=1
                )

                randomize_seed = gr.Checkbox(
                    label="Randomize Seed",
                    value=True
                )

                with gr.Row():
                    width = gr.Number(
                        label="Width",
                        value=DEFAULT_WIDTH,
                        precision=0
                    )
                    height = gr.Number(
                        label="Height",
                        value=DEFAULT_HEIGHT,
                        precision=0
                    )

        with gr.Column():
            output_video = gr.Video(label="Last Generated Clip", autoplay=True)
            final_video = gr.Video(label="Stitched Video", autoplay=False)
            
    def clear_clips():
        return [], "Clips created: 0", None, None

    generate_btn.click(
        fn=generate_video_for_gradio,
        inputs=[
            input_image,
            prompt,
            duration,
            enhance_prompt,
            seed,
            randomize_seed,
            height,
            width,
            clips_state,
        ],
        outputs=[output_video, seed, clips_state, clip_counter_display]
    )

    stitch_btn.click(
        fn=stitch_videos_for_gradio,
        inputs=[clips_state],
        outputs=[final_video]
    )
    
    clear_clips_btn.click(
        fn=clear_clips,
        outputs=[clips_state, clip_counter_display, output_video, final_video]
    )

    # Add example
    gr.Examples(
        examples=[
            [
                "kill_bill.jpeg",
                "A low, subsonic drone pulses as Uma Thurman's character, Beatrix Kiddo, holds her razor-sharp katana blade steady in the cinematic lighting. A faint electrical hum fills the silence. Suddenly, accompanied by a deep metallic groan, the polished steel begins to soften and distort, like heated metal starting to lose its structural integrity. Discordant strings swell as the blade's perfect edge slowly warps and droops, molten steel beginning to flow downward in silvery rivulets while maintaining its metallic sheen—each drip producing a wet, viscous stretching sound. The transformation starts subtly at first—a slight bend in the blade—then accelerates as the metal becomes increasingly fluid, the groaning intensifying. The camera holds steady on her face as her piercing eyes gradually narrow, not with lethal focus, but with confusion and growing alarm as she watches her weapon dissolve before her eyes. She whispers under her breath, voice flat with disbelief: 'Wait, what?' Her heartbeat rises in the mix—thump... thump-thump—as her breathing quickens slightly while she witnesses this impossible transformation. Sharp violin stabs punctuate each breath. The melting intensifies, the katana's perfect form becoming increasingly abstract, dripping like liquid mercury from her grip. Molten droplets fall to the ground with soft, bell-like pings. Unintelligible whispers fade in and out as her expression shifts from calm readiness to bewilderment and concern, her heartbeat now pounding like a war drum, as her legendary instrument of vengeance literally liquefies in her hands, leaving her defenseless and disoriented. All sound cuts to silence—then a single devastating bass drop as the final droplet falls, leaving only her unsteady breathing in the dark.",
                5.0,
            ],
            [
                "wednesday.png",
                "A cinematic close-up of Wednesday Addams frozen mid-dance on a dark, blue-lit ballroom floor as students move indistinctly behind her, their footsteps and muffled music reduced to a distant, underwater thrum; the audio foregrounds her steady breathing and the faint rustle of fabric as she slowly raises one arm, never breaking eye contact with the camera, then after a deliberately long silence she speaks in a flat, dry, perfectly controlled voice, “I don’t dance… I vibe code,” each word crisp and unemotional, followed by an abrupt cutoff of her voice as the background sound swells slightly, reinforcing the deadpan humor, with precise lip sync, minimal facial movement, stark gothic lighting, and cinematic realism.",
                5.0,
            ],
            [
                "astronaut.jpg",
                "An astronaut hatches from a fragile egg on the surface of the Moon, the shell cracking and peeling apart in gentle low-gravity motion. Fine lunar dust lifts and drifts outward with each movement, floating in slow arcs before settling back onto the ground. The astronaut pushes free in a deliberate, weightless motion, small fragments of the egg tumbling and spinning through the air. In the background, the deep darkness of space subtly shifts as stars glide with the camera's movement, emphasizing vast depth and scale. The camera performs a smooth, cinematic slow push-in, with natural parallax between the foreground dust, the astronaut, and the distant starfield. Ultra-realistic detail, physically accurate low-gravity motion, cinematic lighting, and a breath-taking, movie-like shot.",
                3.0,
            ]
            
        ],
        fn=generate_video_for_gradio,
        inputs=[input_image, prompt, duration],
        outputs = [output_video, seed, clips_state, clip_counter_display],
        label="Example",
        cache_examples=True,
        cache_mode="lazy",
    )


css = '''
.gradio-container .contain{max-width: 1200px !important; margin: 0 auto !important}
'''
if __name__ == "__main__":
    demo.launch(theme=gr.themes.Citrus(), css=css)
