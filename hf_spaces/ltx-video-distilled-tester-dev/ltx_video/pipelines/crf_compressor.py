import av
import torch
import io
import numpy as np


def _encode_single_frame(output_file, image_array: np.ndarray, crf):
    container = av.open(output_file, "w", format="mp4")
    try:
        stream = container.add_stream(
            "libx264", rate=1, options={"crf": str(crf), "preset": "veryslow"}
        )
        stream.height = image_array.shape[0]
        stream.width = image_array.shape[1]
        av_frame = av.VideoFrame.from_ndarray(image_array, format="rgb24").reformat(
            format="yuv420p"
        )
        container.mux(stream.encode(av_frame))
        container.mux(stream.encode())
    finally:
        container.close()


def _decode_single_frame(video_file):
    container = av.open(video_file)
    try:
        stream = next(s for s in container.streams if s.type == "video")
        frame = next(container.decode(stream))
    finally:
        container.close()
    return frame.to_ndarray(format="rgb24")


def compress(image: torch.Tensor, crf=29):
    # Gemini-comment: This function is a very clever and unconventional way to preprocess a conditioning
    # image. It applies lossy video compression to a single frame.
    # How it works:
    # 1. Takes an image tensor.
    # 2. Encodes it into a single-frame mp4 video in memory using the libx264 codec.
    # 3. The `crf` (Constant Rate Factor) value controls the quality. A higher CRF means more
    #    compression and lower quality. `crf=29` is a medium-to-high compression level.
    # 4. Immediately decodes the video frame back into a tensor.
    #
    # The purpose is not to save memory, but to alter the characteristics of the image. This can be
    # seen as a form of "video-like" regularization or pre-processing.
    # Possible reasons for this:
    # - To make a clean conditioning image better match the distribution of compressed video frames
    #   that the model may have been trained on.
    # - To reduce high-frequency noise and detail in the conditioning image, which might otherwise
    #   distract the model.
    # - It's a more sophisticated alternative to a simple Gaussian blur.
    if crf == 0:
        return image

    image_array = (
        (image[: (image.shape[0] // 2) * 2, : (image.shape[1] // 2) * 2] * 255.0)
        .byte()
        .cpu()
        .numpy()
    )
    with io.BytesIO() as output_file:
        _encode_single_frame(output_file, image_array, crf)
        video_bytes = output_file.getvalue()
    with io.BytesIO(video_bytes) as video_file:
        image_array = _decode_single_frame(video_file)
    tensor = torch.tensor(image_array, dtype=image.dtype, device=image.device) / 255.0
    return tensor
