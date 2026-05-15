"""VLM-backed success evaluator for grasp / manipulation tasks.

Calls Qwen-VL via the DashScope OpenAI-compatible endpoint with a
fixed prompt and a single screenshot; returns 1 / 0. The fallback
SmolVLM block below the API call is unreachable in practice (the
``try`` returns or raises before it runs) — it's kept as inert
reference code for environments where DashScope is unavailable, until
someone wires up a proper local-model branch.
"""

import torch
from PIL import Image
import numpy as np
import os
import base64
import tempfile
from openai import OpenAI
from transformers.image_utils import load_image

# Global, lazily-initialised Qwen-VL client.
_qwen_client = None


def _get_qwen_client():
    """Lazy-load the Qwen-VL client. Reads DASHSCOPE_API_KEY from the env."""
    global _qwen_client
    if _qwen_client is None:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("Environment variable DASHSCOPE_API_KEY is not set for Qwen-VL API.")
        _qwen_client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    return _qwen_client


def _pil_to_base64(pil_image):
    """Encode a PIL Image as PNG-base64 (the format the chat API wants)."""
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        pil_image.save(tmp.name)
        with open(tmp.name, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")


def evaluate_task_success(image, task_description):
    """Ask Qwen-VL whether a robot succeeded at the described task.

    Args:
        image: PIL.Image, numpy array, torch.Tensor, or file path. Tensor and
            numpy inputs are normalised to PIL.Image internally so the API
            call sees the same input shape regardless of caller.
        task_description: short English sentence; gets quoted into the prompt.

    Returns:
        ``1`` (success) or ``0`` (failure). Any exception or ambiguous model
        output falls back to ``0`` so the caller never sees a non-binary
        result.
    """

    # Coerce the image into a PIL.Image.
    if isinstance(image, str):
        image = load_image(image)
    elif isinstance(image, torch.Tensor):
        # ManiSkill returns [B, H, W, C]; drop the leading 1.
        if image.dim() == 4 and image.shape[0] == 1:
            image = image.squeeze(0)
        image = image.cpu().numpy()
        if image.dtype != np.uint8:
            # float in [0, 1] → uint8 in [0, 255]; otherwise just cast.
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        image = Image.fromarray(image)
    elif isinstance(image, Image.Image):
        pass
    elif isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        # BGR → RGB for 3-channel numpy arrays (the convention OpenCV uses).
        image = Image.fromarray(image[..., ::-1]) if image.shape[-1] == 3 else Image.fromarray(image)

    # Drop a debug copy so the prompt input can be eyeballed if a result
    # looks suspect.
    debug_image_path = "vlm_debug_image.png"
    image.save(debug_image_path)
    print(f"Debug: VLM input image saved to {debug_image_path}")

    image_b64 = _pil_to_base64(image)
    client = _get_qwen_client()

    user_prompt = f"Task: {task_description}. Has the robot succeeded?"

    try:
        completion = client.chat.completions.create(
            model="qwen-vl-max",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert robotic vision evaluator. "
                        "Respond ONLY with 'Success' or 'Failure'. "
                        "Do not explain, do not add punctuation, do not use Chinese."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ],
            max_tokens=5,
            temperature=0.0,
            top_p=1.0,
        )

        raw_output = completion.choices[0].message.content.strip()

        # Strict parsing first; loosen only if the model didn't follow the rule.
        if raw_output == "Success":
            return 1
        elif raw_output == "Failure":
            return 0
        else:
            # Fallback: substring match. Bias to failure on ambiguity.
            output_lower = raw_output.lower()
            if "success" in output_lower and "failure" not in output_lower:
                return 1
            else:
                return 0

    except Exception as e:
        print(f"[Qwen-VL API Error] {e}")
        return 0  # Conservative: treat API errors as failure.

    # ----- Inert SmolVLM fallback below -----
    # Kept for reference; the early return / raise above means this is
    # unreachable in the current API-driven path. Convert to a real branch
    # if you ever wire up a local VLM.
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
        temp_image_path = temp_file.name
        image.save(temp_file)

        debug_image_path = "vlm_debug_image.png"
        image.save(debug_image_path)
        print(f"Debug: VLM input image saved to {debug_image_path}")

    try:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "path": temp_image_path},
                    {"type": "text", "text": f"""The robot attempted to {task_description}. Is the grasp successful? Answer "SUCCESS" or "FAILURE" only."""}
                ]
            }
        ]

        # Prepare inputs for SmolVLM/Idefics3
        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(DEVICE)
    finally:
        # Clean up the temporary input file.
        if os.path.exists(temp_image_path):
            os.unlink(temp_image_path)

    # Generate outputs with constrained generation
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.1,
            do_sample=True,
            top_p=0.9,
            pad_token_id=processor.tokenizer.eos_token_id
        )

    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    response = generated_texts[0].strip()

    # Parse the ASSISTANT segment of the chat-template output, stripping
    # any trailing HTML / extra lines the template might inject.
    if "ASSISTANT:" in response.upper():
        assistant_start = response.upper().find("ASSISTANT:")
        assistant_part = response[assistant_start + len("ASSISTANT:"):].strip()
        assistant_part = assistant_part.split('\n')[0].split('<')[0].strip().upper()
        if "SUCCESS" in assistant_part and "FAILURE" not in assistant_part:
            return 1
        elif "FAILURE" in assistant_part:
            return 0

    # Backup parsing path with looser keyword matching.
    response_upper = response.upper()
    if "SUCCESS" in response_upper and "FAILURE" not in response_upper:
        return 1
    elif "FAILURE" in response_upper:
        return 0
    else:
        # Ambiguous output — score positive vs negative keywords.
        success_keywords = ["YES", "COMPLETE", "DONE", "ACHIEVED", "GRASPED", "GRABBED"]
        failure_keywords = ["NO", "INCOMPLETE", "FAILED", "NOT", "MISSING"]

        response_upper = response.upper()
        success_score = sum(1 for keyword in success_keywords if keyword in response_upper)
        failure_score = sum(1 for keyword in failure_keywords if keyword in response_upper)

        if success_score > failure_score:
            return 1
        elif failure_score > success_score:
            return 0
        else:
            # Final fallback: treat genuinely ambiguous outputs as failure.
            return 0


def evaluate_grasp_success(image, object_name, part_name):
    """Convenience wrapper that builds the task description from object/part.

    Args:
        image: same image types as :func:`evaluate_task_success`.
        object_name: e.g. ``"mug"`` or ``"bottle"``.
        part_name: e.g. ``"handle"`` or ``"cap"``.

    Returns:
        ``0`` (failure) or ``1`` (success).
    """
    task_description = f"grasp the {part_name} of the {object_name}"
    return evaluate_task_success(image, task_description)


# Example usage / micro-benchmark.
if __name__ == "__main__":
    import time

    print("Using Qwen-VL-Max via DashScope API (no local model loading)...")

    # Sanity check + simple latency probe.
    image_path = "/home/robot/workspace/MetaFine/vlm_debug_image.png"

    print("\nTesting inference speed with 5 runs...")
    inference_times = []

    for i in range(5):
        start_time = time.time()
        result = evaluate_grasp_success(image_path, "bottle", "cap")
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        print(f"Run {i+1}: {inference_time:.3f}s, result: {result}")

    avg_inference_time = sum(inference_times) / len(inference_times)
    print(f"\nAverage inference time: {avg_inference_time:.3f}s")
    print(f"Inference speed: {1.0/avg_inference_time:.1f} inferences/second")
