"""
Candidate modality vectors from Qwen2-VL hidden states.

For a set of PixelProse (image, caption) pairs, cache per-layer hidden states,
then build a per-layer candidate vector = mean(text acts) - mean(image acts).
"""

import io

import requests
import torch
from datasets import load_dataset
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


# ---------------------------------------------------------------------------
# Load Qwen2-VL
# ---------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.bfloat16

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
).to(device)
model.eval()

processor = AutoProcessor.from_pretrained(MODEL_ID)
print(f"Model loaded on {device}.")


# ---------------------------------------------------------------------------
# Collect 50 (image, caption) pairs from PixelProse (detailed vlm_caption)
# ---------------------------------------------------------------------------
N_PAIRS = 50
PROMPT_TEXT = "What is in the image?"

# tomg-group-umd/pixelprose: `vlm_caption` = long detailed Gemini caption,
# `url` = image to download. Stream + shuffle so we get a random set.
dataset = load_dataset("tomg-group-umd/pixelprose", split="train", streaming=True)
dataset = dataset.shuffle(seed=42, buffer_size=1000)


def fetch_image(url, timeout=10):
    """Download image from URL -> PIL. Return None if dead/broken."""
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None


pairs = []
for sample in dataset:
    if len(pairs) >= N_PAIRS:
        break
    image = fetch_image(sample["url"])
    if image is None:
        continue  # dead url, skip
    caption = sample["vlm_caption"]
    if not caption:
        continue  # no caption, skip
    pairs.append((image, caption))
print(f"Collected {len(pairs)} image-caption pairs.")


def build_inputs(image, caption):
    """Joint input: image + caption + prompt in one (Qwen2-VL chat template).

    Returns (inputs, caption_mask) where caption_mask marks exactly the caption
    tokens in the full sequence (not the prompt/chat/image tokens).
    """
    text = f"{caption}\n{PROMPT_TEXT}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text},
            ],
        }
    ]
    chat_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[chat_text], images=[image], return_tensors="pt").to(device)

    # --- caption_mask: which tokens are the caption (exclude prompt/chat/image) ---
    # Tokenize the text chunk alone with char offsets; the caption occupies chars
    # [0, len(caption)) of it. Then find that token subsequence in the full input.
    enc = processor.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    text_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    cap_end_char = len(caption)
    cap_positions = [
        i for i, (a, b) in enumerate(offsets) if b <= cap_end_char and b > a
    ]

    full = inputs["input_ids"][0].tolist()
    tlen = len(text_ids)
    start = next(
        (i for i in range(len(full) - tlen + 1) if full[i : i + tlen] == text_ids),
        None,
    )

    caption_mask = torch.zeros(len(full), dtype=torch.bool, device=device)
    if start is not None and cap_positions:
        caption_mask[start + cap_positions[0] : start + cap_positions[-1] + 1] = True

    return inputs, caption_mask


# ---------------------------------------------------------------------------
# Collect per-layer activations, split by modality, across all pairs
# ---------------------------------------------------------------------------
# text_rows[L]  = list of 50 matrices, each [n_caption_tokens_of_that_pair, d_model]
# image_rows[L] = list of 50 matrices, each [n_vision_tokens_of_that_pair, d_model]
# Layers are indexed 1..N by block number: cache[L] = hidden_states[L] = residual
# stream after block L. We skip hidden_states[0] (embeddings) and go through the
# last layer (N).
N_LAYERS = model.config.num_hidden_layers   # 28 for Qwen2-VL-2B
image_token_id = model.config.image_token_id

text_rows = {L: [] for L in range(1, N_LAYERS + 1)}
image_rows = {L: [] for L in range(1, N_LAYERS + 1)}

for i, (image, caption) in enumerate(pairs):
    inputs, caption_mask = build_inputs(image, caption)
    vision_mask = inputs["input_ids"][0] == image_token_id

    # forward pass ON GPU
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # hidden_states = (embeddings, after-block-1, ..., after-block-N).
    # Start at block 1 (first computed activation), go through block N (output).
    hs = outputs.hidden_states
    for L in range(1, N_LAYERS + 1):
        acts = hs[L][0]                                 # [seq_len, d_model] on GPU
        text_rows[L].append(acts[caption_mask].cpu())   # STORE on CPU
        image_rows[L].append(acts[vision_mask].cpu())   # STORE on CPU

    del outputs  # free GPU activations before next pair
    print(
        f"[{i + 1}/{len(pairs)}] "
        f"n_cap={int(caption_mask.sum())}, n_vis={int(vision_mask.sum())}"
    )


# ---------------------------------------------------------------------------
# Candidate modality vectors: per layer, mean(text) - mean(image)
# ---------------------------------------------------------------------------
# Per-pair mean FIRST (each pair -> one vector by averaging its tokens), THEN
# average the 50 per-pair vectors. This weights every image equally, so long
# captions don't dominate. mu_text, mu_img are [d_model]; their difference is the
# candidate modality vector. (float() so fp16 means don't lose precision.)
candidates = {}
for L in range(1, N_LAYERS + 1):
    # each matrix [n_tokens_of_pair, d_model] -> [d_model] per pair, then stack 50
    text_per_pair = torch.stack([m.float().mean(dim=0) for m in text_rows[L]])   # [50, d_model]
    image_per_pair = torch.stack([m.float().mean(dim=0) for m in image_rows[L]]) # [50, d_model]
    mu_text = text_per_pair.mean(dim=0)   # [d_model]
    mu_img = image_per_pair.mean(dim=0)   # [d_model]
    candidates[L] = mu_text - mu_img      # text - image

CANDIDATE_LAYERS = [3, 14, 26]
v3, v14, v26 = (candidates[L] for L in CANDIDATE_LAYERS)

for L, v in zip(CANDIDATE_LAYERS, (v3, v14, v26)):
    print(f"candidate v{L}: shape={tuple(v.shape)}, norm={v.norm():.3f}")

