# 3. Message builders for the two masked passes (+ a joint version for testing later)


def build_caption_only_messages(caption, prompt_text):
    """Image masked: caption + prompt only, no image."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{caption}\n{prompt_text}"},
            ],
        }
    ]


def build_image_only_messages(image, prompt_text):
    """Text masked: image + prompt only, no caption."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]


def build_joint_messages(image, caption, prompt_text):
    """Both modalities present — used only for the probe-testing stage later."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"{caption}\n{prompt_text}"},
            ],
        }
    ]
