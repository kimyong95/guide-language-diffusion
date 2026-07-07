from transformers import DiffusionGemmaForBlockDiffusion, AutoProcessor
import time

import torch
from accelerate import Accelerator
acc = Accelerator()

MODEL_ID = "google/diffusiongemma-26B-A4B-it"

max_memory = {i: torch.cuda.get_device_properties(i).total_memory for i in range(acc.process_index, torch.cuda.device_count(), acc.num_processes)}

# Load model
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = DiffusionGemmaForBlockDiffusion.from_pretrained(
    MODEL_ID,
    dtype="auto",
    device_map="auto",
    max_memory=max_memory,
)
# Prompt
message = [
    {"role": "user", "content": "Why is the sky blue?"}
]


# Process input
input_ids = processor.apply_chat_template(
    message,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)
output = model.generate(**input_ids, max_new_tokens=512)

# Parse output
text = processor.decode(output[0], skip_special_tokens=False)