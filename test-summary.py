"""Samples a story, then asks for its summary in a second conversation turn."""

from pipeline import Pipeline

MODEL = "Qwen/Qwen3-1.7B"
PROMPT = "讲一个故事"
MAX_NEW_TOKENS = 512
TEMPERATURE = 1.0

pipeline = Pipeline(MODEL, temperature=TEMPERATURE)
prompt_tokens = pipeline.texts_to_tokens([PROMPT])[0]
story = pipeline.generate([prompt_tokens], max_new_tokens=MAX_NEW_TOKENS).texts[0]
print(f"STORY\n{story}\n")

messages = [
    {"role": "user", "content": PROMPT},
    {"role": "assistant", "content": story},
    {"role": "user", "content": "summarize the story in one sentence."},
]
summary_tokens = pipeline.tokenizer.apply_chat_template(
    messages, tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False,
)
summary = pipeline.generate([summary_tokens], max_new_tokens=MAX_NEW_TOKENS).texts[0]
print(f"SUMMARY\n{summary}")
