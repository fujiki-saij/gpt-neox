import transformers

model_path = ""
model = transformers.AutoModelForCausalLM.from_pretrained(model_path)

tokenizer_path = "EleutherAI/gpt-neox-20b"
tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.paddding_side = "left"

prompts = [
    "The Statue of Liberty was a gift from",
    "What is the difference between a list and a tuple in Python?",
    "Write a function that checks whether a string is a palindrome or not.\ndef is_palindrome(s):",
    "",
    "",
]

batch_encoding = tokenizer(prompts, return_tensors="pt", padding=True)

print(f"Generating {len(prompts)} prompts...")
samples = model.generate(
    **batch_encoding,
    max_new_tokens=64,
    temperature=0.4,
    do_sample=True,
)

for i, sample in enumerate(samples):
    print(f"Prompt: {prompts[i]}")
    print(f"⇥ {tokenizer.decode(sample, skip_special_tokens=True)}")
    print()
