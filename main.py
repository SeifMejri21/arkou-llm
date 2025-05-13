from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding
import torch
from time import time

model_name = "Qwen/Qwen3-1.7B"
start = time()

torch.backends.cudnn.benchmark = True


# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
collator = DataCollatorWithPadding(tokenizer, padding=True)

st = time()

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto"
)
print(f"loading model took: {round(time()-st,4)}")


# Freeze the first N layers
N = 8  # Adjust based on experimentation
for i, layer in enumerate(model.model.layers):
    if i < N:
        for param in layer.parameters():
            param.requires_grad = False

print(f"Freezing first {N} layers.")

# Check CUDA device details
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB")
    print(f"Memory reserved: {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")

# Prepare the model input
prompt = "Are you Familiar with the island of Djerba in the south of Tunisia."
print("prompt: ", prompt)

messages = [
    {"role": "user", "content": prompt}
]

st = time()
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True
)
print(f"tokenizing took: {round(time()-st,4)}")

# Tokenize and move inputs to the appropriate device
model_inputs = tokenizer([text], return_tensors="pt", padding=True)
model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

# Generate response
st = time()

max_tokens = 1024 
max_tokens = 256 

with torch.inference_mode():
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_tokens,
        temperature=0.7,
        # do_sample=True
        num_beams=3,
        num_return_sequences=3
    )
output_ids = generated_ids[0][len(model_inputs["input_ids"][0]):].tolist()
print(f"ids generation: {round(time()-st,4)}")

st = time()
# Decode and print output
response = tokenizer.decode(output_ids, skip_special_tokens=True)
print(f"Decoding output: {round(time()-st,4)}")
print(response)
print(f"Responding took {round(time() - start, 4)} seconds")
