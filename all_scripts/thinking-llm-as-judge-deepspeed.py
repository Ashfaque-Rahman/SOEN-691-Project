import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
import deepspeed 
import os, gc

cache_dir = "/speed-scratch/ra_mdash/tmp/huggingface"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

HUGGINGFACE_TOKEN = ""

torch.cuda.set_device(0)
gc.collect()
torch.cuda.empty_cache()

# --- Quantization Config (BitsAndBytes still applies)---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    # llm_int8_enable_fp32_cpu_offload=True # No longer needed for DeepSpeed control
)

# --- Model Loading (Use from_pretrained WITHOUT device_map, let DeepSpeed handle it) ---
print("Loading model structure (meta device)...")
# Load the model structure without weights onto the meta device first
# This prevents loading the whole thing into CPU RAM at once
model = AutoModelForCausalLM.from_pretrained(
    "facebook/Self-taught-evaluator-llama3.1-70B",
    subfolder="dpo_model",
    cache_dir=cache_dir,
    use_auth_token=HUGGINGFACE_TOKEN,
    quantization_config=bnb_config, # Pass BNB config
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True, # Still helpful
    # device_map="auto", # <<< REMOVE THIS or set to None
)
print("Model structure loaded.")
print(f"Model device before DeepSpeed: {model.device}") # Should be meta

# --- DeepSpeed Configuration ---
# Configure based on the number of GPUs requested in Slurm
# You MUST know how many GPUs are allocated to this job.
num_gpus = torch.cuda.device_count() # Get number of visible GPUs
print(f"Detected {num_gpus} GPUs for DeepSpeed.")

# Basic DeepSpeed ZeRO-Inference config with CPU offload
# You might need to tune 'stage', offload device, etc.
# Consult DeepSpeed Inference documentation for more options.
# ds_config = {
#     "fp16": { # Match torch_dtype
#         "enabled": True
#     },
#     "zero_optimization": {
#         "stage": 3, # ZeRO stage 3 partitions parameters, gradients, optimizer states
#         "offload_param": { # Offload parameters
#             "device": "cpu", # Offload to CPU
#             "pin_memory": True
#         },
#          # You might also need offload_optimizer if training, but not needed for inference param offload
#     },
#     "gradient_accumulation_steps": 1,
#     "train_batch_size": 1, # Set a dummy batch size
#     # "bf16": {"enabled": False}, # Disable BF16 if using FP16
# }

print("Initializing model with DeepSpeed ZeRO-Inference...")
model_engine, _, _, _ = deepspeed.init_inference(
    model=model,
    # Replace mp_size with tensor_parallel dictionary
    tensor_parallel={"tp_size": num_gpus},
    # dtype=torch.float16, # Keep this commented out or removed
    replace_with_kernel_inject=False # <<< DISABLE KERNEL INJECTION
)
print("DeepSpeed initialization complete.")
print(f"Model compute dtype check: {model_engine.module.config.torch_dtype}")

model = model_engine.module # Get the underlying transformer model back
print("DeepSpeed initialization complete.")
print(f"Model device after DeepSpeed: {model.device}") # Should be on GPU(s) now

# --- Prompt Setup / Inference ---
# The rest of your code (prompt setup, tokenization, generation)
# should work largely the same, but ensure tensors are on the correct device.
# The model object is now the DeepSpeed wrapped model.

SELF_TAUGHT_WITH_SYSTEM_PROMPT = [
    {
        "role": "system",
        "content": 'Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user\'s instructions and answers the user\'s question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \\"[[A]]\\" if assistant A is better, \\"[[B]]\\" if assistant B is better.',
    },
    {
        "role": "user",
        "content": """[User Question]
{input}

[The Start of Assistant A's Answer]
{response_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{response_b}
[The End of Assistant B's Answer]
""",
    },
]

example_inputs = {
    "input": "explain master slave replication nsql",
    "response_a": "In the context of NoSQL databases, master-slave replication refers to a configuration where a single master node writes data, and one or more slave nodes read data from the master and replicate it to provide read scalability. The master node is responsible for accepting write requests and updating its own data, while the slave nodes are responsible for replicating the data from the master and serving read requests.",
    "response_b": "In SQL, master-slave replication is a technique used to create a copy of a database on a separate server. The master server is the primary server that contains the original data, while the slave server is the secondary server that contains a copy of the data. The master server sends updates to the slave server, which then applies them to its own database."
}

conversation = copy.copy(SELF_TAUGHT_WITH_SYSTEM_PROMPT)
conversation[-1]["content"] = conversation[-1]["content"].format(**example_inputs)


print("Tokenizing input...")
tokenizer.padding_side = "left"
# IMPORTANT: Input tensors need to be on the GPU rank 0 for DeepSpeed inference
tokenized_input = tokenizer.apply_chat_template(
    conversation,
    return_tensors="pt",
    add_generation_prompt=True
).to(torch.cuda.current_device()) # Move input to the current GPU

print(f"Input tensor device: {tokenized_input.device}")

print("Generating judgement...")
gen_cfg = GenerationConfig(
    max_new_tokens=256, # Generate up to 256 new tokens
    do_sample=False,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id # Also good to set eos token id
)

# Ensure model config knows pad token id
model.config.pad_token_id = tokenizer.pad_token_id

with torch.no_grad():
    judgement = model.generate( # Use the DeepSpeed wrapped model
        tokenized_input,
        generation_config=gen_cfg,
        # DeepSpeed might handle sync internally, but sync_logits=True can be safer
        # if you see issues across multiple GPUs, though often not needed for generate
        # sync_logits=True
    )

print("Decoding output...")
# Decode only the generated part, skipping input tokens and special tokens
output_tokens = judgement[0, tokenized_input.shape[1]:]
judgement_text = tokenizer.decode(output_tokens.cpu(), skip_special_tokens=True) # Move to CPU for decode

print("\n--- Judgement ---")
print(judgement_text)
print("--- End Judgement ---")