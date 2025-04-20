from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
import copy, os, torch, gc

cache_dir = "/speed-scratch/ra_mdash/tmp/huggingface"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

HUGGINGFACE_TOKEN = ""

torch.cuda.set_device(0)
gc.collect()
torch.cuda.empty_cache()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, # Compute dtype
    bnb_4bit_use_double_quant=True,
    # llm_int8_enable_fp32_cpu_offload=True # Keep this enabled for offloading
)

# --- Tokenizer ---
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    "facebook/Self-taught-evaluator-llama3.1-70B",
    subfolder="dpo_model",
    cache_dir=cache_dir,
    use_auth_token=HUGGINGFACE_TOKEN
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("Set pad_token to eos_token")

# --- Model Loading ---
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "facebook/Self-taught-evaluator-llama3.1-70B",
    subfolder="dpo_model",
    device_map="auto",                   # Keep using auto device mapping
    cache_dir=cache_dir,
    use_auth_token=HUGGINGFACE_TOKEN,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)
print("Model loaded.")
print(f"Model footprint: {model.get_memory_footprint() / 1e9:.2f} GB") # Check memory usage
print("Device Map:")
print(model.hf_device_map)

# --- Prompt Setup ---
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


# --- Inference ---
print("Tokenizing input...")
tokenizer.padding_side = "left" # Set padding side for batch generation
tokenized_input = tokenizer.apply_chat_template(
    conversation,
    return_tensors="pt",
    add_generation_prompt=True # Add prompt for generation
).to(model.device) # Move to the device where the model's input embeddings are

print(f"Input tensor device: {tokenized_input.device}")

print("Generating judgement...")
# Generation Config
gen_cfg = GenerationConfig(
    max_new_tokens=1024, # Generate up to 256 new tokens
    do_sample=True,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id # Also good to set eos token id
)

# Ensure model config also knows pad token id if needed by generate
model.config.pad_token_id = tokenizer.pad_token_id

with torch.no_grad(): # Use no_grad for inference
    judgement = model.generate(
        tokenized_input,
        generation_config=gen_cfg
    )

# --- Decode and Print ---
print("Decoding output...")
# Decode only the generated part, skipping input tokens and special tokens
output_tokens = judgement[0, tokenized_input.shape[1]:]
judgement_text = tokenizer.decode(output_tokens, skip_special_tokens=True)

print("\n--- Judgement ---")
print(judgement_text)
print("--- End Judgement ---")