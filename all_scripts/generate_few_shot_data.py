import argparse, json, re, time, os, gc, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import time
from google import genai
from google.genai import types

from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import requests

# Set the environment variable to help mitigate fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Force use of GPU device 0 and clear any unused memory
torch.cuda.set_device(0)
gc.collect()
torch.cuda.empty_cache()

# Set Gemini API key
os.environ["GEMINI_API_KEY"] = ""

HUGGINGFACE_TOKEN = ""
cache_dir = "/speed-scratch/ra_mdash/tmp/huggingface"

model_id = "google/gemma-3-27b-it"
    
model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, device_map="auto", cache_dir=cache_dir,
    
).eval()

processor = AutoProcessor.from_pretrained(model_id)

generation_args = {
    "max_new_tokens": 50,
    "temperature": 0.1,
    "do_sample": True,
}

# Load the data mapping
with open("project_sample_mapping.json", "r") as f:
    project_data = json.load(f)

# Storage for generated reasonings
reasoning_results = {}

# Prompt template
def build_prompt(example):
    base_prompt = f"""You are an AI vulnerability analyst.

Project:
{example.get("project", "N/A")}

Commit Message:
{example.get("commit_message", "N/A")}

Code Snippet:
{example.get("func", "N/A")}

"""
    if example["target"] == 1:
        base_prompt += f"""This code is vulnerable. Based on the information provided, explain in 2 sentences why this code is vulnerable. You may also consider this CVE description:

CVE Description:
{example.get("cve_desc", "N/A")}

Answer:"""
    else:
        base_prompt += """This code is NOT vulnerable. Based on the provided information, explain in 2 sentences why this code is safe or not flagged as vulnerable.

Answer:"""
    return base_prompt
    

def initiate_model(model_id):
    model_name = model_id  # e.g. "bigcode/starcoder2-3b"
    
    # Create a BitsAndBytes configuration for 4-bit loading with compute_dtype set to float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,  # Use FP16 for computations
        # Optionally, you can also set:
        # bnb_4bit_use_double_quant=True,
        # bnb_4bit_quant_type="nf4"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        cache_dir=cache_dir, 
        use_auth_token=HUGGINGFACE_TOKEN
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=cache_dir,
        use_auth_token=HUGGINGFACE_TOKEN,
        device_map="auto",
        quantization_config=bnb_config  # Pass the configuration here
    )
    model.to('cuda')
    model.eval()  # Set model to evaluation mode
    # generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    # return generator

    return model, tokenizer



def run_generation(prompt):
    
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an AI vulnerability analyst."}]
        },
        {
            "role": "user",
            "content": [
                # {"type": "image", "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt", do_sample=True
    ).to(model.device, dtype=torch.bfloat16)
    
    input_len = inputs["input_ids"].shape[-1]
    
    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        generation = generation[0][input_len:]
    
    decoded = processor.decode(generation, skip_special_tokens=True)
    
    print(decoded)
    return decoded
        

# Function to call Gemini and get reasoning
def get_reasoning(prompt, model, tokenizer):
    
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids
    input_ids = input_ids.to('cuda')
    
    # Generate the output
    with torch.no_grad():
        output_ids = model.generate(input_ids, **generation_args)
    
    # Determine the length of the input prompt
    input_length = input_ids.shape[1]
    
    # Slice the output to exclude the input prompt and decode
    generated_text = tokenizer.decode(output_ids[0, input_length:], skip_special_tokens=True)

    return generated_text



if __name__ == "__main__":
    
    count = 0

    # model, tokenizer = initiate_model(model_id)
    # Generate reasoning for each example
    for project, targets in project_data.items(): 
        torch.cuda.empty_cache()
        gc.collect()
        if count < 2:
            reasoning_results[project] = { "0": [], "1": [] }
            for target_str, examples in targets.items():
                for example in examples:
                    prompt = build_prompt(example)
                    print(prompt)
                    # print(f"\nGenerating reasoning for project: {project}, target: {target_str}, idx: {example['idx']}")
                    # reasoning = get_reasoning(prompt, model, tokenizer)
                    reasoning = run_generation(prompt)
                    reasoning_results[project][target_str].append({
                        "idx": example["idx"],
                        "reasoning": reasoning,
                        "func": example["func"],
                        "commit_message": example["commit_message"],
                        "cve_desc": example.get("cve_desc", ""),
                        "target": example["target"]
                    })
                    time.sleep(1)  # Sleep to be kind to API rate limits
    
    # Save reasoning results
    with open("reasoning_output.json", "w") as out_file:
        json.dump(reasoning_results, out_file, indent=2)
    
    print("✅ All reasonings generated and saved to 'reasoning_output.json'")
