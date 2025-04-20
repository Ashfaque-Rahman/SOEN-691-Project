#!/bin/sh
import argparse, json, re, time, os, gc, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

# Set the environment variable to help mitigate fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Force use of GPU device 0 and clear any unused memory
torch.cuda.set_device(0)
gc.collect()
torch.cuda.empty_cache()

from prompts import *

HUGGINGFACE_TOKEN = ""
cache_dir = "/speed-scratch/ra_mdash/tmp/huggingface"

generation_args = {
    "max_new_tokens": 50,
    "temperature": 0.1,
    "do_sample": True,
}

# def initiate_model(model_id):
#     model_name = model_id  # e.g., "bigcode/starcoder2-3b"
    
#     tokenizer = AutoTokenizer.from_pretrained(
#         model_name, cache_dir=cache_dir, use_auth_token=HUGGINGFACE_TOKEN
#     )
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         trust_remote_code=True,
#         cache_dir=cache_dir,
#         use_auth_token=HUGGINGFACE_TOKEN,
#         device_map="auto",
#         load_in_4bit=True
#     )
#     model.eval()  # Set model to evaluation mode to avoid storing gradients
    
#     generator = pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         framework="pt"
#     )
#     return generator


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
    model.eval()  # Set model to evaluation mode
    # generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    # return generator
    return model, tokenizer
    
    

def run_on_data(data_path, prompt_template, model, tokenizer, output_file):
    count = 0
    max_test = 10
    
    # start = 396
    # count = start
    # end = 405
    debug = True #False
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    with open(data_path, "r") as f, open(output_file, "a") as output_f:
        for line in f:
            # Clear unused GPU memory and collect garbage
            torch.cuda.empty_cache()
            gc.collect()
            
            # if count >= start and count <= end:
            if count <= max_test:
                if line.strip():
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON: {e}")
                        continue
                    
                    # Extract fields from the JSON line
                    commit_message = data.get("commit_message", "")
                    func = data.get("func", "")
                    target = data.get("target", "")
                    cwe = data.get("cwe", [])
                    cve = data.get("cve", "")
                    cve_desc = data.get("cve_desc", "")
    
                    final_prompt = prompt_template.format(
                        commit_message=commit_message, func=func
                    )

                    messages=[
                        { 'role': 'user', 'content': final_prompt}
                    ]
                    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
                    # tokenizer.eos_token_id is the id of <｜end▁of▁sentence｜>  token
                    
                    with torch.no_grad():
                        outputs = model.generate(inputs, max_new_tokens=512, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
                        output = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    
                    # # Use no_grad to avoid tracking gradients during generation
                    # with torch.no_grad():
                    #     output = generator(final_prompt, **generation_args)
    
                    # if debug:
                    #     print(output[0]['generated_text'])
                    #     print("-------------")
                    
                    try:
                        result_match = re.search(
                            r'Answer:\s*(Yes|No)', output[0]['generated_text'], re.IGNORECASE
                        )
                        result = result_match.group(1) if result_match else "-1"
                    except Exception as e:
                        result = "-1"
                    
                    try:
                        cot_match = re.search(
                            rf'Answer:\s*{result}\s*\n*(.*)',
                            output[0]['generated_text'],
                            re.IGNORECASE | re.DOTALL
                        )
                        cot = cot_match.group(1).strip() if cot_match else output[0]['generated_text']
                    except Exception as e:
                        cot = output[0]['generated_text']
    
                    output_data = {
                        "commit_message": commit_message,
                        "func": func,
                        "target": target,
                        "cwe": cwe,
                        "cve": cve,
                        "cve_desc": cve_desc,
                        "result": result,
                        "cot": cot
                    }
    
                    output_f.write(json.dumps(output_data) + "\n")
                    
                    # Pause briefly and increment count
                    time.sleep(1)
                    count += 1
    
    print(f"Processing complete. Results written to {output_file}")
    
        
def main(data_path, model_id, prompting, output_file):
    if prompting == "few_shot":
        prompt_template = get_few_shot_prompt()
    elif prompting == "zero_shot":
        prompt_template = get_zero_shot_prompt()
    
    # generator = initiate_model(model_id)
    model, tokenizer = initiate_model(model_id)
    
    run_on_data(data_path, prompt_template, model, tokenizer, output_file)
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run PrimeVul processing with specified parameters.")
    
    parser.add_argument("--data_path", type=str,
                        help="Path to the input JSONL data file.")
    parser.add_argument("--model_id", type=str, default="bigcode/starcoder2-3b",
                        help="Identifier of the model to be used (e.g., bigcode/starcoder2-3b).")
    parser.add_argument("--prompting", type=str, default="zero_shot",  # few_shot, zero_shot
                        help="Type of prompting to use (default: zero_shot).")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to the output file. If not provided, a default path is generated.")
    
    args = parser.parse_args()
    
    # Set default paths if not provided
    if args.data_path is None:
        args.data_path = "/speed-scratch/ra_mdash/PrimeVul_v0.1/primevul_valid.jsonl"
    if args.output_file is None:
        model_name = args.model_id.split("/")[1]
        args.output_file = f"/speed-scratch/ra_mdash/results/prime_vul/{model_name}_{args.prompting}.jsonl"
    
    main(args.data_path, args.model_id, args.prompting, args.output_file)
