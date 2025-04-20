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
    # return model, tokenizer
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return generator
    
    

def run_on_data(data_path, prompt_template, generator, output_file):
    count = 0
    max_test = 5000
    
    # start = 396
    # count = start
    # end = 405
    debug = False #True #False
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(data_path, "r") as f:
        samples = json.load(f)
    
    # with open(data_path, "r") as f, open(output_file, "a") as output_f:
    with open(output_file, "a") as output_f:
        for line in samples:
            # Clear unused GPU memory and collect garbage
            torch.cuda.empty_cache()
            gc.collect()
            
            # if count >= start and count <= end:
            if count < max_test:
                data = line
                # if line.strip():
                #     try:
                #         data = json.loads(line)
                #     except json.JSONDecodeError as e:
                #         print(f"Error decoding JSON: {e}")
                #         continue
                    
                # Extract fields from the JSON line
                project = data.get("project", "")
                commit_message = data.get("commit_message", "")
                func = data.get("func", "")
                target = data.get("target", "")
                cwe = data.get("cwe", [])
                cve = data.get("cve", "")
                cve_desc = data.get("cve_desc", "")
                fallback = data.get("fallback", "")
                # reasoning = data.get("reasoning", "")

                few_shot_examples = data.get("few_shot_samples", "")
                zero_shot_query = prompt_template.format(
                    commit_message=commit_message,
                    func=func
                )
                # print("zero_shot_query")
                # print(zero_shot_query)
                # Concatenate the few-shot examples with the zero-shot prompt.
                # Few-shot examples should provide context for the LLM prior to the query.
                final_prompt = f"{few_shot_examples}\n\n{zero_shot_query}"                 

                # Use no_grad to avoid tracking gradients during generation
                with torch.no_grad():
                    output = generator(final_prompt, **generation_args)
                    output_text = output[0]['generated_text'][len(final_prompt):]

                if debug:
                    print(output[0]['generated_text'])
                    print("-------------")
                    print(output_text)
                
                match = re.match(r'^\s*(?:(?:Answer:\s*)|(?:Reasoning:\s*))?(Yes|No)\b[.]*\s*(.*)', output_text, re.DOTALL)
                
                if match:
                    result = match.group(1)  # Captures 'Yes' or 'No'
                    cot = match.group(2).strip()
                else:
                    result = -1
                    cot = output_text

                output_data = {
                    "project": project,
                    "commit_message": commit_message,
                    "func": func,
                    "target": target,
                    "cwe": cwe,
                    "cve": cve,
                    "cve_desc": cve_desc,
                    "result": result,
                    "cot": cot,
                    "fallback": fallback,
                    # "reasoning": reasoning
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
    
    # model, tokenizer = initiate_model(model_id)
    generator = initiate_model(model_id)
    run_on_data(data_path, prompt_template, generator, output_file)
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run PrimeVul processing with specified parameters.")
    
    parser.add_argument("--data_path", type=str,
                        help="Path to the input JSONL data file.")
    parser.add_argument("--model_id", type=str, default="bigcode/starcoder2-3b", #google/codegemma-2b
                        help="Identifier of the model to be used (e.g., bigcode/starcoder2-3b).")
    parser.add_argument("--prompting", type=str, default="zero_shot",  # few_shot, zero_shot
                        help="Type of prompting to use (default: zero_shot).")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to the output file. If not provided, a default path is generated.")
    
    args = parser.parse_args()
    
    # Set default paths if not provided
    
    if args.data_path is None:
        args.data_path = "/speed-scratch/ra_mdash/PrimeVul_v0.1/primevul_with_4_shot.json"
    if args.output_file is None:
        model_name = args.model_id.split("/")[1]
        args.output_file = f"/speed-scratch/ra_mdash/results/prime_vul/{model_name}_few_shot.jsonl"
    
    main(args.data_path, args.model_id, args.prompting, args.output_file)

