from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
import argparse, json, re, time, os, gc, torch, copy

from prompts import *

cache_dir = "/speed-scratch/ra_mdash/tmp/huggingface"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

HUGGINGFACE_TOKEN = ""

torch.cuda.set_device(0)
gc.collect()
torch.cuda.empty_cache()

def initiate_model(model_id="facebook/Self-taught-evaluator-llama3.1-70B"):

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
        model_id,
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
        model_id,
        subfolder="dpo_model",
        device_map="auto",                   
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

    # Ensure model config also knows pad token id if needed by generate
    model.config.pad_token_id = tokenizer.pad_token_id
    
    tokenizer.padding_side = "left" 
    
    return model, tokenizer
    



def generate_verdict(model, tokenizer, prompt):
    
    # conversation = copy.copy(SELF_TAUGHT_WITH_SYSTEM_PROMPT)
    # conversation[-1]["content"] = conversation[-1]["content"].format(**example_inputs)

    # Generation Config
    gen_cfg = GenerationConfig(
        max_new_tokens=1024, 
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )    

    conversation = prompt
    
    # --- Inference ---
    # print("Tokenizing input...")
    
    tokenized_input = tokenizer.apply_chat_template(
        conversation,
        return_tensors="pt",
        add_generation_prompt=True # Add prompt for generation
    ).to(model.device) # Move to the device where the model's input embeddings are
    
    with torch.no_grad(): # Use no_grad for inference
        judgement = model.generate(
            tokenized_input,
            generation_config=gen_cfg
        )
    
    # --- Decode and Print ---
    # print("Decoding output...")
    # Decode only the generated part, skipping input tokens and special tokens
    output_tokens = judgement[0, tokenized_input.shape[1]:]
    judgement_text = tokenizer.decode(output_tokens, skip_special_tokens=True)
    
    # print("\n--- Judgement ---")
    # print(judgement_text)
    # print("--- End Judgement ---")
    return judgement_text


def contruct_prompt(response_a, response_b, commit, func, model, tokenizer):

    # eval_plan_prompt_temp = get_think_llm_as_judge_eval_plan_prompt()

    # eval_plan_prompt = [
    #     {
    #         "role": message["role"],
    #         "content": message["content"].format(
    #             commit_message=commit,
    #             func=func,
    #             # response_a="Yes. The function lacks proper bounds checking...",
    #             # response_b="No. The function appears safe because..."
    #         )
    #     }
    #     for message in eval_plan_prompt_temp
    # ]    
    # eval_plan_prompt = eval_plan_prompt_temp.format(
    #                 commit_message=commit,
    #                 func=func
    #             )
    # eval_plan_res = generate_verdict(model, tokenizer, eval_plan_prompt)
    # print("**##eval_plan_res**##")
    # print(eval_plan_res)
    # print("\n\n")
    
    # pattern = r'\[Start of Evaluation Plan\](.*?)\[End of Evaluation Plan\]'
    
    # # Search for the pattern in the prompt
    # match = re.search(pattern, eval_plan_res, re.DOTALL)
    
    # # Check if a match was found and extract the content
    # if match:
    #     eval_plan = match.group(1).strip()
    #     print("Extracted Evaluation Plan:")
    #     print(eval_plan)
    # else:
    #     # print("No evaluation plan found between the specified tags.")
    #     eval_plan = ""

    eval_plan = EVAL_PLAN_FOR_COMPARISON

    exec_plan_prompt_temp = get_think_llm_as_judge_exec_plan_prompt()

    exec_plan_prompt = [
        {
            "role": message["role"],
            "content": message["content"].format(
                commit_message=commit,
                func=func,
                response_a=response_a,
                response_b=response_b,
                eval_plan=eval_plan
            )
        }
        for message in exec_plan_prompt_temp
    ]
    
    # exec_plan_prompt = exec_plan_prompt_temp.format(
    #                 commit_message=commit,
    #                 func=func,
    #                 response_a=response_a,
    #                 response_b=response_b,
    #                 eval_plan=eval_plan
    #             )
    
    return exec_plan_prompt

    
def run_on_data(response_1_path, response_2_path, model_id, output_file, model, tokenizer):

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

    # with open(response_1_path, "r") as f:
    #     response_1_samples = json.load(f)
        
    # with open(response_2_path, "r") as f:
    #     response_2_samples = json.load(f)    
    
    with open(response_1_path, "r") as f1, open(response_2_path, "r") as f2, open(output_file, "a") as output_f:
        
        for line1, line2 in zip(f1, f2):
            # Clear unused GPU memory and collect garbage
            torch.cuda.empty_cache()
            gc.collect()
            
            # if count >= start and count <= end:
            if count < max_test:

                if line1.strip():
                    try:
                        data1 = json.loads(line1)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON: {e}")
                        continue
                    
                if line2.strip():
                    try:
                        data2 = json.loads(line2)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON: {e}")
                        continue
                
                # Extract fields from the JSON line
                project = data1.get("project", "")
                commit = data1.get("commit_message", "")
                func = data1.get("func", "")
                target = data1.get("target", "")
                
                result1 = data1.get("result", "")
                cot1 = data1.get("cot", "")
                
                result2 = data2.get("result", "")
                cot2 = data2.get("cot", "")

                response_a = str(result1) + "\n" + str(cot1)
                response_b = str(result2) + "\n" + str(cot2)

                final_prompt = contruct_prompt(response_a, response_b, commit, func, model, tokenizer)

                verdict_res = generate_verdict(model, tokenizer, final_prompt)

                match = re.search(r'\[\[([A-Z])\]\]', verdict_res)
                if match:
                    verdict = match.group(1)
                else:
                    verdict = None

                if debug:
                    print(final_prompt)
                    print("-------------")
                    print(verdict_res)
                    print("***Final verdict***: ", verdict)


                output_data = {
                    "project": project,
                    "commit_message": commit,
                    "func": func,
                    "target": target,
                    "response_a": response_a,
                    "response_b": response_b,
                    "verdict_full": verdict_res,
                    "verdict": verdict,
                    "result_a": str(result1),
                    "result_b": str(result2)
                    # "result": result,
                    # "cot": cot,
                    # "fallback": fallback,
                    # "reasoning": reasoning
                }

                output_f.write(json.dumps(output_data) + "\n")
                
                # Pause briefly and increment count
                time.sleep(1)
                count += 1

    print(f"Processing complete. Results written to {output_file}")    
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run PrimeVul processing with specified parameters.")
    
    parser.add_argument("--data_path", type=str,
                        help="Path to the input JSONL data file.")
    parser.add_argument("--response_1_path", type=str,
                        help="Path to the input JSONL data file.")
    parser.add_argument("--response_2_path", type=str,
                        help="Path to the input JSONL data file.")
    parser.add_argument("--model_id", type=str, default="facebook/Self-taught-evaluator-llama3.1-70B", #google/codegemma-2b
                        help="Identifier of the model to be used (e.g., bigcode/starcoder2-3b).")
    # parser.add_argument("--prompting", type=str, default="zero_shot",  # few_shot, zero_shot
    #                     help="Type of prompting to use (default: zero_shot).")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to the output file. If not provided, a default path is generated.")
    
    args = parser.parse_args()
    
    # Set default paths if not provided
    
    if args.data_path is None:
        args.data_path = "/speed-scratch/ra_mdash/PrimeVul_v0.1/sample_data.json"
    
    if args.output_file is None:
        model_name = args.model_id.split("/")[1]
        args.output_file = f"/speed-scratch/ra_mdash/results/prime_vul/{model_name}_thinking_judge_zeroshot.jsonl"

    if args.response_1_path is None:
        args.response_1_path = f"/speed-scratch/ra_mdash/results/prime_vul/Llama-3.2-3B-Instruct_zero_shot_CORRECTED.jsonl"

    if args.response_2_path is None:
        args.response_2_path = f"/speed-scratch/ra_mdash/results/prime_vul/Qwen2.5-Coder-3B_zero_shot_CORRECTED.jsonl"

    model, tokenizer = initiate_model(args.model_id)

    run_on_data(args.response_1_path, args.response_2_path, args.model_id, args.output_file, model, tokenizer)
    