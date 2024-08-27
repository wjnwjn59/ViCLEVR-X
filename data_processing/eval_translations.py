import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import time
import regex as re
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
os.environ["WORLD_SIZE"] = '2'
torch.random.manual_seed(0)

DATASETS_DIR = '../../../datasets'
HF_ACCESS_TOKEN = "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"
generation_args = {
    "max_new_tokens": 500,
    "do_sample": True,  # Changed from False to True
    "temperature": 0.7,  # Changed from 0.0 to 0.7
    "top_p": 0.9,
    "num_return_sequences": 1,
    "pad_token_id": 50256  # Adding pad_token_id explicitly
}

def create_pipeline(model_id, model_kwargs=None, **kwargs):
    return pipeline(
        "text-generation",
        model=model_id,
        model_kwargs=model_kwargs or {"torch_dtype": torch.bfloat16},
        token=HF_ACCESS_TOKEN,
        device_map="auto",
        **kwargs
    )

def load_model(model_name):
    if model_name == "llama":
        return create_pipeline("meta-llama/Meta-Llama-3.1-8B-Instruct")
    elif model_name == "gemma":
        return create_pipeline("google/gemma-2-9b-it")
    elif model_name == "phi":
        model_id = "microsoft/Phi-3-medium-4k-instruct"
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto", 
            torch_dtype="auto", 
            trust_remote_code=True, 
            attn_implementation='eager'
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        return create_pipeline(model, tokenizer=tokenizer)
    elif model_name == "qwen":
        model_id = "Qwen/Qwen2-7B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        return model, tokenizer
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def create_translation_evaluation_prompt(original: str, translations: dict[str, str]) -> str:
    prompt = f"""
### Instruction:
Compare the Vietnamese translations below for the given English sentence. Evaluate each translation based on accuracy, fluency, and cultural appropriateness. Assign a score between 0 and 100 for each translation.
You only return scores of the translated versions from 0 - 100. Answer must be in JSON format.
Please do not generate the explanation for your answer.

Here is an example:

### Input
English Sentence: Which of the following factors affects long-run aggregate supply?
Translation from gemini: Những yếu tố nào dưới đây ảnh hưởng đến tổng cung dài hạn?
Translation from gemini: Những yếu tố nào sau đây tác động đến tổng cung dài hạn?
Translation from vinai: Những yếu tố nào trong số này có ảnh hưởng đến tổng cung dài hạn?
### Output: 
{{
"gemini": "90",
"gemini": "80",
"vinai": "92"
}}
Now, please answer this:

### Input:
English Sentence: {original}
"""

    for model, translation in translations.items():
        prompt += f"Translation from {model}: {translation}\n"

    prompt += "### Output:\n"

    return prompt

def evaluate_translations(model_name, original: str, translations: dict[str, str]):
    prompt = create_translation_evaluation_prompt(original, translations)
    model = load_model(model_name)
    print(f"Prompt: {prompt}")
    start_time = time.time()
    
    if model_name == "gemma":
        # Gemma doesn't support system messages, so we'll include everything in the user message
        messages = [
            {"role": "user", "content": "You are a helpful AI assistant that can complete a given instruction.\n\n" + prompt}
        ]
        response = model(messages, **generation_args)
        generated_text = response[0]['generated_text']
    elif model_name in ["llama", "phi"]:
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant that can complete a given instruction."},
            {"role": "user", "content": prompt}
        ]
        response = model(messages, **generation_args)
        generated_text = response[0]['generated_text']
    elif model_name == "qwen":
        model, tokenizer = model
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant that can complete a given instruction."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to("cuda")
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=generation_args["max_new_tokens"],
            do_sample=generation_args["do_sample"],
            temperature=generation_args["temperature"],
            top_p=generation_args["top_p"],
            num_return_sequences=generation_args["num_return_sequences"],
            pad_token_id=generation_args["pad_token_id"],
            attention_mask=model_inputs["attention_mask"],
        )
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    end_time = time.time()
    
    print(f"RESPONSE from {model_name}: ======================================")
    if model_name != "qwen":
        generated_text = generated_text[-1]['content']

    match = re.search(r'(?s){(.+)}', generated_text)
    if match:
        json_str = match.group()
        scores = json.loads(json_str)
    else:
        scores = None
        
    result = {
        "response": generated_text,
        "parsed_scores": scores,
        "time": end_time - start_time
    }
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate translations using a specified model.")
    parser.add_argument("model", choices=["llama", "gemma", "phi", "qwen"], help="The model to use for evaluation.")
    args = parser.parse_args()

    translations = {
        "Model A": "Tôi đang ăn cơm.",
        "Model B": "Tôi đang ăn gạo.",
        "Model C": "Tôi ăn cơm."
    }

    original = "I am eating rice."

    result = evaluate_translations(args.model, original, translations)
    print(json.dumps(result, indent=4))