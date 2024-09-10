import os
import json
import torch
from transformers import pipeline
import regex as re
import argparse
import openai
from dotenv import load_dotenv

load_dotenv()
os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'
os.environ["WORLD_SIZE"] = '2'
torch.random.manual_seed(0)

DATASETS_DIR = '../../../datasets'
HF_ACCESS_TOKEN = os.getenv("HF_ACCESS_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai.api_key = OPENAI_API_KEY

generation_args = {
    "max_new_tokens": 500,
    "do_sample": True,
    "temperature": 0.2,
    "top_p": 0.9,
    "num_return_sequences": 1,
    "pad_token_id": 50256
}

def create_pipeline(model_id, **kwargs):
    return pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        token=HF_ACCESS_TOKEN,
        device_map="auto",
        **kwargs
    )

def load_model(model_name):
    model_configs = {
        "llama": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "gemma": "google/gemma-2-9b-it",
        "phi": "microsoft/Phi-3-medium-4k-instruct",
        "qwen": "Qwen/Qwen2-7B-Instruct",
        "gpt": "gpt-4o-mini"
    }
    
    if model_name in model_configs:
        if model_name != "gpt":
            return create_pipeline(model_configs[model_name])
        else:
            return model_configs["gpt"]
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def create_evaluation_prompt(question: str, answer: str, explanations: list[str], translations: dict) -> str:
    prompt = f"""
### Instruction:
You will be given an English question, answer, and explanations for context. Then, you will evaluate Vietnamese translations of the question, answer, and explanations. Evaluate each translation based on accuracy, fluency, and cultural appropriateness, considering the full context provided. Assign a score between 0 and 100 for each translation.

Return only the scores of the translated versions from 0 - 100 for each category (question, answer, and explanations) separated by commas. Do not generate any explanation for your answer.

Here's an example:

### Input
English Question: What is the capital of France?
English Answer: The capital of France is Paris.
English Explanations:
Explanation 1: Paris has been the capital of France since the Middle Ages.
Explanation 2: It is known for its iconic landmarks like the Eiffel Tower.

Translations:
Question translations:
Translation 1: Thủ đô của Pháp là gì?
Translation 2: Kinh đô của nước Pháp là thành phố nào?

Answer translations:
Translation 1: Thủ đô của Pháp là Paris.
Translation 2: Paris là kinh đô của nước Pháp.

Explanation 1 translations:
Translation 1: Paris đã là thủ đô của Pháp từ thời Trung cổ.
Translation 2: Paris là thủ đô của Pháp kể từ thời Trung Cổ.

Explanation 2 translations:
Translation 1: Nó nổi tiếng với các địa danh mang tính biểu tượng như tháp Eiffel.
Translation 2: Thành phố này được biết đến với các công trình mang tính biểu tượng như tháp Eiffel.

### Output:
95,90,98,95,93,91,94,92

Now, please evaluate the following:

### Input
English Question: {question}
English Answer: {answer}
English Explanations:
{' '.join([f'Explanation {i+1}: {exp}' for i, exp in enumerate(explanations)])}

Translations:
Question translations:
""" + '\n'.join([f"Translation {i+1}: {trans}" for i,trans in enumerate(translations['question'])]) + """

Answer translations:
""" + '\n'.join([f"Translation {i+1}: {trans}" for i,trans in enumerate(translations['answer'])]) + """

""" + '\n\n'.join(
        [f"Explanation {i+1} translations:\n" + '\n'.join(
            [f"Translation {j+1}: {trans}" for j, trans in enumerate(expl_trans)]
        ) for i, expl_trans in enumerate(translations['explain'])]) + """
### Output:
"""
    return prompt


def evaluate_translations(model_name, question: str, answer: str, explanations: list[str], translations: dict):
    prompt = create_evaluation_prompt(question, answer, explanations, translations)
    model = load_model(model_name)
    print(f"Prompt: {prompt}")
    
    try:
        if model_name in ["llama", "phi", "qwen"]:
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant that can complete a given instruction."},
                {"role": "user", "content": prompt}
            ]
            response = model(messages, **generation_args)
            generated_text = response[0]['generated_text']
            generated_text = generated_text[-1]['content']
        elif model_name == "gemma":
            messages = [
                {"role": "user", "content": "You are a helpful AI assistant that can complete a given instruction." + prompt}
            ]
            response = model(messages, **generation_args)
            generated_text = response[0]['generated_text']
            generated_text = generated_text[-1]['content']
        else:
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant that can complete a given instruction."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=generation_args["max_new_tokens"],
                temperature=generation_args["temperature"],
                top_p=generation_args["top_p"],
                n=generation_args["num_return_sequences"]
            )
            generated_text = response.choices[0].message.content

        print(f"RESPONSE from {model_name}: ======================================")
        print(generated_text)
        
        scores = [int(score) for score in re.findall(r'\d+', generated_text)]
        return {
            'question_score': scores[:len(translations['question'])],
            'answer_score': scores[len(translations['question']):len(translations['question'])+len(translations['answer'])],
            'explain_score': [scores[len(translations['question'])+len(translations['answer'])+i*len(exp_trans):len(translations['question'])+len(translations['answer'])+(i+1)*len(exp_trans)] for i, exp_trans in enumerate(translations['explain'])]
        }
    except Exception as e:
        print(f"Error evaluating translations: {e}")
        return {
            'question_score': [0] * len(translations['question']),
            'answer_score': [0] * len(translations['answer']),
            'explain_score': [[0] * len(exp_trans) for exp_trans in translations['explain']]
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate translations using a specified model.")
    parser.add_argument("model", choices=["llama", "gemma", "phi", "qwen", "gpt"], help="The model to use for evaluation.")
    args = parser.parse_args()

    # Sample data
    question = "What is this?"
    answer = "This is a shower."
    explanations = [
        "The image shows a shower head and faucet typically found in a bathroom shower.",
        "It's a common household fixture used for personal hygiene and cleaning."
    ]

    translations = {
        'question': [
            "Đây là cái gì?",
            "Vật này là gì?",
            "Đây là vật gì?"
        ],
        'answer': [
            "Đây là một vòi hoa sen.",
            "Đây là một buồng tắm.",
            "Đây là một vòi sen."
        ],
        'explain': [
            [
                "Hình ảnh cho thấy một đầu vòi sen và vòi nước thường thấy trong buồng tắm.",
                "Bức ảnh hiển thị đầu vòi hoa sen và vòi nước thường có trong phòng tắm.",
                "Hình ảnh thể hiện một vòi sen và vòi nước điển hình trong buồng tắm."
            ],
            [
                "Đây là một thiết bị gia dụng phổ biến được sử dụng cho vệ sinh cá nhân và làm sạch.",
                "Nó là một vật dụng thông thường trong nhà dùng để tắm rửa và vệ sinh cơ thể.",
                "Đây là một thiết bị thường gặp trong hộ gia đình, dùng để tắm và làm sạch."
            ]
        ]
    }

    result = evaluate_translations(args.model, question, answer, explanations, translations)
    print(result)