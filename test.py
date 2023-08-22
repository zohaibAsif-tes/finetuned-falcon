import bitsandbytes as bnb

import torch

from peft import (
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

import gradio as gr


# Load Trained Model

PEFT_MODEL = "reach_trained_model"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

config = PeftConfig.from_pretrained(PEFT_MODEL)

model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    return_dict=True,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token

model = PeftModel.from_pretrained(model, PEFT_MODEL)

DEVICE = "cuda:0"

def generate_response(prompt: str) -> str:
    encoding = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            max_new_tokens=50,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response[response.find("\n")+1 : ]
    response = response[ : response.find("\n")]
    return response.strip()
    

# Interface
    
gr.Interface(
    fn=generate_response,
    inputs=["text"],
    outputs=["text"],
    title="Finetuned Falcon-7b",
    description="Trained on Reach Chatbot Data",
    theme="finlaymacklon/boxy_violet",
).launch(server_port=8050, share=True)

