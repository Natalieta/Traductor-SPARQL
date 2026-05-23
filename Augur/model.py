import os

import torch
import accelerate 
from accelerate import Accelerator

from functools import lru_cache
from threading import Thread
from typing import Iterator

#from transformers import (AutoConfig, 
                        #  AutoModelForCausalLM, 
                        # AutoTokenizer, 
                        # TextIteratorStreamer, 
                        #  BitsAndBytesConfig, 
                        # TextStreamer,
                        # pipeline,
                        # LlamaForCausalLM,
                        # GenerationMixin)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def conversation_init(prompt, user_query, few_shot = False, cot = False, rag = False, cheating=False):
    text_prompt = prompt.generate_prompt(user_query, few_shot, cot, rag, cheating=cheating)
    return f"{prompt.SYSTEM}\n{text_prompt}"

def conversation_init_dict(prompt, user_query, few_shot = False, cot = False, rag = False, cheating=False):
    messages=[
        {"role": "system", "content": prompt.SYSTEM},
        {"role": "user", "content": prompt.generate_prompt(user_query, few_shot, cot, rag, cheating=cheating) },
    ]
    return messages

def conversational_pipeline_st(model_id, max_new_tokens = 500):
    model, tokenizer = load_quant_model(model_id)

    streamer = TextStreamer(tokenizer)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        max_new_tokens=max_new_tokens,
        device_map="auto",
        return_full_text=False,
        streamer=streamer,
    )
    
    return streamer, pipe