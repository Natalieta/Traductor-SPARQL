import os

import torch
import accelerate 
from accelerate import Accelerator

from functools import lru_cache
from threading import Thread
from typing import Iterator
import threading

from transformers import (AutoConfig, 
                        AutoModelForCausalLM, 
                        AutoTokenizer, 
                        TextIteratorStreamer, 
                        BitsAndBytesConfig, 
                        TextStreamer,
                        pipeline,
                        LlamaForCausalLM,
                        GenerationMixin)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_openai(message, chat_history, system_prompt, client, model_name="gpt-4o-mini", max_tokens=1024, temperature=0.1):
    """
    Envía un prompt a la API de OpenAI y devuelve la respuesta completa.
    """
    # Construir el historial de mensajes en formato OpenAI
    messages = [{"role": "system", "content": system_prompt}]
    for user, assistant in chat_history:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": message})
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content

@lru_cache(maxsize=1)
def load_quant_model(model_id, on_loading=None):
    """
    Carga el modelo cuantizado en un hilo aparte para evitar bloquear la UI.
    Si se pasa un callback on_loading, se llamará durante la carga.
    """
    result = {}
    def _load():
        if on_loading:
            on_loading("Cargando modelo...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model_config = AutoConfig.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            config=model_config,
            quantization_config=bnb_config,
            device_map='auto',
        )
        tokenizer = _get_tokenizer_with_system_prompt(model_id)
        result['model'] = model
        result['tokenizer'] = tokenizer
    thread = threading.Thread(target=_load)
    thread.start()
    thread.join()  # Espera a que termine la carga
    return result['model'], result['tokenizer']

def conversational_pipeline(model_id, max_new_tokens=500):
    model, tokenizer = load_quant_model(model_id)

    return pipeline(
        "conversational",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        max_new_tokens=max_new_tokens,
        device_map="auto",
        streamer=TextStreamer(tokenizer)
    )

def conversation_init(prompt, user_query, few_shot = False, cot = False, rag = False, cheating=False):
    text_prompt = prompt.generate_prompt(user_query, few_shot, cot, rag, cheating=cheating)
    return f"{prompt.SYSTEM}\n{text_prompt}"

def conversation_init_dict(prompt, user_query, few_shot = False, cot = False, rag = False, cheating=False):
    try:
        user_content = prompt.generate_prompt(user_query, few_shot, cot, rag, cheating=cheating)
    except Exception as e:
        user_content = f"[ERROR] No se pudo generar el prompt: {str(e)}"
    messages=[
        {"role": "system", "content": prompt.SYSTEM},
        {"role": "user", "content": user_content },
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
