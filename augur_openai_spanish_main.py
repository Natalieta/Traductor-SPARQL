"""
Script para usar OpenAI con datos en español -ontología y datos de entrenamiento-
Adaptación de augur_openai_icl_test_main.py para usar train_data.csv y ontology.ttl
"""
import re
import gc
import os

import pandas as pd
import Augur.context_templates as ctx

from dotenv import load_dotenv
from openai import OpenAI

from tqdm import tqdm

from Augur import (model, rag)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMB_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"  
TEMP = 0


def capture_code(text):
    code_pattern = r"```(?:sparql)?(.*?)```"
    code = re.findall(code_pattern, text, re.DOTALL)
    return code[0] if code else "None"


def main():
    # Definir ruta base relativa al script
    base_dir = os.path.dirname(__file__)

    # Cargar datos de entrenamiento en español (CSV)
    train_path = os.path.join(base_dir, "datafiles", "train_data.csv")
    df_train = pd.read_csv(train_path)

    # Cargar datos de test (test_data.csv)
    test_path = os.path.join(base_dir, "datafiles", "test_data.csv")
    if os.path.exists(test_path):
        df_test = pd.read_csv(test_path)
        test_file = test_path
    else:
        df_test = pd.read_csv(train_path)
        test_file = train_path

    # Si es el mismo archivo, tomar los últimos 5 como test
    if test_file == train_path:
        df_test = df_test.tail(5)

    # Cargar ontología del usuario
    ontology_files = [
        os.path.join(base_dir, "datafiles", "ontology.ttl"),
    ]

    # Instanciar objetos RAG
    ontology_db = rag.GraphRAG(EMB_MODEL_ID, ontology_files)
    queries_db = rag.SparQLRAG(
        EMB_MODEL_ID,
        df_train["corrected_question"].to_list(),
        df_train["sparql_query"].to_list()
    )

    # Instanciar gestor de prompts
    chat_code_agent = ctx.PromptOpenAI(ontology_db, queries_db)

    model_name = "gpt-4o-mini"  # Modelo más actual y económico
    os.makedirs(model_name, exist_ok=True)
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Realizar inferencia con todas las combinaciones de métodos
    all_combined = []
    combinations = [(bool(i & 4), bool(i & 2), bool(i & 1)) for i in range(8)]
    
    print(f"Usando {len(df_test)} consultas de test")
    print(f"Usando {len(df_train)} ejemplos de entrenamiento")
    
    for cot, few_s, rag_s in tqdm(list(combinations)):
        ordered_list = []
        for query in tqdm(df_test["corrected_question"].to_list()):
            
            conversation = model.conversation_init_dict(
                chat_code_agent,
                query,
                few_shot=few_s,
                cot=cot,
                rag=rag_s
            )

            result = client.chat.completions.create(
                model = model_name,
                messages = conversation,
                temperature = TEMP,
            )
            result = result.choices[0].message.content

            result_dict = {
                'model': model_name,
                'method': f"{'FS_' if few_s else ''}{'CoT_' if cot else ''}{'ont_rag_' if rag_s else ''}".strip('_'),
                'consult': capture_code(result),
                'prompt': conversation[1]['content'],
                'query': query
            }

            ordered_list.append(result_dict)
            all_combined.append(result_dict)

            # Limpiar memoria
            del conversation
            gc.collect()

        # Guardar resultado parcial
        method_str = f"{'FS_' if few_s else ''}{'CoT_' if cot else ''}{'ont_rag_' if rag_s else ''}".strip('_')
        pd.DataFrame(ordered_list).to_csv(
            f"{model_name}/spanish_{method_str}_{model_name}.csv"
        )

    # Guardar resultados combinados
    pd.DataFrame(all_combined).to_csv(f"{model_name}/spanish_combined.csv")
    print(f"Resultados guardados en {model_name}/spanish_combined.csv")


if __name__ == '__main__':
    main()
