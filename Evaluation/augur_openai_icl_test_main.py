import re
import gc
import os

import pandas as pd
import Augur.context_templates as ctx
from Augur.execute_sparql import execute_sparql_query

from dotenv import load_dotenv
from torch.cuda import empty_cache
from openai import OpenAI

from tqdm import tqdm

from Augur import (model, rag)

load_dotenv()

OPENAI_API_KEY =  os.getenv("OPENAI_API_KEY")
EMB_MODEL_ID = "sentence-transformers/all-mpnet-base-v2"
MAX_NEW_TOKENS = 2048
TEMP = 0


def capture_code(text):
    code_pattern = r"```(?:sparql)?\s*(.*?)\s*```"
    code = re.findall(code_pattern, text, re.DOTALL)
    if code:
        return code[0].strip()
    return "None"

# --- Funciones de limpieza y prefijos ---
import re as _re
def clean_markdown_links(text):
    if not isinstance(text, str):
        return text
    # Reemplaza [uri](uri) por uri
    text = _re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', r'\2', text)
    # Elimina corchetes sobrantes en <...>
    text = _re.sub(r'<([^>]+)>', r'\1', text)
    return text

def add_prefixes_if_needed(query):
    prefixes = {'np:': 'PREFIX np: <http://novelas-populares.org/>'}
    lines = query.strip().split('\n')
    # Detecta si ya hay algún prefijo
    has_prefix = any(line.strip().lower().startswith('prefix') for line in lines)
    # Detecta si se usa np: en la consulta
    uses_np = 'np:' in query
    prefix_lines = []
    if uses_np and not any('np:' in line for line in lines if line.strip().lower().startswith('prefix')):
        prefix_lines.append(prefixes['np:'])
    # Si hay que añadir prefijos, los ponemos al inicio
    if prefix_lines:
        return '\n'.join(prefix_lines + [query])
    else:
        return query


def main():
    print("[DEBUG] Cargando datasets...")
    data_dir = os.path.join(os.path.dirname(__file__), "datafiles")
    df_test = pd.read_json(os.path.join(data_dir, "test-data.json"))
    print("[DEBUG] test-data.json cargado, filas:", len(df_test))
    df_train = pd.read_json(os.path.join(data_dir, "train-data.json"))
    print("[DEBUG] train-data.json cargado, filas:", len(df_train))

    # URL del endpoint SPARQL
    endpoint_url = "http://localhost:3030/KG_Novelas_Populares/sparql"
    print("[DEBUG] Inicializando GraphRAG...")
    ontology_db = rag.GraphRAG(EMB_MODEL_ID, endpoint_url)
    print("[DEBUG] Inicializando SparQLRAG...")
    queries_db = rag.SparQLRAG(
        EMB_MODEL_ID,
        df_train["corrected_question"].to_list(),
        df_train["sparql_query"].to_list()
    )

    print("[DEBUG] Inicializando PromptOpenAI...")
    chat_code_agent = ctx.PromptOpenAI(ontology_db, queries_db)

    model_name = "gpt-4o-mini"
    os.makedirs(model_name, exist_ok=True)
    print("[DEBUG] Inicializando cliente OpenAI...")
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    #Perform inference with all method combinations
    all_combined = []
    combinations = [(bool(i & 4), bool(i & 2), bool(i & 1)) for i in range(8)]
    print("[DEBUG] Combinaciones a probar:", combinations)
    for cot, few_s, rag_s in tqdm(list(combinations)):
        print(f"[DEBUG] Combinación: cot={cot}, few_s={few_s}, rag_s={rag_s}")
        ordered_list = []
        for query in tqdm(df_test["corrected_question"].to_list()):
            print(f"[DEBUG] Procesando query: {query}")
            
            print("[DEBUG] Generando conversación...")
            conversation = model.conversation_init_dict(
                chat_code_agent,
                query,
                few_shot=few_s,
                cot=cot,
                rag=rag_s
            )
            print("[DEBUG] Llamando a OpenAI...")
            result = client.chat.completions.create(
                model = model_name,
                messages = conversation,
                temperature = 0,
            )
            print("[DEBUG] Respuesta recibida de OpenAI.")
            result_text = result.choices[0].message.content

            sparql_query = capture_code(result_text)
            # Limpieza y prefijos como en el notebook
            sparql_query = clean_markdown_links(sparql_query)
            sparql_query = add_prefixes_if_needed(sparql_query)
            print("[DEBUG] Consulta SPARQL generada (limpia):\n", sparql_query)
            # Ejecutar la consulta SPARQL y guardar el resultado estructurado
            try:
                sparql_result = execute_sparql_query(endpoint_url, sparql_query)
            except Exception as e:
                sparql_result = {"output error": str(e)}

            result = {
                'model': model_name,
                'method': f"{few_s * 'FS'}{cot * 'CoT'}{rag_s * 'ont_rag'}",
                'consult': sparql_query,
                'prompt': conversation[1]['content'],
                'query': query,
                'output': sparql_result
            }

            ordered_list.append(result)
            all_combined.append(result)
            print("[DEBUG] Resultado guardado.")
            # Clear memory cache
            del conversation
            gc.collect()
            empty_cache()



    # Save to disk combined results
    #pd.DataFrame(all_combined).to_csv(os.path.join(model_name, f"{model_name}_combined.csv"))
    print("[DEBUG] Guardando resultados en disco...")
    results_dir = os.path.join(os.path.dirname(__file__), "Results")
    os.makedirs(results_dir, exist_ok=True)
    pd.DataFrame(all_combined).to_csv(os.path.join(results_dir, 'test_icl_main.csv'), index=False)
    print(f"Resultados guardados en: {os.path.join(results_dir, 'test_icl_main.csv')}")


if __name__ == '__main__':
    main()
