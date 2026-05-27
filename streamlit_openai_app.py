"""
Streamlit app usando OpenAI (gpt-4o-mini) para traducción NL a SPARQL
Versión con API de OpenAI
"""
import os
import json
import re

import pandas as pd 

import streamlit as st

from dotenv import load_dotenv
from openai import OpenAI

import Augur.context_templates as ctx
from Augur import (model, rag)

load_dotenv()

# Configuración de OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# Modelo de embeddings para RAG
EMB_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_ID = EMB_MODEL_ID  # Definición añadida para evitar NameError
MAX_NEW_TOKENS = 2048
TEMP = 1e-10
MAX_INPUT_TOKEN_LENGTH = 6000

st.set_page_config(layout="wide")
st.title('Interfaz de consultas Novelas Populares')
st.markdown("""
<style>
    div[data-baseweb="textarea"] > div {
        height: 20vh !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource()
def load_conversation_model():
    # Usar datos de entrenamiento en español (CSV)
    df_train = pd.read_csv(os.path.join(os.path.dirname(__file__), "datafiles", "train_data.csv"))
    # Usar la ontología del proyecto (ontology.ttl)
    ontology_db = rag.GraphRAG(EMB_MODEL_ID, [os.path.join(os.path.dirname(__file__), "datafiles", "ontology.ttl")])
    queries_db = rag.SparQLRAG(EMB_MODEL_ID, 
                               df_train["corrected_question"].to_list(), 
                               df_train["sparql_query"].to_list())
    chat_code_agent = ctx.PromptOpenAI(ontology_db, queries_db)
    return chat_code_agent


# Verificar API key
if not client:
    st.error("⚠️ No se encontró OPENAI_API_KEY en el archivo .env")
    st.stop()

# Load model and prompt formatter
code_agent = load_conversation_model()

def capture_code(text): 
    code_pattern = r"```(?:sparql)?(.*?)```" 
    code = re.findall(code_pattern, text, re.DOTALL)
    return code[0] if code else "None"


# Configuración de Fuseki
FUSEKI_URL = os.getenv("FUSEKI_URL", "http://localhost:3030")
FUSEKI_DATASET = os.getenv("FUSEKI_DATASET", "KG_Novelas_Populares")

def send_consult(query):
    """Envía una consulta SPARQL al endpoint de Fuseki"""
    import requests
    endpoint = f"{FUSEKI_URL}/{FUSEKI_DATASET}/sparql"
    print("DEBUG: Endpoint:", endpoint)
    print("DEBUG: Consulta SPARQL enviada:")
    print(query)
    response = requests.get(
        endpoint,
        params={'query': query},
        headers={'Accept': 'application/sparql-results+json'}
    )
    print("DEBUG: Código de estado HTTP:", response.status_code)
    print("DEBUG: Respuesta cruda:", response.text)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"HTTP {response.status_code}: {response.text}"}


with st.sidebar:
    st.subheader('Model parameters')
    model_name = st.selectbox(
        'OpenAI Model',
        ['gpt-4o-mini', 'gpt-4o', 'gpt-3.5-turbo'],
        index=0
    )
    temperature = st.slider('temperature', min_value=0.0, max_value=2.0, value=0.0, step=0.1)
    max_tokens = st.slider('max_tokens', min_value=100, max_value=4000, value=500, step=100)

    st.subheader('In-context Learning')
    cove = st.checkbox('Chain of Verification')
    cot = st.checkbox('Chain of Thought')
    rag_ont = st.checkbox('Retrieval Augmented Generation')
    few_shot = st.checkbox('Few-shot Learning')
    agents = st.checkbox('Multi-Agents and ensemble')

# User-provided prompt

if prompt := st.chat_input():
    with st.container():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

col1, col2 = st.columns([1, 1])

if 'output' not in st.session_state:
    st.session_state.output = ''

if 'first_message' not in st.session_state:
    st.session_state.first_message = True

with col1:
    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "Haz una consulta a la base de datos en lenguaje natural:"}]

    # Display or clear chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    def clear_chat_history():
        st.session_state.messages = [{"role": "assistant", "content": "Haz una consulta a la base de datos en lenguaje natural:"}]
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    # Generate a new response if last message is not from assistant

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
               
                st.session_state.first_message = False

                # Generar prompt usando el sistema de Augur
                conversation = model.conversation_init_dict(
                    code_agent,
                    prompt,
                    few_shot=few_shot,
                    cot=cot,
                    rag=rag_ont
                )

                # Llamar a OpenAI
                try:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=conversation,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    model_response = response.choices[0].message.content
                except Exception as e:
                    model_response = f"Error: {str(e)}"

                st.markdown(model_response)
                
                # Extraer código SPARQL
                code = capture_code(model_response)
                st.session_state.output = code

        message = {"role": "assistant", "content": model_response}
        st.session_state.messages.append(message)

with col2: 
    st.write("CONSULTA SPARQL CAPTURADA")
    code_box = st.empty()
    if st.session_state.output:
        code_box.markdown(f'```sparql\n{st.session_state.output}\n```')
    else:
        st.write("No se ha generado ninguna consulta aún")
    
    if st.button('EJECUTAR CONSULTA'):
        if st.session_state.output:
            with st.spinner("Ejecutando consulta..."):
                result = send_consult(st.session_state.output)
                st.json(result)
        else:
            st.warning("No hay consulta SPARQL para ejecutar")
