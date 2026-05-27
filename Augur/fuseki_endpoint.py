import pandas as pd

from dotenv import load_dotenv
load_dotenv()

from rdflib import Graph, URIRef, Literal
from SPARQLWrapper import SPARQLWrapper, JSON, TURTLE

from tqdm import tqdm
tqdm.pandas(desc="Progress")

sparql = SPARQLWrapper("http://localhost:3030/KG_Novelas_Populares/sparql")
sparql.setTimeout(100)


def send_consult(text):
    sparql.setQuery(text)
    sparql.setReturnFormat(JSON)
    try:
        result = sparql.queryAndConvert()
    except Exception as e:
        print('error')
        result = [f'output error: {e}']
    return str(result) if result else 'none'

def send_consult_json(text):
    sparql.setQuery(text)
    sparql.setReturnFormat(JSON)
    try:
        result = sparql.queryAndConvert()
    except Exception as e:
        print('Error in db output')
        result = dict()
    return result if result else dict()

def send_consult_convert(text):
    sparql.setQuery(text)
    sparql.setReturnFormat(JSON)
    graph = Graph()
    try:
        result = sparql.queryAndConvert()
        
        for result_row in result["results"]["bindings"]:
            s = URIRef(result_row["subject"]["value"])
            p = URIRef(result_row["predicate"]["value"])
            # Check if object is a URI or a literal to handle accordingly
            if result_row["object"]["type"] == "uri":
                o = URIRef(result_row["object"]["value"])
            else:
                o = Literal(result_row["object"]["value"])
            graph.add((s, p, o))
        
        for prefix, namespace in graph.namespaces():
            graph.namespace_manager.remove_namespace(prefix)
        
    except Exception as e:
        print('error')
        result = [f'output error: {e}']
    return graph
