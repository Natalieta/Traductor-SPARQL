import datetime
import math

from abc import ABC, abstractmethod

from functools import lru_cache
from langchain_community.vectorstores.chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rdflib import Graph, Namespace
from rdflib.namespace import RDFS, RDF
from rdflib.plugins.stores.sparqlstore import SPARQLStore


@lru_cache(maxsize=1)
def load_embedding_model(emb_model):
    return HuggingFaceEmbeddings(model_name=emb_model)


class RAGBase(ABC):
    def __init__(self, emb_model_id):
        self.emb_model = load_embedding_model(emb_model_id)
        self.unique_db_dir = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    @abstractmethod
    def process_query(self):
        ...

    @abstractmethod
    def load_vector_db(self):
        ...

    @abstractmethod
    def raw_rag_output(self):
        ...


class GraphRAG(RAGBase):
    """
    Class intended to manage the vector db for a Knowled Graph
    """    
    def add_prefixes_to_query(self, sparql_query):
        prefix_map = {
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "owl": "http://www.w3.org/2002/07/owl#",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
            "skos": "http://www.w3.org/2004/02/skos/core#",
            "bf": "http://id.loc.gov/ontologies/bibframe/",
            "dc": "http://purl.org/dc/elements/1.1/",
            "dct": "http://purl.org/dc/terms/",
            "schema": "https://schema.org/",
            "np": "http://novelas-populares.org/"
        }
        prefix_str = "".join([f"PREFIX {k}: <{v}>\n" for k, v in prefix_map.items()])
        return prefix_str + sparql_query


    def __init__(self, emb_model_id, endpoint_url=None):
        super().__init__(emb_model_id)
        if endpoint_url is None:
            endpoint_url = "http://localhost:3030/KG_Novelas_Populares/sparql"
        self.graph = Graph(store=SPARQLStore(endpoint_url))
        self.unique_db_dir = 'db_ontology/' + self.unique_db_dir
        # Ya no se usa load_vector_db, porque el grafo está en Fuseki
        # self.rag = self.load_vector_db(file_list)


    # load_vector_db ya no es necesario para Fuseki
    def load_vector_db(self, file_list):
        raise NotImplementedError("No se usa load_vector_db cuando se consulta un endpoint SPARQL.")
    

    def raw_rag_output(self, text, k=10):
        return self.rag.similarity_search(text, k=k)
    

    def _get_connected_nodes_and_prefixes(self, node):
        connected_nodes = set()
        for subj, pred, obj in self.graph:
            if str(subj) in node:
                connected_nodes.add(subj)
                connected_nodes.add(obj)
            # if str(obj) in node:
            #     connected_nodes.add(subj)
        connected_graph = Graph()
        for subj, pred, obj in self.graph:
            if subj in connected_nodes:
                if (pred == RDFS.label and obj.language != 'en' or
                        pred == RDFS.comment and (obj.language and obj.language != 'en')):
                    continue
                connected_graph.add((subj, pred, obj))

        # prefixes_to_unbind = [prefix for prefix, _ in connected_graph.namespace_manager.namespaces()]
        # for prefix in prefixes_to_unbind:
        #     connected_graph.namespace_manager.bind(prefix, None, replace=True)
        # for prefix, namespace in self.graph.namespaces():
        #     connected_graph.bind(prefix, namespace)

        return connected_graph
    
    # def _get_connected_nodes_and_prefixes(self, node):
    #     connected_nodes = set()
    #     for subj, pred, obj in self.graph:
    #         if str(subj) in node:
    #             connected_nodes.add(subj)
    #             #connected_nodes.add(obj)
    #         # if str(obj) in node:
    #         #     connected_nodes.add(subj)
    #     connected_graph = Graph()
    #     for subj, pred, obj in self.graph:
    #         if subj in connected_nodes:
    #             if (pred == RDFS.label and obj.language != 'en' or
    #                     pred == RDFS.comment and (obj.language and obj.language != 'en')):
    #                 continue
    #             connected_graph.add((subj, pred, obj))

    #     # prefixes_to_unbind = [prefix for prefix, _ in connected_graph.namespace_manager.namespaces()]
    #     # for prefix in prefixes_to_unbind:
    #     #     connected_graph.namespace_manager.bind(prefix, None, replace=True)
    #     # for prefix, namespace in self.graph.namespaces():
    #     #     connected_graph.bind(prefix, namespace)

    
    
    def process_query(self, text, max_k=10):
        # Si text es una URI local (ej: np:autor_emilio_salgari), buscar sus tripletas
        if isinstance(text, str):
            uris = [text]
        else:
            uris = text

        results = []
        for uri in uris:
            # Construir consulta SPARQL para obtener todas las tripletas del recurso
            sparql = f'''
            SELECT ?p ?o WHERE {{
                {uri} ?p ?o .
            }} LIMIT {max_k}
            '''
            sparql = self.add_prefixes_to_query(sparql)
            print("[DEBUG] Consulta SPARQL enviada al endpoint:\n", sparql)
            try:
                res = list(self.graph.query(sparql))
            except Exception as e:
                print(f"[ERROR] Fallo al ejecutar la consulta SPARQL: {e}")
                res = []
            results.append({"uri": uri, "triples": [(str(uri), str(p), str(o)) for p, o in res]})
        return results
    

    def full_schema(self):
        full_graph = Graph()
        for subj, pred, obj in self.graph:
            if (pred == RDFS.label and obj.language != 'en' or
                    pred == RDFS.comment and obj.language != 'en'):
                continue
            if ('#Class' in str(obj) or
                '#Property' in str(obj) or
                '#domain' in str(pred) or
                    '#range' in str(pred)):
                full_graph.add((subj, pred, obj))

        for prefix, namespace in self.graph.namespaces():
            full_graph.bind(prefix, namespace)

        full_graph = full_graph.serialize(format='turtle')

        return "@prefix np: <http://novelas-populares.org/> .\n" + full_graph


class SparQLRAG(RAGBase):
    """
    This class is intended to implement the RAG to retrieve
    relevant samples for in-context learning.
    """

    def __init__(self, emb_model_id, queries, consults):
        super().__init__(emb_model_id)
        self.unique_db_dir = 'db_sparql/' + self.unique_db_dir
        self.rag = self.load_vector_db(queries, consults)
        

    def load_vector_db(self, queries, consults):
        consults = [{'consult': element} for element in consults]
        db = Chroma.from_texts(
            queries,
            embedding=self.emb_model,
            metadatas=consults,
            persist_directory=self.unique_db_dir
        )
        return db


    def process_query(self, text, k=8):
        output = self.raw_rag_output(text, k)
        output = [{'question': document.page_content, 'metadata': document.metadata}
                  for document in output]
        return output

    
    def raw_rag_output(self, text, k=8):
        return self.rag.similarity_search(text, k=k)


if __name__ == "__main__":
    # TEST CASE
    import pandas as pd

    import os
    ontology_files = [
        os.path.join(os.path.dirname(__file__), "datafiles", "ontology.ttl")
    ]

    model = "sentence-transformers/all-mpnet-base-v2"

    df_train = pd.read_json(os.path.join(os.path.dirname(__file__), "datafiles", "train-data.json"))

    ontology_db = GraphRAG(model, ontology_files)
    # fewshot_db = SparQLRAG(model,
    #                        df_train['corrected_question'].to_list(),
    #                        df_train['sparql_query'].to_list())

    query = " Who is the stockholder of the road tunnels operated by the Massachusetts Department of Transportation?"
    print(ontology_db.process_query(query, max_k=10))

    #print(fewshot_db.process_query(query, 5))
    print('done')

