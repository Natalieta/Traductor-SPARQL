from Augur import fuseki_endpoint
from rdflib import URIRef, Graph, RDFS, Literal, Namespace
from abc import ABC, abstractmethod

class PromptTemplate(ABC):

    @abstractmethod
    def generate_prompt(self):
        ...


class PromptOpenAI(PromptTemplate):
    SYSTEM = (
        "Eres un asistente experto en SPARQL, ontologías y web semántica. "
        "El usuario SIEMPRE escribe sus preguntas en lenguaje natural en español, sin ningún conocimiento de SPARQL ni de prefijos. "
        "Tu tarea es traducir la pregunta del usuario a una consulta SPARQL válida, usando EXCLUSIVAMENTE los prefijos y URIs de la ontología del usuario: "
        "np:, bf:, schema:, owl:, rdf:, rdfs:, xsd:, dct:, dc:, skos:. NUNCA uses dbo:, dbp:, foaf: ni URIs de DBpedia ni de otras ontologías externas. "
        "No expliques SPARQL ni los prefijos al usuario. Solo responde con el código SPARQL necesario para responder la pregunta, encerrado en un bloque:\n```sparql\ncódigo\n```\n"
    )
    
    SCHEMA = '### Identificadores del Esquema de la Ontología.\n ### Estos identificadores son ejemplos de lo que puedes encontrar en la base de datos:\n{schema}\n\n'
    
    COT = """
### Ejemplos de solicitudes de usuarios y sus consultas SPARQL correspondientes:
# Escribe el código SPARQL que recupera la respuesta a esta solicitud: ¿Quién escribió Los héroes del aire?

Pensemos paso a paso para crear una consulta SPARQL:

Paso 1: Reconocer Entidades Nombradas y Entidades de la Ontología
- Identifica las entidades clave en la solicitud. Las entidades clave son "Novela" y "Los héroes del aire".
- Relaciona estas entidades con la ontología: La ontología usa el prefijo 'np:' para novelas-populares.org (ej: 'np:Novela', 'np:tieneAutor').

Paso 2: Traducir a Forma Intermedia
- Mapea las entidades reconocidas a sus entidades correspondientes de la ontología.
- Mapea "Novela" a la clase 'np:Novela'.
- Mapea "Los héroes del aire" a un recurso como '<http://novelas-populares.org/novela_1859704832>'.

Paso 3: Generar Código SPARQL
- Construye la consulta SPARQL usando la representación intermedia.
- Incluye los prefijos necesarios: 'PREFIX np: <http://novelas-populares.org/>'.
- Formula el statement 'SELECT' para obtener el autor: 'SELECT DISTINCT ?uri'.
- Construye la cláusula 'WHERE' para especificar la relación tieneAutor: 'WHERE { ?x np:tieneAutor ?uri }'.

```sparql

SELECT DISTINCT ?uri WHERE { <http://novelas-populares.org/novela_1859704832> np:tieneAutor ?uri }
```
"""
    COT_SEPLN = """
### Ejemplo de razonamiento paso a paso para una consulta SPARQL usando la ontología del usuario:
# Escribe el código SPARQL que recupera la respuesta a esta solicitud: ¿Cuántas novelas de la colección se publicaron en Madrid?

Pensemos paso a paso para crear la consulta SPARQL:

1. Identificar las entidades: "novelas", "publicadas en Madrid".
2. Relacionar con la ontología: La clase 'np:Novela', la propiedad 'np:publicadaEnCiudad'.
3. Formular la consulta SPARQL usando SOLO los prefijos y URIs de la ontología del usuario.

```sparql
PREFIX np: <http://novelas-populares.org/>

SELECT DISTINCT ?uri WHERE {
  ?uri np:publicadaEnCiudad "Madrid" .
  ?uri a np:Novela
}
```
```
Esta consulta busca todas las novelas (np:Novela) que fueron publicadas en Madrid, usando únicamente la ontología y prefijos del usuario.
"""
    COT_END = "Pensemos paso a paso para crear una respuesta utilizando elementos ontológicos y consultas SPARQL, centrándonos en las relaciones y entidades involucradas."
    
    FEW_SHOT = (
        "### Ejemplos de solicitudes de usuario y sus consultas SPARQL usando solo la ontología del proyecto"
        "basado en problemas similares:\n"
    )
    FEW_SHOT_TEMPLATE = "# Escribe el código SPARQL que recupera la respuesta a esta solicitud: {question}\n{consult}\n"

    FIXED_FS = """
### Algunos ejemplos de solicitudes de usuario y sus consultas SPARQL correspondientes basadas en problemas similares:
# Escribe el código SPARQL que recupera la respuesta a esta solicitud: ¿En qué ciudad nació Sven Elvestad?
```sparql
PREFIX np: <http://novelas-populares.org/>
SELECT DISTINCT ?uri WHERE { <http://novelas-populares.org/autor_sven_elvestad> <http://novelas-populares.org/lugarDeNacimiento> ?uri }
```

# Escribe el código SPARQL que recupera la respuesta a esta solicitud: ¿Qué novelas publicó la Editorial Sopena?
PREFIX np: <http://novelas-populares.org/>
SELECT DISTINCT ?uri WHERE { ?uri <http://novelas-populares.org/publicadaPor> <http://novelas-populares.org/editorial_editorial_sopena> . ?uri a <http://novelas-populares.org/Novela> }
```

# Escribe el código SPARQL que recupera la respuesta a esta solicitud: ¿En qué idioma está el ejemplar de Retorn al sol?
```sparql
PREFIX np: <http://novelas-populares.org/>
SELECT DISTINCT ?uri WHERE { <http://novelas-populares.org/novela_1814868437> <http://novelas-populares.org/tieneIdiomaEjemplar> ?uri }
```

# Escribe el código SPARQL que recupera la respuesta a esta solicitud: ¿Cuántas novelas escribió Arthur Conan Doyle en la colección?
```sparql
PREFIX np: <http://novelas-populares.org/>
SELECT (COUNT(DISTINCT ?uri) AS ?count) WHERE { ?uri <http://novelas-populares.org/tieneAutor> <http://novelas-populares.org/autor_arthur_conan_doyle> . ?uri a <http://novelas-populares.org/Novela> }
```

# Escribe el código SPARQL que recupera la respuesta a esta solicitud: ¿Qué autores argentinos están en la colección?
```sparql
PREFIX np: <http://novelas-populares.org/>
SELECT DISTINCT ?uri WHERE { ?uri <http://novelas-populares.org/paisDeNacimiento> <http://novelas-populares.org/pais_argentina> . ?uri a <http://novelas-populares.org/Autor> }
```

# Escribe el código SPARQL que recupera la respuesta a esta solicitud: ¿Cultivó Jules Verne el género de ciencia ficción?
```sparql
PREFIX np: <http://novelas-populares.org/>
ASK WHERE { <http://novelas-populares.org/autor_jules_verne> <http://novelas-populares.org/tieneGenero> <http://novelas-populares.org/genero_ciencia_ficcion> }

```
"""
    def __init__(self, schema_rag, sparql_rag, agent=None):
       self.schema_rag = schema_rag
       self.sparql_rag = sparql_rag
       self.agent = agent

    def search_resource(self, name_list):
        g = Graph()
        ids = set()
        for resource in name_list:
            query = (
                "SELECT ?s ?o  "
                "WHERE {"
                " ?s <http://www.w3.org/2000/01/rdf-schema#label> ?o ."
                f" ?o bif:contains '\"{resource}\"'@en"
                "} LIMIT 10"
            )
            _output = fuseki_endpoint.send_consult_json(query)
            if _output:
                _output = _output["results"]["bindings"]
                for result in _output:
                    if 'Category:' in result["s"]["value"]:
                        continue
                    resource_uri = URIRef(result["s"]["value"])
                    label = Literal(result["o"]["value"])
                    g.add([resource_uri, RDFS.label, label])
                    ids.add(str(resource_uri))
        # prefixes_to_unbind = [prefix for prefix, _ in g.namespace_manager.namespaces()]
        # for prefix in prefixes_to_unbind:
        #     g.namespace_manager.bind(prefix, None, replace=True)
        # return ' '.join([f'<{idss}> \n' for idss in ids])
        return g.serialize(format='turtle')
    def generate_prompt(
        self,
        user_query,
        few_s=False,
        chain_t=False,
        rag=False,
        cheating=False
    ):
        prompt = []

        if rag:
            prompt.append(
                self.SCHEMA.format(
                    schema=self.schema_rag.process_query(user_query),
                    max_k=5
                )
            )
        elif self.agent:
            _result = self.agent(user_query)
            _result_schema = self.schema_rag.process_query(_result['predicted_ids'], max_k=5)
            _result_schema = _result_schema + self.search_resource(_result['predicted_names'])
            prompt.append(
                self.SCHEMA.format(
                    schema=_result_schema
                )
            )
        elif cheating:
            prompt.append(
                self.SCHEMA.format(
                    schema=cheating,
                )
            )

        if self.agent and cheating:
            prompt.append(cheating)

        if few_s:
            fs_rag_output = self.sparql_rag.process_query(user_query)
            few_shot_concat = [
                self.FEW_SHOT_TEMPLATE.format(
                    question=document['question'],
                    consult="```sparql\n" + document['metadata']['consult'] + "\n```"
                )
                for document in fs_rag_output
            ]
            few_shot_concat = '\n'.join(few_shot_concat)
            prompt.append(self.FEW_SHOT + few_shot_concat)
            # prompt.append(self.FIXED_FS)

        if chain_t:
            prompt.append(self.COT_SEPLN)

        prompt.append(f"# TASK:\n# Write the SparQL code that retrieves the answer to this request: {user_query}.\n\n")

        if chain_t:
            prompt.append(self.COT_END)

        return '\n'.join(prompt)
    