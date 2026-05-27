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
        "Tu tarea es traducir la pregunta del usuario a una consulta SPARQL válida, usando EXCLUSIVAMENTE los prefijos y URIs de la ontología del proyecto: "
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

# — PATRÓN 1: Propiedad directa de un recurso conocido (1 salto) —

# Escribe el código SPARQL que recupera la respuesta a esta solicitud: ¿Quién escribió Los héroes del aire?
```sparql
SELECT DISTINCT ?uri WHERE { <http://novelas-populares.org/novela_1859704832> <http://novelas-populares.org/tieneAutor> ?uri }
```

# Escribe el código SPARQL que recupera la respuesta a esta solicitud: ¿En qué ciudad nació Sven Elvestad?
```sparql
SELECT DISTINCT ?uri WHERE { <http://novelas-populares.org/autor_sven_elvestad> <http://novelas-populares.org/lugarDeNacimiento> ?uri }
```

# Escribe el código SPARQL que recupera la respuesta a esta solicitud: ¿En qué idioma está el ejemplar de Retorn al sol?
```sparql
SELECT DISTINCT ?uri WHERE { <http://novelas-populares.org/novela_1814868437> <http://novelas-populares.org/tieneIdiomaEjemplar> ?uri }
```

# Escribe el código SPARQL que recupera la respuesta a esta solicitud: ¿En qué ciudad se publicó Llamas sobre el Bósforo?
```sparql
SELECT DISTINCT ?uri WHERE { <http://novelas-populares.org/novela_1859682529> <http://novelas-populares.org/publicadaEnCiudad> ?uri }
```

# Escribe el código SPARQL que recupera la respuesta a esta solicitud: ¿Quién fundó la Casa Editorial Maucci?
```sparql
SELECT DISTINCT ?uri WHERE { <http://novelas-populares.org/editorial_casa_editorial_maucci> <http://novelas-populares.org/fundadaPor> ?uri }
```

# Escribe el código SPARQL que recupera la respuesta a esta solicitud: ¿En qué idioma fue escrita originalmente Los conquistadores del polo?
```sparql
SELECT DISTINCT ?uri WHERE { <http://novelas-populares.org/novela_1814866396> <http://novelas-populares.org/tieneIdiomaOriginal> ?uri }
```

# — PATRÓN 2: Todos los recursos de una clase con una propiedad dada (inverso, 1 salto) —

# Escribe el código SPARQL que recupera la respuesta a esta solicitud: ¿Qué novelas escribió Jules Verne?
```sparql
SELECT DISTINCT ?uri WHERE { ?uri <http://novelas-populares.org/tieneAutor> <http://novelas-populares.org/autor_jules_verne> . ?uri a <http://novelas-populares.org/Novela> }
```

# Escribe el código SPARQL que recupera la respuesta a esta solicitud: ¿Qué novelas publicó la Editorial Sopena?
```sparql
SELECT DISTINCT ?uri WHERE { ?uri <http://novelas-populares.org/publicadaPor> <http://novelas-populares.org/editorial_editorial_sopena> . ?uri a <http://novelas-populares.org/Novela> }
```

# Escribe el código SPARQL que recupera la respuesta a esta solicitud: ¿Qué autores cultivaron la novela policíaca?
```sparql
SELECT DISTINCT ?uri WHERE { ?uri <http://novelas-populares.org/tieneGenero> <http://novelas-populares.org/genero_novela_policiaca> . ?uri a <http://novelas-populares.org/Autor> }
```

# Escribe el código SPARQL que recupera la respuesta a esta solicitud: ¿Qué editores nacieron en España?
```sparql
SELECT DISTINCT ?uri WHERE { ?uri <http://novelas-populares.org/paisDeNacimiento> <http://novelas-populares.org/pais_espana> . ?uri a <http://novelas-populares.org/Editor> }
```

# — PATRÓN 3: Propiedad de un recurso intermedio (2 saltos, cadena) —

# Escribe el código SPARQL que recupera la respuesta a esta solicitud: ¿Qué editoriales publicaron novelas de Emilio Salgari?
```sparql
SELECT DISTINCT ?uri WHERE { ?x <http://novelas-populares.org/tieneAutor> <http://novelas-populares.org/autor_emilio_salgari> . ?x <http://novelas-populares.org/publicadaPor> ?uri . ?uri a <http://novelas-populares.org/Editorial> }
```

# Escribe el código SPARQL que recupera la respuesta a esta solicitud: ¿Qué ciudades de publicación tienen las novelas de Gaston Leroux?
```sparql
SELECT DISTINCT ?uri WHERE { ?x <http://novelas-populares.org/tieneAutor> <http://novelas-populares.org/autor_gaston_leroux> . ?x <http://novelas-populares.org/publicadaEnCiudad> ?uri }
```

# Escribe el código SPARQL que recupera la respuesta a esta solicitud: ¿Qué idiomas originales tienen las novelas de la Editorial Calleja?
```sparql
SELECT DISTINCT ?uri WHERE { ?x <http://novelas-populares.org/publicadaPor> <http://novelas-populares.org/editorial_editorial_calleja> . ?x <http://novelas-populares.org/tieneIdiomaOriginal> ?uri }
```

# Escribe el código SPARQL que recupera la respuesta a esta solicitud: ¿Quién fundó la editorial que publicó ¡Sola en el mundo!?
```sparql
SELECT DISTINCT ?uri WHERE { <http://novelas-populares.org/novela_1682489914> <http://novelas-populares.org/publicadaPor> ?e . ?e <http://novelas-populares.org/fundadaPor> ?uri }
```

# — PATRÓN 4: Dos condiciones sobre el mismo recurso (multi-filtro) —

# Escribe el código SPARQL que recupera la respuesta a esta solicitud: ¿Qué novelas publicó la Editorial Sopena con ejemplar en castellano?
```sparql
SELECT DISTINCT ?uri WHERE { ?uri <http://novelas-populares.org/publicadaPor> <http://novelas-populares.org/editorial_editorial_sopena> . ?uri <http://novelas-populares.org/tieneIdiomaEjemplar> <http://novelas-populares.org/idioma_castellano> . ?uri a <http://novelas-populares.org/Novela> }
```

# Escribe el código SPARQL que recupera la respuesta a esta solicitud: ¿Qué autores nacidos en Francia cultivaron la novela de misterio?
```sparql
SELECT DISTINCT ?uri WHERE { ?uri <http://novelas-populares.org/paisDeNacimiento> <http://novelas-populares.org/pais_francia> . ?uri <http://novelas-populares.org/tieneGenero> <http://novelas-populares.org/genero_novela_de_misterio> . ?uri a <http://novelas-populares.org/Autor> }
```

# Escribe el código SPARQL que recupera la respuesta a esta solicitud: ¿Qué novelas son traducciones del francés al castellano?
```sparql
SELECT DISTINCT ?uri WHERE { ?uri <http://novelas-populares.org/tieneIdiomaOriginal> <http://novelas-populares.org/idioma_frances> . ?uri <http://novelas-populares.org/tieneIdiomaEjemplar> <http://novelas-populares.org/idioma_castellano> . ?uri a <http://novelas-populares.org/Novela> }
```

# Escribe el código SPARQL que recupera la respuesta a esta solicitud: ¿Qué autores cultivaron tanto la novela como el periodismo?
```sparql
SELECT DISTINCT ?uri WHERE { ?uri <http://novelas-populares.org/tieneGenero> <http://novelas-populares.org/genero_novela> . ?uri <http://novelas-populares.org/tieneGenero> <http://novelas-populares.org/genero_periodismo> . ?uri a <http://novelas-populares.org/Autor> }
```

# — PATRÓN 5: Cadena de 3 saltos (recurso → intermedio → propiedad final) —

# Escribe el código SPARQL que recupera la respuesta a esta solicitud: ¿Qué novelas de autores franceses se publicaron en Madrid?
```sparql
SELECT DISTINCT ?uri WHERE { ?uri <http://novelas-populares.org/publicadaEnCiudad> <http://novelas-populares.org/ciudad_madrid> . ?uri <http://novelas-populares.org/tieneAutor> ?a . ?a <http://novelas-populares.org/paisDeNacimiento> <http://novelas-populares.org/pais_francia> . ?uri a <http://novelas-populares.org/Novela> }
```

# Escribe el código SPARQL que recupera la respuesta a esta solicitud: ¿Qué autores escribieron novelas publicadas en Madrid con ejemplar en castellano?
```sparql
SELECT DISTINCT ?uri WHERE { ?x <http://novelas-populares.org/publicadaEnCiudad> <http://novelas-populares.org/ciudad_madrid> . ?x <http://novelas-populares.org/tieneIdiomaEjemplar> <http://novelas-populares.org/idioma_castellano> . ?x <http://novelas-populares.org/tieneAutor> ?uri . ?uri a <http://novelas-populares.org/Autor> }
```

# Escribe el código SPARQL que recupera la respuesta a esta solicitud: ¿En qué países se ubican las ciudades de nacimiento de los autores de novelas publicadas en Madrid?
```sparql
SELECT DISTINCT ?uri WHERE { ?nov <http://novelas-populares.org/publicadaEnCiudad> <http://novelas-populares.org/ciudad_madrid> . ?nov <http://novelas-populares.org/tieneAutor> ?a . ?a <http://novelas-populares.org/lugarDeNacimiento> ?c . ?c <http://novelas-populares.org/ubicadaEn> ?uri }
```

# Escribe el código SPARQL que recupera la respuesta a esta solicitud: ¿Qué géneros cultivaron los autores de novelas con idioma original húngaro?
```sparql
SELECT DISTINCT ?uri WHERE { ?x <http://novelas-populares.org/tieneIdiomaOriginal> <http://novelas-populares.org/idioma_hungaro> . ?x <http://novelas-populares.org/tieneAutor> ?a . ?a <http://novelas-populares.org/tieneGenero> ?uri }
```

# — PATRÓN 6: Dos entidades que comparten una propiedad (intersección) —

# Escribe el código SPARQL que recupera la respuesta a esta solicitud: ¿Qué editoriales publicaron novelas tanto de Jules Verne como de Arthur Conan Doyle?
```sparql
SELECT DISTINCT ?uri WHERE { ?x <http://novelas-populares.org/tieneAutor> <http://novelas-populares.org/autor_jules_verne> . ?x <http://novelas-populares.org/publicadaPor> ?uri . ?y <http://novelas-populares.org/tieneAutor> <http://novelas-populares.org/autor_arthur_conan_doyle> . ?y <http://novelas-populares.org/publicadaPor> ?uri . ?uri a <http://novelas-populares.org/Editorial> }
```

# Escribe el código SPARQL que recupera la respuesta a esta solicitud: ¿Qué autores tienen novelas publicadas tanto en Barcelona como en Madrid?
```sparql
SELECT DISTINCT ?uri WHERE { ?x <http://novelas-populares.org/tieneAutor> ?uri . ?x <http://novelas-populares.org/publicadaEnCiudad> <http://novelas-populares.org/ciudad_barcelona> . ?y <http://novelas-populares.org/tieneAutor> ?uri . ?y <http://novelas-populares.org/publicadaEnCiudad> <http://novelas-populares.org/ciudad_madrid> . ?uri a <http://novelas-populares.org/Autor> }
```

# — PATRÓN 7: Novelas que comparten propiedad con otra novela + FILTER —

# Escribe el código SPARQL que recupera la respuesta a esta solicitud: ¿Qué novelas comparten autor con Los abandonados del Galveston?
```sparql
SELECT DISTINCT ?uri WHERE { <http://novelas-populares.org/novela_1859687938> <http://novelas-populares.org/tieneAutor> ?a . ?uri <http://novelas-populares.org/tieneAutor> ?a . FILTER(?uri != <http://novelas-populares.org/novela_1859687938>) }
```

# Escribe el código SPARQL que recupera la respuesta a esta solicitud: ¿Qué novelas comparten tanto autor como editorial con A través del Atlántico en globo?
```sparql
SELECT DISTINCT ?uri WHERE { <http://novelas-populares.org/novela_1859671837> <http://novelas-populares.org/tieneAutor> ?a . <http://novelas-populares.org/novela_1859671837> <http://novelas-populares.org/publicadaPor> ?e . ?uri <http://novelas-populares.org/tieneAutor> ?a . ?uri <http://novelas-populares.org/publicadaPor> ?e . FILTER(?uri != <http://novelas-populares.org/novela_1859671837>) }
```

# — PATRÓN 8: SELECT con múltiples variables —

# Escribe el código SPARQL que recupera la respuesta a esta solicitud: ¿Cuál es la ciudad de publicación y la editorial de Los misterios del Mar Indiano?
```sparql
SELECT DISTINCT ?ciudad ?editorial WHERE { <http://novelas-populares.org/novela_1859688217> <http://novelas-populares.org/publicadaEnCiudad> ?ciudad . <http://novelas-populares.org/novela_1859688217> <http://novelas-populares.org/publicadaPor> ?editorial }
```

# Escribe el código SPARQL que recupera la respuesta a esta solicitud: ¿Cuál es el fundador y la ciudad de publicación de las novelas de La Novela Ilustrada?
```sparql
SELECT DISTINCT ?fundador ?ciudad WHERE { <http://novelas-populares.org/editorial_la_novela_ilustrada> <http://novelas-populares.org/fundadaPor> ?fundador . ?x <http://novelas-populares.org/publicadaPor> <http://novelas-populares.org/editorial_la_novela_ilustrada> . ?x <http://novelas-populares.org/publicadaEnCiudad> ?ciudad }
```

# — PATRÓN 9: Novelas con el mismo idioma original y de ejemplar (variable compartida) —

# Escribe el código SPARQL que recupera la respuesta a esta solicitud: ¿Qué novelas tienen el mismo idioma original y del ejemplar?
```sparql
SELECT DISTINCT ?uri WHERE { ?uri <http://novelas-populares.org/tieneIdiomaOriginal> ?io . ?uri <http://novelas-populares.org/tieneIdiomaEjemplar> ?io . ?uri a <http://novelas-populares.org/Novela> }
```

# — PATRÓN 10: COUNT simple —

# Escribe el código SPARQL que recupera la respuesta a esta solicitud: ¿Cuántas novelas escribió Arthur Conan Doyle en la colección?
```sparql
SELECT (COUNT(DISTINCT ?uri) AS ?count) WHERE { ?uri <http://novelas-populares.org/tieneAutor> <http://novelas-populares.org/autor_arthur_conan_doyle> . ?uri a <http://novelas-populares.org/Novela> }
```

# Escribe el código SPARQL que recupera la respuesta a esta solicitud: ¿Cuántas novelas de la colección se publicaron en Madrid?
```sparql
SELECT (COUNT(DISTINCT ?uri) AS ?count) WHERE { ?uri <http://novelas-populares.org/publicadaEnCiudad> <http://novelas-populares.org/ciudad_madrid> . ?uri a <http://novelas-populares.org/Novela> }
```

# Escribe el código SPARQL que recupera la respuesta a esta solicitud: ¿Cuántos géneros literarios cultivó Jules Verne?
```sparql
SELECT (COUNT(DISTINCT ?uri) AS ?count) WHERE { <http://novelas-populares.org/autor_jules_verne> <http://novelas-populares.org/tieneGenero> ?uri }
```

# Escribe el código SPARQL que recupera la respuesta a esta solicitud: ¿Cuántas novelas son traducciones del francés al castellano?
```sparql
SELECT (COUNT(DISTINCT ?uri) AS ?count) WHERE { ?uri <http://novelas-populares.org/tieneIdiomaOriginal> <http://novelas-populares.org/idioma_frances> . ?uri <http://novelas-populares.org/tieneIdiomaEjemplar> <http://novelas-populares.org/idioma_castellano> . ?uri a <http://novelas-populares.org/Novela> }
```

# Escribe el código SPARQL que recupera la respuesta a esta solicitud: ¿Cuántos autores de novelas publicadas en Madrid nacieron en Francia?
```sparql
SELECT (COUNT(DISTINCT ?uri) AS ?count) WHERE { ?x <http://novelas-populares.org/publicadaEnCiudad> <http://novelas-populares.org/ciudad_madrid> . ?x <http://novelas-populares.org/tieneAutor> ?uri . ?uri <http://novelas-populares.org/paisDeNacimiento> <http://novelas-populares.org/pais_francia> . ?uri a <http://novelas-populares.org/Autor> }
```

# — PATRÓN 11: ASK simple (un triple) —

# Escribe el código SPARQL que recupera la respuesta a esta solicitud: ¿Cultivó Jules Verne el género de ciencia ficción?
```sparql
ASK WHERE { <http://novelas-populares.org/autor_jules_verne> <http://novelas-populares.org/tieneGenero> <http://novelas-populares.org/genero_ciencia_ficcion> }
```

# Escribe el código SPARQL que recupera la respuesta a esta solicitud: ¿Nació Jules Verne en Nantes?
```sparql
ASK WHERE { <http://novelas-populares.org/autor_jules_verne> <http://novelas-populares.org/lugarDeNacimiento> <http://novelas-populares.org/ciudad_nantes> }
```

# Escribe el código SPARQL que recupera la respuesta a esta solicitud: ¿Fundó Ramón Sopena la Editorial Sopena?
```sparql
ASK WHERE { <http://novelas-populares.org/editorial_editorial_sopena> <http://novelas-populares.org/fundadaPor> <http://novelas-populares.org/editor_ramon_sopena> }
```

# Escribe el código SPARQL que recupera la respuesta a esta solicitud: ¿Es Malasia una traducción?
```sparql
ASK WHERE { <http://novelas-populares.org/novela_1859705065> <http://novelas-populares.org/esTraduccion> 'true'^^<http://www.w3.org/2001/XMLSchema#boolean> }
```

# — PATRÓN 12: ASK con múltiples condiciones —

# Escribe el código SPARQL que recupera la respuesta a esta solicitud: ¿Publicó la Casa Editorial Maucci alguna novela en Barcelona?
```sparql
ASK WHERE { ?uri <http://novelas-populares.org/publicadaPor> <http://novelas-populares.org/editorial_casa_editorial_maucci> . ?uri <http://novelas-populares.org/publicadaEnCiudad> <http://novelas-populares.org/ciudad_barcelona> }
```

# Escribe el código SPARQL que recupera la respuesta a esta solicitud: ¿Publicó la Editorial Sopena alguna novela de idioma original inglés?
```sparql
ASK WHERE { ?uri <http://novelas-populares.org/publicadaPor> <http://novelas-populares.org/editorial_editorial_sopena> . ?uri <http://novelas-populares.org/tieneIdiomaOriginal> <http://novelas-populares.org/idioma_ingles> }
```

# Escribe el código SPARQL que recupera la respuesta a esta solicitud: ¿Publicó la Casa Editorial Maucci novelas de algún autor italiano?
```sparql
ASK WHERE { ?uri <http://novelas-populares.org/publicadaPor> <http://novelas-populares.org/editorial_casa_editorial_maucci> . ?uri <http://novelas-populares.org/tieneAutor> ?a . ?a <http://novelas-populares.org/paisDeNacimiento> <http://novelas-populares.org/pais_italia> }
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

        # Siempre incluir ejemplos reales y fragmentos de la ontología, igual que few-shot
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


        # Añadir fragmento relevante de la ontología (clases y propiedades principales)
        try:
            # Si schema_rag tiene full_schema, úsalo
            full_schema = self.schema_rag.full_schema()
            # Limitar el tamaño del fragmento para el prompt
            fragmento_ontologia = '\n'.join(full_schema.split('\n')[:40])  # Primeras 40 líneas
            prompt.append('### Fragmento de la ontología del usuario:\n' + fragmento_ontologia + '\n')
        except Exception as e:
            prompt.append(f"# [DEBUG] No se pudo añadir fragmento de la ontología: {e}\n")

        # Añadir schema generado por process_query (por compatibilidad)
        schema = self.schema_rag.process_query(user_query)
        if isinstance(schema, str) and 'SELECT' in schema:
            from Augur.rag import GraphRAG
            schema = GraphRAG.add_prefixes_to_query(GraphRAG, schema)
        prompt.append(self.SCHEMA.format(schema=schema, max_k=5))

        # Añadir razonamiento paso a paso si corresponde
        if chain_t:
            prompt.append(self.COT_SEPLN)

        prompt.append(f"# TAREA:\n# Escribe el código SPARQL que recupere la respuesta a esta solicitud: {user_query}.\n\n")

        if chain_t:
            prompt.append(self.COT_END)

        # return '\n'.join(prompt)

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

        prompt.append(f"# TAREA:\n# Escribe el código SPARQL que recupere la respuesta a esta solicitud: {user_query}.\n\n")

        if chain_t:
            prompt.append(self.COT_END)

        return '\n'.join(prompt)
    prompt)
    
