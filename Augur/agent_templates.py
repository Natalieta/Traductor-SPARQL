import stanza
import spacy_stanza
import spacy
import re
import json
import os

from Augur import fuseki_endpoint   
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()
OPENAI_API_KEY =  os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

from abc import ABC, abstractmethod
from .model import load_quant_model
from transformers import pipeline


class Agent(ABC):
    @abstractmethod
    def __call__(self):
        pass


class StanzaTaggingAgent(Agent):
    def __init__(self, model_id):
        stanza.download("es")
        spacy.require_cpu()
        self.nlp = spacy_stanza.load_pipeline("es", use_gpu=False)
        self.pipe = pipeline("conversational", model=model_id)

    def get_definitions(self, tag_list):
        agg_output = dict()
        for element in tag_list:
            prompt = (
                "Eres un asistente útil, respetuoso y honesto experto en codificación de ontologías y web semántica. SOLO da la respuesta a la tarea.\n"
                "# EJEMPLO: Autor\n Un autor es la persona que crea, escribe o compone una obra intelectual, como una novela. Es quien origina el contenido y posee los derechos sobre su creación.\n"
                f"Escribe un rdfs:comment adecuado para el significado de este identificador de ontología: {element}. Sé conciso."
            )
            response = self.pipe(prompt, max_new_tokens=1000)
            # El pipeline devuelve una lista de dicts con 'generated_text'
            output = response[0]['generated_text'].strip('"') if isinstance(response, list) and 'generated_text' in response[0] else str(response)
            agg_output[element] = output
        return agg_output

    def __call__(self, text):
        doc = self.nlp(text)

        verbs_with_aux = []
        propn_groups = []
        nouns = []
        pron_adv = []
        aux_buffer = []

        for token in doc:
            if token.pos_ in {'PRON', 'ADV'} and 'PronType=Int' in token.morph: 
                pron_adv.append(token.text)
            if token.pos_ == "AUX":
                aux_buffer.append(token.text)
            elif token.pos_ == "VERB":
                if aux_buffer:
                    grouped_verb = " ".join(aux_buffer) + " " + token.text
                    verbs_with_aux.append(grouped_verb)
                    aux_buffer = [] 
                else:
                    verbs_with_aux.append(token.text)
            elif token.pos_ == "PROPN":
                if propn_groups and token.i - 1 == propn_groups[-1][-1].i:
                    propn_groups[-1].append(token)
                else:
                    propn_groups.append([token])
            elif token.pos_ == "NOUN":
                nouns.append(token.text)
            else:
                aux_buffer = []

        # Merge
        propn_groups_merged = [" ".join([token.text for token in group]) for group in propn_groups]

        final_list = verbs_with_aux + propn_groups_merged + nouns
        output = self.get_definitions(final_list)

        return {
            "predicted_ids":output, 
            "predicted_names":propn_groups_merged
        }

class OpenAIStanzaTaggingAgent(StanzaTaggingAgent):
    def __init__(self):
        stanza.download("es")
        spacy.require_cpu()
        self.nlp = spacy_stanza.load_pipeline("es", use_gpu=False)

    
    def get_definitions(self, tag_list):
        agg_output = dict()
        for element in tag_list:
            conversation = [
                {
                    'role': 'system',
                    'content': (
                        "###Eres un asistente útil, respetuoso y honesto, experto en codificación,"
                        " ontologías y web semántica. SOLO da la respuesta a la tarea."
                        "\n# EJEMPLO: Autor\n Un autor es la persona que crea, escribe o compone una obra intelectual, como una novela." 
                        " Es quien origina el contenido y posee los derechos sobre su creación."
                    )
                },
                {
                    'role': 'user',
                    'content': (
                        f"Escribe un rdfs:comment adecuado para el significado de este"
                        f" identificador de ontología: {element}. Sé conciso."
                    )
                }
            ]
            result = client.chat.completions.create(
                    model = "gpt-4o-mini",
                    messages = conversation,
                    temperature = 0,
                )
            output  = result.choices[0].message.content.strip('"')
            agg_output[element] = output
        return agg_output
    

class GenerativeTaggingAgent(Agent):
    instruction = """
###Eres un asistente útil, respetuoso y honesto, experto en codificación, ontologías y web semántica. SOLO realiza la instrucción solicitada, en un único JSON. DEBE incluir el código en un bloque:
```json
code
```

### EXAMPLES

# Extrae posibles clases, propiedades y relaciones. Escribe un JSON con los nombres y descripciones genéricas concisas de ellos: ¿Cuántas novelas publicó la Editorial Sopena?
DA FORMATO a la respuesta EXACTAMENTE en este formato JSON:
```json
{
  "Novela": "Clase que representa una obra literaria de ficción en prosa de la colección digitalizada.",
  "Editorial": "Clase que representa una casa editorial responsable de publicar y distribuir novelas.",
  "publicadaPor": "Propiedad de objeto que relaciona una Novela con la Editorial que la publicó.",
  "Editorial Sopena": "Instancia específica de la clase Editorial, la casa editorial fundada por Ramón Sopena en Barcelona."
}
```

# Extrae posibles clases, propiedades y relaciones. Escribe un JSON con los nombres y descripciones genéricas concisas de ellos: ¿Qué novelas escribió Jules Verne?
```json
{
  "Novela": "Clase que representa una obra literaria de ficción en prosa de la colección digitalizada.",
  "Autor": "Clase que representa a la persona que crea o escribe una novela.",
  "tieneAutor": "Propiedad de objeto que relaciona una Novela con su Autor.",
  "Jules Verne": "Instancia específica de la clase Autor, escritor francés de novelas de aventuras y ciencia ficción."
}
```

# Extrae posibles clases, propiedades y relaciones. Escribe un JSON con los nombres y descripciones genéricas concisas de ellos: ¿En qué ciudad nació Maurice Leblanc?
```json
{
  "Autor": "Clase que representa a la persona que crea o escribe una novela.",
  "Ciudad": "Clase que representa un lugar geográfico de tipo ciudad.",
  "lugarDeNacimiento": "Propiedad de objeto que relaciona una Persona con la Ciudad donde nació.",
  "Maurice Leblanc": "Instancia específica de la clase Autor, escritor francés creador del personaje Arsène Lupin."
}
```

# Extrae posibles clases, propiedades y relaciones. Escribe un JSON con los nombres y descripciones genéricas concisas de ellos: ¿En qué país nació Arthur Conan Doyle?
```json
{
  "Autor": "Clase que representa a la persona que crea o escribe una novela.",
  "Pais": "Clase que representa un lugar geográfico de tipo país.",
  "paisDeNacimiento": "Propiedad de objeto que relaciona una Persona con el País donde nació.",
  "Arthur Conan Doyle": "Instancia específica de la clase Autor, escritor británico creador del personaje Sherlock Holmes."
}
```

# Extrae posibles clases, propiedades y relaciones. Escribe un JSON con los nombres y descripciones genéricas concisas de ellos: ¿En qué ciudad se publicó Llamas sobre el Bósforo?
```json
{
  "Novela": "Clase que representa una obra literaria de ficción en prosa de la colección digitalizada.",
  "Ciudad": "Clase que representa un lugar geográfico de tipo ciudad.",
  "publicadaEnCiudad": "Propiedad de objeto que relaciona una Novela con la Ciudad donde fue publicada.",
  "Llamas sobre el Bósforo": "Instancia específica de la clase Novela perteneciente a la colección digitalizada."
}
```

# Extrae posibles clases, propiedades y relaciones. Escribe un JSON con los nombres y descripciones genéricas concisas de ellos: ¿En qué idioma está el ejemplar de Retorn al sol?
```json
{
  "Novela": "Clase que representa una obra literaria de ficción en prosa de la colección digitalizada.",
  "Idioma": "Clase que representa una lengua en la que está escrito el ejemplar físico de una novela.",
  "tieneIdiomaEjemplar": "Propiedad de objeto que relaciona una Novela con el Idioma en que está escrito el ejemplar de la colección.",
  "Retorn al sol": "Instancia específica de la clase Novela perteneciente a la colección digitalizada."
}
```

# Extrae posibles clases, propiedades y relaciones. Escribe un JSON con los nombres y descripciones genéricas concisas de ellos: ¿En qué idioma fue escrita originalmente Los conquistadores del polo?
```json
{
  "Novela": "Clase que representa una obra literaria de ficción en prosa de la colección digitalizada.",
  "Idioma": "Clase que representa la lengua original en la que fue redactada una novela.",
  "tieneIdiomaOriginal": "Propiedad de objeto que relaciona una Novela con el Idioma en que fue escrita originalmente.",
  "Los conquistadores del polo": "Instancia específica de la clase Novela perteneciente a la colección digitalizada."
}
```

# Extrae posibles clases, propiedades y relaciones. Escribe un JSON con los nombres y descripciones genéricas concisas de ellos: ¿Qué novelas son traducciones del francés al castellano?
```json
{
  "Novela": "Clase que representa una obra literaria de ficción en prosa de la colección digitalizada.",
  "Idioma": "Clase que representa una lengua, tanto el idioma original como el del ejemplar.",
  "tieneIdiomaOriginal": "Propiedad de objeto que relaciona una Novela con el Idioma en que fue escrita originalmente.",
  "tieneIdiomaEjemplar": "Propiedad de objeto que relaciona una Novela con el Idioma en que está el ejemplar de la colección.",
  "esTraduccion": "Propiedad de dato booleana que indica si una Novela es una traducción de otra lengua.",
  "Francés": "Instancia de la clase Idioma que representa la lengua francesa.",
  "Castellano": "Instancia de la clase Idioma que representa la lengua castellana."
}
```

# Extrae posibles clases, propiedades y relaciones. Escribe un JSON con los nombres y descripciones genéricas concisas de ellos: ¿Quién fundó la Casa Editorial Maucci?
```json
{
  "Editorial": "Clase que representa una casa editorial responsable de publicar y distribuir novelas.",
  "Editor": "Clase que representa a la persona que funda o dirige una editorial.",
  "fundadaPor": "Propiedad de objeto que relaciona una Editorial con el Editor o Persona que la fundó.",
  "Casa Editorial Maucci": "Instancia específica de la clase Editorial, fundada por Emanuele Maucci en Barcelona."
}
```

# Extrae posibles clases, propiedades y relaciones. Escribe un JSON con los nombres y descripciones genéricas concisas de ellos: ¿Qué autores cultivaron la ciencia ficción?
```json
{
  "Autor": "Clase que representa a la persona que crea o escribe una novela.",
  "Genero": "Clase que representa un género o subgénero literario cultivado por un autor.",
  "tieneGenero": "Propiedad de objeto que relaciona un Autor con el Género literario que cultiva.",
  "Ciencia ficción": "Instancia específica de la clase Genero que representa la literatura especulativa de ciencia ficción."
}
```

# Extrae posibles clases, propiedades y relaciones. Escribe un JSON con los nombres y descripciones genéricas concisas de ellos: ¿Qué autores nacidos en Francia cultivaron la novela de misterio?
```json
{
  "Autor": "Clase que representa a la persona que crea o escribe una novela.",
  "Pais": "Clase que representa un lugar geográfico de tipo país.",
  "Genero": "Clase que representa un género o subgénero literario cultivado por un autor.",
  "paisDeNacimiento": "Propiedad de objeto que relaciona un Autor con el País donde nació.",
  "tieneGenero": "Propiedad de objeto que relaciona un Autor con el Género literario que cultiva.",
  "Francia": "Instancia específica de la clase Pais que representa el país Francia.",
  "Novela de misterio": "Instancia específica de la clase Genero que representa el género de novela de misterio."
}
```

# Extrae posibles clases, propiedades y relaciones. Escribe un JSON con los nombres y descripciones genéricas concisas de ellos: ¿Qué editoriales publicaron novelas tanto de Jules Verne como de Arthur Conan Doyle?
```json
{
  "Editorial": "Clase que representa una casa editorial responsable de publicar y distribuir novelas.",
  "Novela": "Clase que representa una obra literaria de ficción en prosa de la colección digitalizada.",
  "Autor": "Clase que representa a la persona que crea o escribe una novela.",
  "publicadaPor": "Propiedad de objeto que relaciona una Novela con la Editorial que la publicó.",
  "tieneAutor": "Propiedad de objeto que relaciona una Novela con su Autor.",
  "Jules Verne": "Instancia específica de la clase Autor, escritor francés de novelas de aventuras y ciencia ficción.",
  "Arthur Conan Doyle": "Instancia específica de la clase Autor, escritor británico creador del personaje Sherlock Holmes."
}
```

# Extrae posibles clases, propiedades y relaciones. Escribe un JSON con los nombres y descripciones genéricas concisas de ellos: ¿Qué novelas comparten autor con Los abandonados del Galveston?
```json
{
  "Novela": "Clase que representa una obra literaria de ficción en prosa de la colección digitalizada.",
  "Autor": "Clase que representa a la persona que crea o escribe una novela.",
  "tieneAutor": "Propiedad de objeto que relaciona una Novela con su Autor, usada para encontrar novelas con el mismo autor.",
  "Los abandonados del Galveston": "Instancia específica de la clase Novela de referencia cuyo autor se utiliza para encontrar otras novelas."
}
```

# Extrae posibles clases, propiedades y relaciones. Escribe un JSON con los nombres y descripciones genéricas concisas de ellos: ¿En qué países se ubican las ciudades de publicación de novelas en catalán?
```json
{
  "Novela": "Clase que representa una obra literaria de ficción en prosa de la colección digitalizada.",
  "Ciudad": "Clase que representa un lugar geográfico de tipo ciudad donde se publicó una novela.",
  "Pais": "Clase que representa un lugar geográfico de tipo país en el que se ubica una ciudad.",
  "Idioma": "Clase que representa la lengua del ejemplar de una novela.",
  "tieneIdiomaEjemplar": "Propiedad de objeto que relaciona una Novela con el Idioma de su ejemplar.",
  "publicadaEnCiudad": "Propiedad de objeto que relaciona una Novela con la Ciudad donde fue publicada.",
  "ubicadaEn": "Propiedad de objeto que relaciona una Ciudad con el País en que se encuentra.",
  "Catalán": "Instancia específica de la clase Idioma que representa la lengua catalana."
}
```

# Extrae posibles clases, propiedades y relaciones. Escribe un JSON con los nombres y descripciones genéricas concisas de ellos: ¿Cuántas novelas escribió Emilio Salgari en la colección?
```json
{
  "Novela": "Clase que representa una obra literaria de ficción en prosa de la colección digitalizada.",
  "Autor": "Clase que representa a la persona que crea o escribe una novela.",
  "tieneAutor": "Propiedad de objeto que relaciona una Novela con su Autor.",
  "Emilio Salgari": "Instancia específica de la clase Autor, prolífico escritor italiano de novelas de aventuras."
}
```

# Extrae posibles clases, propiedades y relaciones. Escribe un JSON con los nombres y descripciones genéricas concisas de ellos: ¿Cultivó Jules Verne el género de ciencia ficción?
```json
{
  "Autor": "Clase que representa a la persona que crea o escribe una novela.",
  "Genero": "Clase que representa un género o subgénero literario cultivado por un autor.",
  "tieneGenero": "Propiedad de objeto que relaciona un Autor con el Género literario que cultiva.",
  "Jules Verne": "Instancia específica de la clase Autor, escritor francés de novelas de aventuras y ciencia ficción.",
  "Ciencia ficción": "Instancia específica de la clase Genero que representa la literatura especulativa de ciencia ficción."
}
```

# Extrae posibles clases, propiedades y relaciones. Escribe un JSON con los nombres y descripciones genéricas concisas de ellos: ¿Publicó la Casa Editorial Maucci alguna novela en Barcelona?
```json
{
  "Novela": "Clase que representa una obra literaria de ficción en prosa de la colección digitalizada.",
  "Editorial": "Clase que representa una casa editorial responsable de publicar y distribuir novelas.",
  "Ciudad": "Clase que representa un lugar geográfico de tipo ciudad.",
  "publicadaPor": "Propiedad de objeto que relaciona una Novela con la Editorial que la publicó.",
  "publicadaEnCiudad": "Propiedad de objeto que relaciona una Novela con la Ciudad donde fue publicada.",
  "Casa Editorial Maucci": "Instancia específica de la clase Editorial, fundada por Emanuele Maucci en Barcelona.",
  "Barcelona": "Instancia específica de la clase Ciudad que representa la ciudad española de Barcelona."
}
```


### INSTRUCTION
"""
    cot_instruction = """
###Eres un asistente útil, respetuoso y honesto, experto en codificación, ontologías y web semántica. SOLO realiza la instrucción solicitada, en un único JSON. DEBE incluir el código en un bloque:
```json
code
```
### EXAMPLES

# Extrae posibles clases, propiedades y relaciones. Escribe un JSON con los nombres y descripciones genéricas concisas de ellos: ¿En qué ciudad nació Emilio Salgari?

Pensemos paso a paso para mapear las etiquetas POS a elementos de la ontología; comencemos identificando los roles de sustantivos, verbos, adjetivos y nombres compuestos dentro de una oración.
1. Los sustantivos suelen representar clases o instancias en una ontología. "Emilio Salgari" es una instancia de la clase Autor (subclase de Persona). "ciudad" sugiere la clase Ciudad (subclase de Lugar). Esto nos dice qué entidades están involucradas.
2. Los verbos y adjetivos identifican propiedades. El verbo "nació" indica un evento de nacimiento, lo que sugiere la propiedad de objeto lugarDeNacimiento que conecta una Persona con una Ciudad.
3. "Emilio Salgari" es el sujeto (la instancia de Autor consultada); "ciudad" es el objeto buscado, el valor de retorno de la propiedad lugarDeNacimiento.
4. Considerando el contexto semántico: la pregunta "¿En qué ciudad...?" establece que buscamos un valor geográfico de tipo Ciudad. La tripleta resultante es: (Emilio Salgari, lugarDeNacimiento, ?ciudad).

```json
{
  "Autor": "Clase que representa a la persona que crea o escribe una novela, subclase de Persona.",
  "Ciudad": "Clase que representa un lugar geográfico de tipo ciudad, subclase de Lugar.",
  "lugarDeNacimiento": "Propiedad de objeto que relaciona una Persona con la Ciudad donde nació.",
  "Emilio Salgari": "Instancia específica de la clase Autor, prolífico escritor italiano de novelas de aventuras."
}
```

# Extrae posibles clases, propiedades y relaciones. Escribe un JSON con los nombres y descripciones genéricas concisas de ellos: ¿Qué novelas publicó la Editorial Sopena?

Pensemos paso a paso:
1. Los sustantivos clave son "novelas" y "Editorial Sopena". "novelas" apunta a la clase Novela; "Editorial Sopena" es una instancia de la clase Editorial.
2. El verbo "publicó" indica la propiedad publicadaPor, que conecta una Novela con su Editorial. En SPARQL se invierte: buscamos novelas cuya editorial sea Sopena.
3. "Editorial Sopena" es el sujeto conocido (valor fijo); la variable a recuperar son las instancias de Novela.
4. Contexto: "¿Qué novelas...?" busca un conjunto de instancias de Novela filtradas por la Editorial. Tripleta: (?novela, publicadaPor, Editorial Sopena).

```json
{
  "Novela": "Clase que representa una obra literaria de ficción en prosa de la colección digitalizada.",
  "Editorial": "Clase que representa una casa editorial responsable de publicar y distribuir novelas.",
  "publicadaPor": "Propiedad de objeto que relaciona una Novela con la Editorial que la publicó.",
  "Editorial Sopena": "Instancia específica de la clase Editorial, la casa editorial fundada por Ramón Sopena en Barcelona."
}
```

# Extrae posibles clases, propiedades y relaciones. Escribe un JSON con los nombres y descripciones genéricas concisas de ellos: ¿En qué idioma fue escrita originalmente Los conquistadores del polo?

Pensemos paso a paso:
1. Los sustantivos clave son "idioma" y "Los conquistadores del polo". "idioma" apunta a la clase Idioma; "Los conquistadores del polo" es una instancia de la clase Novela.
2. "originalmente" distingue esta pregunta de una sobre el idioma del ejemplar: el modificador adverbial señala la propiedad tieneIdiomaOriginal, no tieneIdiomaEjemplar.
3. El sujeto conocido es la novela concreta; el valor buscado es el Idioma original.
4. Tripleta: (Los conquistadores del polo, tieneIdiomaOriginal, ?idioma).

```json
{
  "Novela": "Clase que representa una obra literaria de ficción en prosa de la colección digitalizada.",
  "Idioma": "Clase que representa la lengua en la que fue escrita originalmente una novela.",
  "tieneIdiomaOriginal": "Propiedad de objeto que relaciona una Novela con el Idioma en que fue escrita originalmente.",
  "Los conquistadores del polo": "Instancia específica de la clase Novela perteneciente a la colección digitalizada."
}
```

# Extrae posibles clases, propiedades y relaciones. Escribe un JSON con los nombres y descripciones genéricas concisas de ellos: ¿Qué novelas son traducciones del francés al castellano?

Pensemos paso a paso:
1. Los sustantivos clave son "novelas", "francés" y "castellano". "novelas" → clase Novela; "francés" y "castellano" → instancias de la clase Idioma.
2. "traducciones" activa dos propiedades simultáneas: tieneIdiomaOriginal (francés) y tieneIdiomaEjemplar (castellano). También sugiere la propiedad booleana esTraduccion.
3. Los valores conocidos son los dos idiomas; la variable a recuperar son las instancias de Novela que satisfacen ambas condiciones.
4. Tripletas: (?novela, tieneIdiomaOriginal, Francés) y (?novela, tieneIdiomaEjemplar, Castellano).

```json
{
  "Novela": "Clase que representa una obra literaria de ficción en prosa de la colección digitalizada.",
  "Idioma": "Clase que representa una lengua, tanto el idioma original como el del ejemplar.",
  "tieneIdiomaOriginal": "Propiedad de objeto que relaciona una Novela con el Idioma en que fue escrita originalmente.",
  "tieneIdiomaEjemplar": "Propiedad de objeto que relaciona una Novela con el Idioma en que está el ejemplar de la colección.",
  "esTraduccion": "Propiedad de dato booleana que indica si una Novela es una traducción de otra lengua.",
  "Francés": "Instancia de la clase Idioma que representa la lengua francesa.",
  "Castellano": "Instancia de la clase Idioma que representa la lengua castellana."
}
```

# Extrae posibles clases, propiedades y relaciones. Escribe un JSON con los nombres y descripciones genéricas concisas de ellos: ¿Qué autores nacidos en Francia cultivaron la novela de misterio?

Pensemos paso a paso:
1. Los sustantivos clave son "autores", "Francia" y "novela de misterio". "autores" → clase Autor; "Francia" → instancia de Pais; "novela de misterio" → instancia de Genero.
2. "nacidos en" indica la propiedad paisDeNacimiento. "cultivaron" indica la propiedad tieneGenero.
3. Esta pregunta combina dos condiciones sobre la misma variable ?autor: debe tener paisDeNacimiento=Francia Y tieneGenero=novela_de_misterio.
4. Tripletas: (?autor, paisDeNacimiento, Francia) y (?autor, tieneGenero, NovelaDeM isterio).

```json
{
  "Autor": "Clase que representa a la persona que crea o escribe una novela.",
  "Pais": "Clase que representa un lugar geográfico de tipo país.",
  "Genero": "Clase que representa un género o subgénero literario cultivado por un autor.",
  "paisDeNacimiento": "Propiedad de objeto que relaciona un Autor con el País donde nació.",
  "tieneGenero": "Propiedad de objeto que relaciona un Autor con el Género literario que cultiva.",
  "Francia": "Instancia específica de la clase Pais que representa el país Francia.",
  "Novela de misterio": "Instancia específica de la clase Genero que representa el género de novela de misterio."
}
```

# Extrae posibles clases, propiedades y relaciones. Escribe un JSON con los nombres y descripciones genéricas concisas de ellos: ¿En qué países se ubican las ciudades de nacimiento de los autores de novelas publicadas en Madrid?

Pensemos paso a paso:
1. Esta pregunta tiene una cadena de 3 saltos: Novela → Autor → Ciudad → País.
2. Los sustantivos indican las clases: Novela, Autor, Ciudad (de nacimiento), Pais.
3. Las propiedades encadenadas son: publicadaEnCiudad (Novela→Ciudad de publicación, valor fijo Madrid), tieneAutor (Novela→Autor), lugarDeNacimiento (Autor→Ciudad de nacimiento), ubicadaEn (Ciudad→Pais, el valor buscado).
4. El recorrido semántico es: (?novela, publicadaEnCiudad, Madrid) → (?novela, tieneAutor, ?autor) → (?autor, lugarDeNacimiento, ?ciudad) → (?ciudad, ubicadaEn, ?pais).

```json
{
  "Novela": "Clase que representa una obra literaria de ficción en prosa de la colección digitalizada.",
  "Autor": "Clase que representa a la persona que crea o escribe una novela.",
  "Ciudad": "Clase que representa un lugar geográfico de tipo ciudad, tanto de publicación como de nacimiento.",
  "Pais": "Clase que representa un lugar geográfico de tipo país en el que se ubica una ciudad.",
  "publicadaEnCiudad": "Propiedad de objeto que relaciona una Novela con la Ciudad donde fue publicada.",
  "tieneAutor": "Propiedad de objeto que relaciona una Novela con su Autor.",
  "lugarDeNacimiento": "Propiedad de objeto que relaciona una Persona con la Ciudad donde nació.",
  "ubicadaEn": "Propiedad de objeto que relaciona una Ciudad con el País en que se encuentra.",
  "Madrid": "Instancia específica de la clase Ciudad que representa la ciudad española de Madrid."
}
```

# Extrae posibles clases, propiedades y relaciones. Escribe un JSON con los nombres y descripciones genéricas concisas de ellos: ¿Cultivó Jules Verne el género de ciencia ficción?

Pensemos paso a paso:
1. La estructura interrogativa "¿Cultivó X el género Y?" corresponde a una consulta ASK: la respuesta es verdadero o falso.
2. Los sustantivos son "Jules Verne" (instancia de Autor) y "ciencia ficción" (instancia de Genero).
3. "Cultivó" activa la propiedad tieneGenero que conecta un Autor con un Genero.
4. Tripleta ASK: (Jules Verne, tieneGenero, ciencia_ficcion) → resultado booleano.

```json
{
  "Autor": "Clase que representa a la persona que crea o escribe una novela.",
  "Genero": "Clase que representa un género o subgénero literario cultivado por un autor.",
  "tieneGenero": "Propiedad de objeto que relaciona un Autor con el Género literario que cultiva.",
  "Jules Verne": "Instancia específica de la clase Autor, escritor francés de novelas de aventuras y ciencia ficción.",
  "Ciencia ficción": "Instancia específica de la clase Genero que representa la literatura especulativa de ciencia ficción."
}
```

### INSTRUCTION
"""
    
    
    def __init__(self, model_id):
        self.pipe = conversational_pipeline(model_id=model_id)


    def extract_first_json_object(self,text):
        open_braces = 0
        first_object = ""
        started = False

        for char in text:
            if char == "{":
                open_braces += 1
                started = True
            elif char == "}":
                open_braces -= 1
            
            if started:
                first_object += char
            
            if open_braces == 0 and started:
                break

        return first_object
    

    def extract_first_list_object(self,text):
        open_braces = 0
        first_object = ""
        started = False

        for char in text:
            if char == "[":
                open_braces += 1
                started = True
            elif char == "]":
                open_braces -= 1
            
            if started:
                first_object += char
            
            if open_braces == 0 and started:
                break

        return first_object
    

    def capture_json(self, text):
        code_pattern = r"```(?:json)?(.*?)```"
        code = re.findall(code_pattern, text, re.DOTALL)
        return code[0] if code else "None"
    
    
    def __call__(self, text, cot=False):
        instruction_text =  f"\n# Tu tarea es extraer posibles clases, propiedades y relaciones. Escribe un JSON con los nombres y descripciones genéricas concisas de ellos: {text}." 
        prompt = self.instruction + instruction_text
        
        if cot: prompt = self.cot_instruction + instruction_text
        
        prompt = prompt + """
UTILIZA el siguiente formato:
```json
{
    string:string,
    string:string,
 }
```
Pensemos paso a paso para mapear las etiquetas POS a elementos de la ontología; comencemos identificando los roles de sustantivos, verbos, adjetivos y nombres compuestos dentro de una oración. CORREGIR LOS ERRORES DE ORTOGRAFÍA EN EL TEXTO
"""

        convers = Conversation()
        convers.add_message({'role':'user', 'content': prompt})
        dialog = self.pipe(convers, max_new_tokens = 1000, do_sample=False)
        output  = dialog.generated_responses[-1].strip('"')
        #output = self.capture_json(output)
        output = self.extract_first_json_object(output)
        try:
            output = json.loads(output)
        except Exception as e:
            print(e, output)
            output = {'default':'default'}
        
        # convers.add_message({'role':'user', 'content': f'A proper noun is a noun that serves as the name for a specific place, person, or thing (for example, "John Smith" or "Everest"). Your task is to identify the PROPER NOUNS that are shown in the following text: {text}. DO Answer as a python list [\"name1\", \"name2\",...] with double quotations.'})
        # dialog = self.pipe(convers, max_new_tokens = 1000, do_sample=False)
        # output_names  = dialog.generated_responses[-1].strip('"')
        # #output = self.capture_json(output)
        # output_names = self.extract_first_list_object(output_names)
        # try:
        #     output_names = json.loads(output_names)
        # except Exception as e:
        #     print(e, output_names)
        #     output_names = list()

        # print(output_names)
        return {
            "predicted_ids":output, 
            "predicted_names":[]#output_names
        }


class OpenAITaggingAgent(GenerativeTaggingAgent):
    def __init__(self):
        pass

    def __call__(self, text):
        prompt = self.instruction + f"\n# Tu tarea es extraer posibles clases, propiedades y relaciones. Escribe un JSON con los nombres y descripciones genéricas concisas de ellos: {text}."
        prompt = prompt + """
DO use the following format:
```json
{
    string:string,
    string:string,
 }
```
NO utilices campos anidados. NO olvides la ',' después de cada elemento.
"""
        conversation = [{"role": "user", "content": prompt}]
        result = client.chat.completions.create(
                    model = "gpt-4o-mini",
                    messages = conversation,
                    temperature = 0,
                )

        output = self.capture_json(result.choices[0].message.content)
        output = self.extract_first_json_object(output)
        try:
            output = json.loads(output)
        except Exception as e:
            print(e, output)
            output = {'default':'default'}

        # conversation.append({"role": "assistant", "content": result.choices[0].message.content})
        # conversation.append({"role":"user", "content": f'A proper noun is a noun that serves as the name for a specific place, person, or thing (for example, "John Smith" or "Everest"). Your task is to identify the PROPER NOUNS that are shown in the following text: {text}. DO Answer as a python list [\"name1\", \"name2\",...] with double quotations.'})
        
        # dialog = client.chat.completions.create(
        #             model = "gpt-4o-mini",
        #             messages = conversation,
        #             temperature = 0,
        #         )
        
        # output_names  = dialog.choices[0].message.content.strip('"')
        # #output = self.capture_json(output)
        # output_names = self.extract_first_list_object(output_names)
        # try:
        #     output_names = json.loads(output_names)
        # except Exception as e:
        #     print(e, output_names)
        #     output_names = list()

        # print(output_names)
            

        return {
            "predicted_ids":output, 
            "predicted_names":[]#output_names
        }


class OpenAIPreFormatTagging(OpenAITaggingAgent):
    def __init__(self):
        super().__init__(self)
    

    def __call__(self):
        pass

