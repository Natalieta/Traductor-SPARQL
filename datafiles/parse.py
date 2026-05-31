import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import csv
from rdflib import Graph
from rdflib.plugins.sparql import prepareQuery

PREFIXES = """
PREFIX np: <http://novelas-populares.org/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
"""

print("Cargando grafo...")
g = Graph()
g.parse("ontology.ttl", format="turtle")
print(f"Triples cargadas: {len(g)}\n")

with open("test-data.csv", encoding="utf-8") as f:
    queries = list(csv.DictReader(f))

report = []

for q in queries:
    qid = q["_id"]
    cq  = q["corrected_question"]
    sq  = q["sparql_query"].strip()
    sq_with_prefix = PREFIXES + sq

    # 1. Validación sintáctica
    try:
        prepareQuery(sq_with_prefix)
    except Exception as e:
        report.append({"_id": qid, "tipo": "?", "estado": "ERROR_SINTAXIS",
                        "n_resultados": "", "corrected_question": cq,
                        "detalle": str(e)[:120]})
        continue

    # 2. Ejecución
    try:
        res = list(g.query(sq_with_prefix))
    except Exception as e:
        report.append({"_id": qid, "tipo": "?", "estado": "ERROR_EJECUCION",
                        "n_resultados": "", "corrected_question": cq,
                        "detalle": str(e)[:120]})
        continue

    sq_upper = sq.upper()

    # 3. Clasificar resultado
    if sq_upper.startswith("ASK"):
        tipo    = "ASK"
        n       = "TRUE" if bool(res) else "FALSE"
        estado  = "OK" if bool(res) else "ASK_FALSE"
    elif "COUNT" in sq_upper:
        tipo   = "COUNT"
        n      = str(list(res[0])[0]) if res else "0"
        estado = "OK" if n != "0" else "COUNT_CERO"
    else:
        tipo   = "SELECT"
        n      = str(len(res))
        estado = "OK" if len(res) > 0 else "SIN_RESULTADOS"

    report.append({"_id": qid, "tipo": tipo, "estado": estado,
                   "n_resultados": n, "corrected_question": cq, "detalle": ""})

# 4. Escribir CSV de validación
with open("test-data.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["_id","tipo","estado","n_resultados",
                                           "corrected_question","detalle"],
                            quoting=csv.QUOTE_ALL)
    writer.writeheader()
    writer.writerows(report)

# 5. Resumen en consola
total  = len(report)
ok     = sum(1 for r in report if r["estado"] == "OK")
errores = [r for r in report if r["estado"] != "OK"]

print(f"RESULTADO: {ok}/{total} consultas correctas")
if errores:
    print(f"\nProblemas detectados ({len(errores)}):")
    for r in errores:
        print(f"  ID {r['_id']:>3} [{r['estado']}] {r['corrected_question'][:60]}")
else:
    print("Todas las consultas son correctas.")

print("\nInforme guardado en: c:\\Users\\natt\\Desktop\\Triningdata_30.05.26\\test-data.csv")