# Recolector de metadatos MODS para "novela-popular-espanola"
import os
import re
import time
import csv
import datetime
import requests
from bs4 import BeautifulSoup
from lxml import etree

BASE_URL = "https://digital.iai.spk-berlin.de"
CATALOG_URL = BASE_URL + "/viewer/collections/novela-popular-espanola/-/-/{}/SORT_TITLE/-/"
METS_URL = BASE_URL + "/viewer/sourcefile?id={}"

HEADERS = {"User-Agent": "Mozilla/5.0"}

def get_all_ids(target_count=393, per_page=12, max_pages=200):
    ids = []
    seen = set()
    page = 1
    while page <= max_pages and (target_count is None or len(ids) < target_count):
        url = CATALOG_URL.format(page) + (f"?pageSize={per_page}" if per_page else "")
        print(f"Accediendo página {page}: {url}")
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
        except Exception as e:
            print(f"  Error HTTP página {page}: {e}")
            break

        soup = BeautifulSoup(r.content, "html.parser")
        items = soup.select("a[href*='/viewer/image/']")
        if not items:
            print("  No hay items en esta página -> fin.")
            break

        for a in items:
            href = a.get("href", "") or ""
            # esperar formato /viewer/image/<id>/  -> buscar primer número en la URL
            parts = [p for p in href.strip("/").split("/") if p]
            id_part = None
            for part in parts[::-1]:
                if part.isdigit():
                    id_part = part
                    break
            if id_part and id_part not in seen:
                ids.append(id_part)
                seen.add(id_part)
                if target_count and len(ids) >= target_count:
                    break

        print(f"  Acumulados: {len(ids)} IDs")
        page += 1
        time.sleep(0.6)

    return ids

def safe_findtext(tree, xpath, ns):
    try:
        v = tree.findtext(xpath, namespaces=ns)
        return v.strip() if v and v.strip() else ""
    except Exception:
        return ""

def parse_mets(id_):
    url = METS_URL.format(id_)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print(f"ID {id_}: error HTTP -> {e}")
        return [f"{BASE_URL}/viewer/image/{id_}/", "", "", "", "", "", ""]

    content = resp.content
    if b"<mods" not in content and b"mods:" not in content:
        # no es MODS; devolver URL vacíos para campos
        print(f"ID {id_}: no se detecta MODS en respuesta")
        return [f"{BASE_URL}/viewer/image/{id_}/", "", "", "", "", "", ""]

    try:
        tree = etree.fromstring(content)
    except Exception as e:
        print(f"ID {id_}: XML inválido -> {e}")
        return [f"{BASE_URL}/viewer/image/{id_}/", "", "", "", "", "", ""]

    ns = {"mods": "http://www.loc.gov/mods/v3"}

    # Título y subtítulo (varias rutas)
    title = safe_findtext(tree, ".//mods:titleInfo/mods:title", ns) or safe_findtext(tree, ".//mods:title", ns)
    subtitle = safe_findtext(tree, ".//mods:titleInfo/mods:subTitle", ns) or safe_findtext(tree, ".//mods:subTitle", ns)

    # Autor: preferir given + family; si displayForm existe, intentar parsear o usar directamente
    author = ""
    name_el = tree.find(".//mods:name", namespaces=ns)
    if name_el is not None:
        display = safe_findtext(name_el, "mods:displayForm", ns)
        fam = safe_findtext(name_el, "mods:namePart[@type='family']", ns)
        giv = safe_findtext(name_el, "mods:namePart[@type='given']", ns)
        if giv or fam:
            # pedir primero nombre (given) luego apellido (family)
            author = " ".join(filter(None, [giv, fam])).strip()
        elif display:
            # intentar transformar "Apellido, Nombre" a "Nombre Apellido"
            m = re.match(r'^\s*(?P<last>[^,]+),\s*(?P<first>.+)\s*$', display)
            if m:
                author = f"{m.group('first')} {m.group('last')}".strip()
            else:
                author = display.strip()
    else:
        # fallback: cualquier namePart
        np = safe_findtext(tree, ".//mods:name/mods:namePart", ns)
        author = np

    # Editorial
    publisher = safe_findtext(tree, ".//mods:originInfo/mods:publisher", ns) or safe_findtext(tree, ".//mods:publisher", ns)

    # Ciudad
    city = safe_findtext(tree, ".//mods:originInfo/mods:place/mods:placeTerm[@type='text']", ns) or safe_findtext(tree, ".//mods:place/mods:placeTerm", ns)

    # Fecha (extraer año)
    date_text = safe_findtext(tree, ".//mods:originInfo/mods:dateIssued[@keyDate='yes']", ns) or safe_findtext(tree, ".//mods:originInfo/mods:dateIssued", ns)
    year = ""
    if date_text:
        m = re.search(r'(\d{4})', date_text)
        if m:
            year = m.group(1)
        else:
            year = date_text.strip()

    return [f"{BASE_URL}/viewer/image/{id_}/", author or "", title or "", subtitle or "", publisher or "", year or "", city or ""]

def main():
    # parámetros: target_count=None para recorrer hasta que no haya más páginas
    target_count = 393
    per_page = 12
    ids = get_all_ids(target_count=target_count, per_page=per_page)
    print(f"Total IDs a procesar: {len(ids)}")

    today = datetime.date.today().strftime("%Y-%m-%d")
    out_dir = os.getcwd()
    out_csv = os.path.join(out_dir, f"novelas_populares_{today}.csv")
    print("Escribiendo:", out_csv)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["URL", "Autor (nombre luego apellido)", "Título", "Subtítulo", "Editorial", "Año", "Ciudad"])
        for i, id_ in enumerate(ids, 1):
            row = parse_mets(id_)
            writer.writerow(row)
            if i % 10 == 0 or i == len(ids):
                print(f"[{i}/{len(ids)}] {id_} -> {row[2] if row[2] else '(sin título)'}")
            time.sleep(0.35)

    print("Finalizado. CSV en:", out_csv)

if __name__ == "__main__":
    main()