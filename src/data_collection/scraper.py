import os, re, json, time, datetime, requests
from bs4 import BeautifulSoup
from slugify import slugify                     # pip install python-slugify
from typing import List, Tuple
import urllib.parse as up
from urllib.parse import urlparse
# ──────────────────────────────────────────────────────────────
# Common: Whitespace normalization (remove \n\t\t → single space)
# ──────────────────────────────────────────────────────────────
def normalize_ws(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


# ──────────────────────────────────────────────────────────────
# ① Parse Key Data (adapted for .bluebox grid)
# ──────────────────────────────────────────────────────────────
def parse_key_data(soup: BeautifulSoup) -> dict:
    data, costs = {}, {}
    bluebox = soup.find("div", class_="bluebox")
    if not bluebox:
        return data

    for blk in bluebox.select("div[class*=flex__]"):
        label = blk.find("strong")
        if not label:
            continue
        key_txt = normalize_ws(label.get_text()).lower()

        # Extract value (ul or p)
        ul = blk.find("ul")
        if ul:
            items = [normalize_ws(li.get_text(" ", strip=True))
                     for li in ul.select("li")]
            value = ", ".join(items)
        else:
            p = blk.find("p")
            value = normalize_ws(p.get_text()) if p else ""

        # Map fields
        if "type of study" in key_txt:
            data["type_of_study"] = value
        elif "standard duration" in key_txt:
            data["standard_duration_of_studies"] = value
        elif key_txt.startswith("credits"):
            digits = re.findall(r'\d+', value)
            data["credits"] = int(digits[0]) if digits else None
        elif "main locations" in key_txt:
            data["main_locations"] = [x.strip() for x in value.split(",")]
        elif "application period" in key_txt:
            data["application_period"] = value
        elif "admission category" in key_txt:
            data["admission_category"] = value
        elif "start of degree program" in key_txt:
            data["start_of_degree_program"] = value
        elif key_txt.startswith("costs"):
            if ul:
                for li in ul.select("li"):
                    txt = normalize_ws(li.get_text(" ", strip=True)).lower()
                    if "student fee" in txt:
                        costs["student_fees"] = li.get_text(" ", strip=True).split(":", 1)[-1].strip()
                    elif "tuition" in txt:
                        costs["international_tuition"] = li.get_text(" ", strip=True)
        elif "required language proficiency" in key_txt:
            data["required_language_proficiency"] = [x.strip() for x in value.split(",")]

    if costs:
        data["costs"] = costs
    return data


# ──────────────────────────────────────────────────────────────
# ② Generic function: Extract sections with subtitles (flowing text + links)
# ──────────────────────────────────────────────────────────────
def extract_section(soup: BeautifulSoup,
                    h2_keyword: str,
                    mapping: dict[str, str]) -> dict:
    res = {v: {"text": "", "links": []} for v in mapping.values()}
    h2 = soup.find("h2", string=lambda t: t and h2_keyword in t.lower())
    if not h2:
        return res
    container = h2.find_next("div")
    if not container:
        return res

    current_key = None
    for tag in container.descendants:
        # 1) Subtitle switching
        if tag.name in {"h3", "strong"}:
            t = normalize_ws(tag.get_text()).lower()
            for patt, field in mapping.items():
                if patt in t:
                    current_key = field
                    break
        # 2) Text
        elif tag.name == "p" and current_key:
            res[current_key]["text"] += normalize_ws(tag.get_text()) + " "
        # 3) Links
        elif tag.name == "a" and current_key and tag.get("href"):
            url = tag["href"].strip()
            url = f"https://www.tum.de{url}" if url.startswith("/") else url
            res[current_key]["links"].append(
                {"text": normalize_ws(tag.get_text()), "url": url}
            )

    # Clean trailing spaces
    for v in res.values():
        v["text"] = v["text"].strip()
    return res


# ──────────────────────────────────────────────────────────────
# ③ Main function: Single page → Structured JSON
# ──────────────────────────────────────────────────────────────
def scrape_tum_program(url: str, program_name: str) -> dict:
    html = requests.get(url, timeout=15).content
    soup = BeautifulSoup(html, "html.parser")

    data = {
        "program_name": program_name,
        "program_description": "",
        "key_data": {},
        "information_on_degree_program": {},
        "application_and_admission": {}
    }

    # Description
    desc = soup.find("p", class_="lead-text") \
        or soup.select_one("div.in2studyfinder.js p")
    if desc:
        data["program_description"] = normalize_ws(desc.get_text())

    # Key Data
    data["key_data"] = parse_key_data(soup)

    # Sections
    info_map = {
        "program profile": "program_profile",
        "program structure": "program_structure",
        "language of instruction": "language_of_instruction",
        "study organization": "information_on_study_organization",
        "fee": "fees_for_the_program"
    }
    appl_map = {
        "application process": "application_process",
        "documents required for the online application": "documents_required_for_online_application",
        "documents required for enrollment": "documents_required_for_enrollment",
        "additional admission requirements": "additional_admission_requirements",
        "application deadlines": "application_deadlines",
        "admission process": "admission_process"
    }

    data["information_on_degree_program"] = extract_section(
        soup, "information on degree program", info_map)
    data["application_and_admission"] = extract_section(
        soup, "application and admission", appl_map)

    return data


# -- 1) Extract "Name + URL" from list page ----------------------------------
def get_program_list(page_n: int) -> List[Tuple[str, str]]:
    """
    page_n : 1..12   TUM A‑Z list actual page number
    Returns [(program_name, detail_page_full_URL), ...]
    """
    base = "https://www.tum.de/en/studies/degree-programs"
    params = {
        "tx_in2studyfinder_pi1[studyCoursesForPage][currentPage]": str(page_n),
        "type": "1308171055"               # Required; cHash can be omitted
    }
    url = f"{base}?{up.urlencode(params, safe='[]')}"
    print(f"   🔍  Fetching list‑page: {url}")

    html = requests.get(url, timeout=15).content
    soup = BeautifulSoup(html, "html.parser")

    programs = []
    for card in soup.select("article.list-teaser"):
        name_tag  = card.select_one("h3.h4")
        link_tag  = card.select_one("footer.list-teaser__footer a")  # ← Your <a> tag
        if not (name_tag and link_tag):
            continue

        name = name_tag.get_text(strip=True)
        rel  = link_tag["href"].strip()
        if "/detail/" not in rel:
            parts = rel.split("/degree-programs/")
            rel   = f"{parts[0]}/degree-programs/detail/{parts[1]}"

        full = rel if rel.startswith("http") else f"https://www.tum.de{rel}"

        programs.append((name, full))

    print(f"   📊  Found {len(programs)} programs on page {page_n}")
    return programs

# -- 2) Main batch processing --------------------------------------------------

def slug_and_degree(url: str) -> tuple[str, str]:
    """
    Input: Complete detail page URL
    Returns: (slug_without_degree, degree_abbrev)  →  ('aerospace', 'msc')
    """
    path = urlparse(url).path         # /en/studies/…/aerospace-master-of-science-msc
    slug = path.rstrip("/").split("/")[-1]         # aerospace-master-of-science-msc
    parts = slug.split("-")
    deg   = parts[-1].lower()                      # msc / bsc / ma / ba …
    base  = "-".join(parts[:-4])                  # Remove "‑master‑of‑science‑msc"
    if not base:                                  # Some degree abbreviations only have 3 parts
        base = "-".join(parts[:-1])               # Fallback
    return base, deg

# Get the workspace root directory (2 levels up from this script)
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

def scrape_all_pages():
    today_stamp = datetime.date.today().strftime("%Y%m%d")
    out_dir = os.path.join(WORKSPACE_ROOT, "data", "raw")
    os.makedirs(out_dir, exist_ok=True)
    out_jsonl = os.path.join(out_dir, f"tum_programs_{today_stamp}.jsonl")
    
    programs_dir = os.path.join(out_dir, "programs")
    os.makedirs(programs_dir, exist_ok=True)

    with open(out_jsonl, "w", encoding="utf-8") as fout:
        for page in range(1, 13):                       # 0…11 total 12 pages
            print(f"\n🌀  Page {page}/12 …")
            programs = get_program_list(page)
            if not programs:  # If current page has no data, might have reached the last page
                print(f"   ⚠️  Page {page+1} is empty, stopping...")
                break
            for name, url in programs:
                print(f"   ↳  Scraping: {name}")
                try:
                    data = scrape_tum_program(url, name)
                    json_line = json.dumps(data, ensure_ascii=False)
                    fout.write(json_line + "\n")

                    # Optional: single file save
                    base_slug, deg = slug_and_degree(url)
                    fname = f"{base_slug}-{deg}.json"
                    with open(os.path.join(programs_dir, fname), "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)

                    time.sleep(0.5)                  # Gentle crawling
                except Exception as e:
                    print(f"   ❌  {name} — error: {e}")

    print(f"\n✅ All completed! Results written to {out_jsonl} (and single files generated in {programs_dir})")

# ---------------- Entry point ----------------
if __name__ == "__main__":
    scrape_all_pages()
