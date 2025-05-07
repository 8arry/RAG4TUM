"""
TUM degree‑program scraper v5  ·  single‑page → structured JSON
---------------------------------------------------------------
✓ program_description            ✓ key_data（嵌套 costs / language 数组）
✓ information_on_degree_program  ✓ application_and_admission
  └─ 每个小标题 {"text","links"}  ✓ 统一空白清洗
"""

import re, json, requests
from bs4 import BeautifulSoup

# ──────────────────────────────────────────────────────────────
# 共用：空白归一化  (去 \n\t\t → 单空格)
# ──────────────────────────────────────────────────────────────
def normalize_ws(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


# ──────────────────────────────────────────────────────────────
# ① 解析 Key Data（适配 .bluebox 九宫格）
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

        # 抓 value（ul 或 p）
        ul = blk.find("ul")
        if ul:
            items = [normalize_ws(li.get_text(" ", strip=True))
                     for li in ul.select("li")]
            value = ", ".join(items)
        else:
            p = blk.find("p")
            value = normalize_ws(p.get_text()) if p else ""

        # 映射字段
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
# ② 通用函数：抽取带小标题块（流式文本+链接）
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
        # 1) 小标题切换
        if tag.name in {"h3", "strong"}:
            t = normalize_ws(tag.get_text()).lower()
            for patt, field in mapping.items():
                if patt in t:
                    current_key = field
                    break
        # 2) 文本
        elif tag.name == "p" and current_key:
            res[current_key]["text"] += normalize_ws(tag.get_text()) + " "
        # 3) 链接
        elif tag.name == "a" and current_key and tag.get("href"):
            url = tag["href"].strip()
            url = f"https://www.tum.de{url}" if url.startswith("/") else url
            res[current_key]["links"].append(
                {"text": normalize_ws(tag.get_text()), "url": url}
            )

    # 清理末尾空格
    for v in res.values():
        v["text"] = v["text"].strip()
    return res


# ──────────────────────────────────────────────────────────────
# ③ 主函数：单页 → 结构化 JSON
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


# ──────────────────────────────────────────────────────────────
# ④ 单页测试（改 URL 和 NAME 即可）
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    URL  = "https://www.tum.de/en/studies/degree-programs/detail/chemical-biotechnology-bachelor-of-science-bsc"
    NAME = "Bachelor Chemical Biotechnology"

    result = scrape_tum_program(URL, NAME)

    with open("chemical-biotechnology_bachelor_structured.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("✅ 已保存：chemical-biotechnology_bachelor_structured.json")
