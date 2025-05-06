import requests
from bs4 import BeautifulSoup
import json

# 目标页面
url = "https://www.tum.de/en/studies/degree-programs/detail/aerospace-bachelor-of-science-bsc"
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")

# 初始化结构
program_json = {
    "program_name": "Bachelor Aerospace",
    "program_description": "",
    "key_data": {},
    "information_on_degree_program": {
        "program_profile": "",
        "program_structure": "",
        "language_of_instruction": "",
        "information_on_study_organization": "",
        "fees_for_the_program": ""
    },
    "application_and_admission": {
        "application_process": "",
        "documents_required_for_online_application": "",
        "documents_required_for_enrollment": "",
        "additional_admission_requirements": "",
        "application_deadlines": "",
        "admission_process": ""
    }
}

# 1️⃣ 提取描述
desc = soup.select_one('div.in2studyfinder.js p.lead-text')
if desc:
    program_json["program_description"] = desc.get_text(strip=True)

# 2️⃣ 提取 key data
key_data_block = soup.find("h2", string=lambda x: x and "key data" in x.lower())
if key_data_block:
    ul = key_data_block.find_next("ul")
    if ul:
        for li in ul.select('li'):
            text = li.get_text(strip=True)
            if ":" in text:
                key, value = [s.strip() for s in text.split(":", 1)]
                key_lc = key.lower()
                if "type of study" in key_lc:
                    program_json["key_data"]["type_of_study"] = value
                elif "duration" in key_lc:
                    program_json["key_data"]["standard_duration_of_studies"] = value
                elif "credits" in key_lc:
                    program_json["key_data"]["credits"] = int(value.split()[0])
                elif "location" in key_lc:
                    program_json["key_data"]["main_locations"] = [x.strip() for x in value.split(',')]
                elif "application period" in key_lc:
                    program_json["key_data"]["application_period"] = value
                elif "admission category" in key_lc:
                    program_json["key_data"]["admission_category"] = value
                elif "start" in key_lc:
                    program_json["key_data"]["start_of_degree_program"] = value
                elif "fee" in key_lc and "costs" not in program_json["key_data"]:
                    program_json["key_data"]["costs"] = {"student_fees": value}
                elif "tuition" in key_lc:
                    if "costs" not in program_json["key_data"]:
                        program_json["key_data"]["costs"] = {}
                    program_json["key_data"]["costs"]["international_tuition"] = value
                elif "language" in key_lc:
                    program_json["key_data"]["required_language_proficiency"] = [x.strip() for x in value.split(",")]

# 3️⃣ 提取信息块，按小标题拆分
def extract_section(h2_title, mapping):
    result = {key: "" for key in mapping.values()}
    h2 = soup.find("h2", string=lambda x: x and h2_title in x.lower())
    if not h2:
        return result
    container = h2.find_next("div")
    if not container:
        return result

    current_key = None
    for el in container.descendants:
        if el.name in ["h3", "strong"]:
            text = el.get_text(strip=True).lower()
            for pattern, field in mapping.items():
                if pattern in text:
                    current_key = field
                    break
        elif el.name == "p" and current_key:
            result[current_key] += el.get_text(strip=True) + " "

    # 去掉多余空格
    return {k: v.strip() for k, v in result.items()}

# 4️⃣ 拆分两个主要结构块
info_mapping = {
    "program profile": "program_profile",
    "program structure": "program_structure",
    "language of instruction": "language_of_instruction",
    "study organization": "information_on_study_organization",
    "fee": "fees_for_the_program"
}

application_mapping = {
    "application process": "application_process",
    "documents required for the online application": "documents_required_for_online_application",
    "documents required for enrollment": "documents_required_for_enrollment",
    "additional admission requirements": "additional_admission_requirements",
    "application deadlines": "application_deadlines",
    "admission process": "admission_process"
}

program_json["information_on_degree_program"] = extract_section("information on degree program", info_mapping)
program_json["application_and_admission"] = extract_section("application and admission", application_mapping)

# 5️⃣ 保存 JSON
with open("aerospace_bachelor_structured.json", "w", encoding="utf-8") as f:
    json.dump(program_json, f, ensure_ascii=False, indent=2)

print("✅ 保存成功：aerospace_bachelor_structured.json")
