import requests
from bs4 import BeautifulSoup

url = "https://www.tum.de/en/studies/degree-programs/detail/aerospace-bachelor-of-science-bsc"

response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")

# 抓取描述
try:
    description = soup.select_one('div.in2studyfinder.js p.lead-text').get_text(strip=True)
except:
    description = "[No description]"

# 抓取所有带标题的内容块
sections = soup.select('div.margin-top-m')
section_dict = {}

for sec in sections:
    h2 = sec.select_one('h2.h5')
    content_div = sec.find('div', recursive=False)
    if h2 and content_div:
        title = h2.get_text(strip=True)
        content = content_div.get_text(strip=True)
        section_dict[title] = content

# 输出结果
print("📘 Description:\n", description)
print("\n📄 Sections:")
for title, content in section_dict.items():
    print(f"\n--- {title} ---\n{content}")
