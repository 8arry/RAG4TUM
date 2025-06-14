import os
import re  
import json
import datetime
import requests
from bs4 import BeautifulSoup
from typing import Dict, List

def normalize_ws(text: str) -> str:
    """Normalize whitespace in text"""
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text).strip()

def scrape_tum_faq(url: str) -> Dict:
    """Scrape TUM FAQ page"""
    print(f"  🎯 Scraping TUM FAQ: {url}")
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Basic page info
        page_title = soup.find("title")
        title = normalize_ws(page_title.get_text()) if page_title else "TUM FAQ"
        
        data = {
            "page_title": title,
            "source_url": url,
            "scraped_at": datetime.datetime.now().isoformat(),
            "sections": []
        }
        
        print("    📋 Extracting FAQ sections...")
        
        # Find all FAQ sections
        current_section = None
        current_faqs = []
        
        # Process all elements in order
        for elem in soup.find_all(['h2', 'button', 'h3']):
            # Check if this is a section header (H2)
            if elem.name == 'h2':
                section_title = normalize_ws(elem.get_text())
                
                # Skip non-FAQ sections
                if section_title and 'Settings' not in section_title and 'TUM Partners' not in section_title and 'Schools:' not in section_title and 'Latest news' not in section_title and 'Quicklinks:' not in section_title and "Let's shape the future together." not in section_title:
                    # Save previous section if it has FAQs
                    if current_section and current_faqs:
                        data["sections"].append({
                            "section_id": current_section.lower().replace(" ", "_").replace("&", "and").replace("(", "").replace(")", "").replace("-", "_"),
                            "section_title": current_section,
                            "faqs": current_faqs
                        })
                    
                    # Start new section
                    current_section = section_title
                    current_faqs = []
                    print(f"    🔍 Processing section: {section_title}")
            
            # Check if this is a FAQ question (button or H3 with ?)
            elif current_section and ((elem.name == 'button' and '?' in elem.get_text()) or (elem.name == 'h3' and '?' in elem.get_text())):
                question = normalize_ws(elem.get_text())
                
                # Find the answer
                answer_text = ""
                answer_links = []
                
                if elem.name == 'button':
                    # For buttons, look for accordion content
                    parent_container = elem.find_parent('div')
                    if parent_container:
                        content_div = parent_container.find('div', class_='in2template-accordion__content')
                        if content_div:
                            # Extract text
                            answer_text = normalize_ws(content_div.get_text())
                            
                            # Extract links like scraper.py
                            for link in content_div.find_all('a', href=True):
                                url_link = link['href'].strip()
                                url_link = f"https://www.tum.de{url_link}" if url_link.startswith("/") else url_link
                                link_text = normalize_ws(link.get_text())
                                
                                if link_text and url_link:
                                    answer_links.append({
                                        "text": link_text,
                                        "url": url_link,
                                        "type": "link"
                                    })
                
                elif elem.name == 'h3':
                    # For H3, look in next siblings
                    next_sibling = elem.find_next_sibling()
                    while next_sibling and next_sibling.name not in ['h2', 'h3']:
                        if next_sibling.name in ['p', 'div', 'ul'] and next_sibling.get_text().strip():
                            answer_text += " " + normalize_ws(next_sibling.get_text())
                            
                            # Extract links like scraper.py
                            for link in next_sibling.find_all('a', href=True):
                                url_link = link['href'].strip()
                                url_link = f"https://www.tum.de{url_link}" if url_link.startswith("/") else url_link
                                link_text = normalize_ws(link.get_text())
                                
                                if link_text and url_link:
                                    answer_links.append({
                                        "text": link_text,
                                        "url": url_link,
                                        "type": "link"
                                    })
                            break
                        next_sibling = next_sibling.find_next_sibling()
                    
                    answer_text = answer_text.strip()
                
                # Add FAQ if we have both question and answer
                if question and answer_text:
                    # Check for duplicates
                    if not any(faq['question'] == question for faq in current_faqs):
                        faq_entry = {
                            "question": question,
                            "answer": answer_text
                        }
                        if answer_links:
                            faq_entry["links"] = answer_links
                        
                        current_faqs.append(faq_entry)
        
        # Don't forget the last section
        if current_section and current_faqs:
            data["sections"].append({
                "section_id": current_section.lower().replace(" ", "_").replace("&", "and").replace("(", "").replace(")", "").replace("-", "_"),
                "section_title": current_section,
                "faqs": current_faqs
            })
        
        total_faqs = sum(len(section["faqs"]) for section in data["sections"])
        print(f"    ✅ Successfully extracted {len(data['sections'])} sections with {total_faqs} total FAQs")
        
        return data
        
    except Exception as e:
        print(f"    ❌ Error scraping FAQ: {e}")
        return {
            "page_title": "TUM FAQ - Error",
            "source_url": url,
            "scraped_at": datetime.datetime.now().isoformat(),
            "error": str(e),
            "sections": []
        }

def scrape_faq():
    """Main FAQ scraping function"""
    print("\n🌀 Starting TUM FAQ scraping...")
    
    # FAQ URL
    faq_url = "https://www.tum.de/en/studies/support-and-advice/faq"
    
    # Scrape FAQ data
    faq_data = scrape_tum_faq(faq_url)
    
    # Get output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.abspath(os.path.join(script_dir, "../.."))
    out_dir = os.path.join(workspace_root, "data", "raw")
    os.makedirs(out_dir, exist_ok=True)
    
    # Generate filename with today's date
    today_stamp = datetime.date.today().strftime("%Y%m%d")
    
    # Save as JSON
    json_file = os.path.join(out_dir, f"tum_faq_{today_stamp}.json")
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(faq_data, f, ensure_ascii=False, indent=2)
    
    # Save as JSONL (one section per line)
    jsonl_file = os.path.join(out_dir, f"tum_faq_{today_stamp}.jsonl")
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for section in faq_data["sections"]:
            json_line = json.dumps(section, ensure_ascii=False)
            f.write(json_line + "\n")
    
    total_faqs = sum(len(section["faqs"]) for section in faq_data["sections"])
    
    print(f"\n✅ FAQ scraping completed!")
    print(f"   📊 Extracted {len(faq_data['sections'])} sections with {total_faqs} FAQs")
    print(f"   💾 Saved to: {json_file}")
    print(f"   💾 Saved to: {jsonl_file}")

if __name__ == "__main__":
    scrape_faq() 