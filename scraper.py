import requests
import json
import random
import time
import re
import os
from bs4 import BeautifulSoup
from transformers import pipeline
from googletrans import Translator
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore

# Handle Firebase key from env (for GitHub Actions)
firebase_key_content = os.environ.get('FIREBASE_KEY')
if firebase_key_content:
    with open('firebase-key.json', 'w') as f:
        f.write(firebase_key_content)
else:
    print("Warning: FIREBASE_KEY env not set, using local file if exists.")

# Initialize Firebase
cred = credentials.Certificate("firebase-key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load Configs
with open('news_sites.json') as f:
    news_sites = json.load(f)
# with open('proxy_list.txt') as f:
#     proxies = [line.strip() for line in f if line.strip()]

# Initialize Models
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
translator = Translator()

# Helper Functions
def get_random_headers():
    return {'User-Agent': random.choice([
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15)'
    ])}

# def get_proxy():
#     return {'http': f'http://{random.choice(proxies)}'} if proxies else None

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def translate_text(text, lang='hi'):
    try:
        return translator.translate(text, dest=lang).text
    except Exception as e:
        print(f"Translation error: {e}")
        return text

def extract_image(soup, selector):
    img = soup.select_one(selector)
    return img['src'] if img and 'src' in img.attrs else None

def extract_location(content):
    states = ['Delhi', 'Maharashtra', 'UP', 'Bihar']
    for state in states:
        if state.lower() in content.lower():
            return state
    return "India"

def detect_category(title, content):
    categories = {
        'sports': ['cricket', 'football', 'ipl'],
        'politics': ['modi', 'election', 'bjp'],
        'technology': ['ai', 'smartphone', 'tech']
    }
    for cat, keywords in categories.items():
        if any(kw in title.lower() or kw in content.lower() for kw in keywords):
            return cat
    return 'general'

# Main Scraping Function
def scrape_and_save():
    for site in news_sites:
        try:
            # Fetch with Proxy Rotation
            response = requests.get(
                site['url'],
                headers=get_random_headers(),
                # proxies=get_proxy(),
                timeout=10
            )
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Process Articles (top 5)
            article_links = soup.select(site['article_link_selector'])[:5]
            for link in article_links:
                try:
                    article_url = link['href'] if link['href'].startswith('http') else site['url'] + link['href']
                    article_res = requests.get(article_url, headers=get_random_headers(), proxies=get_proxy(), timeout=10)
                    article_soup = BeautifulSoup(article_res.text, 'html.parser')
                    
                    # Extract Data
                    title_elem = article_soup.select_one(site['title_selector'])
                    if not title_elem:
                        continue
                    title = clean_text(title_elem.text)
                    
                    content_paras = article_soup.select(site['content_selector'])[:3]
                    content = ' '.join([p.text for p in content_paras])
                    if not content:
                        continue
                    
                    summary = summarizer(content, max_length=100, min_length=60, do_sample=False)[0]['summary_text']
                    
                    # Save to Firebase
                    doc_ref = db.collection("news").document()
                    doc_ref.set({
                        "title": title,
                        "summary": summary,
                        "hindi_translation": translate_text(summary, 'hi'),
                        "tamil_translation": translate_text(summary, 'ta'),
                        "image": extract_image(article_soup, site['image_selector']),
                        "location": extract_location(content),
                        "category": detect_category(title, content),
                        "source": site['name'],
                        "timestamp": firestore.SERVER_TIMESTAMP
                    })
                    print(f"✅ Saved: {title[:50]}... from {site['name']}")
                    time.sleep(random.randint(3, 7))  # Avoid Blocking
                
                except Exception as e:
                    print(f"⚠️ Article Error in {site['name']}: {e}")
                    continue
                    
        except Exception as e:
            print(f"🚨 Site Error ({site['name']}): {e}")

# Run once (Actions will schedule)
if __name__ == "__main__":

    scrape_and_save()
