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
    with open('firebase_key.json', 'w') as f:
        f.write(firebase_key_content)
else:
    print("Warning: FIREBASE_KEY env not set, using local file if exists.")

# Initialize Firebase
try:
    cred = credentials.Certificate("firebase_key.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
except Exception as e:
    print(f"🚨 Firebase Initialization Error: {e}")
    exit(1)

# Load Configs
try:
    with open('news_sites.json') as f:
        news_sites = json.load(f)
except Exception as e:
    print(f"🚨 Error loading news_sites.json: {e}")
    exit(1)

# Initialize Models
try:
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    translator = Translator()
except Exception as e:
    print(f"🚨 Model Initialization Error: {e}")
    exit(1)

# Helper Functions
def get_random_headers():
    return {'User-Agent': random.choice([
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15)'
    ])}

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def translate_text(text, lang='hi'):
    try:
        return translator.translate(text, dest=lang).text
    except Exception as e:
        print(f"Translation error for {lang}: {e}")
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
            # Fetch without Proxy
            response = requests.get(
                site['url'],
                headers=get_random_headers(),
                timeout=10
            )
            print(f"Response Status for {site['name']}: {response.status_code}")
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Process Articles (top 5, remove duplicates)
            article_links = list(set([link['href'] for link in soup.select(site['article_link_selector'])]))[:5]
            print(f"Found {len(article_links)} unique articles for {site['name']}")
            for article_url in article_links:
                try:
                    # Ensure full URL
                    article_url = article_url if article_url.startswith('http') else site['url'].rstrip('/') + '/' + article_url.lstrip('/')
                    print(f"Scraping Article: {article_url}")
                    article_res = requests.get(article_url, headers=get_random_headers(), timeout=10)
                    article_soup = BeautifulSoup(article_res.text, 'html.parser')
                    
                    # Extract Data
                    title_elem = article_soup.select_one(site['title_selector'])
                    if not title_elem:
                        print(f"No Title Found for {article_url}")
                        continue
                    title = clean_text(title_elem.text)
                    print(f"Title: {title[:50]}...")
                    
                    content_paras = article_soup.select(site['content_selector'])[:3]
                    content = ' '.join([clean_text(p.text) for p in content_paras])
                    if not content:
                        print(f"No Content Found for {article_url}")
                        continue
                    print(f"Content: {content[:100]}...")
                    
                    # Summarize
                    try:
                        summary = summarizer(content, max_length=100, min_length=60, do_sample=False)[0]['summary_text']
                        print(f"Summary: {summary[:100]}...")
                    except Exception as e:
                        print(f"⚠️ Summarization Error for {article_url}: {e}")
                        continue
                    
                    # Translate
                    hindi_translation = translate_text(summary, 'hi')
                    tamil_translation = translate_text(summary, 'ta')
                    print(f"Hindi Translation: {hindi_translation[:100]}...")
                    print(f"Tamil Translation: {tamil_translation[:100]}...")
                    
                    # Save to Firebase
                    try:
                        doc_ref = db.collection("news").document()
                        doc_ref.set({
                            "title": title,
                            "summary": summary,
                            "hindi_translation": hindi_translation,
                            "tamil_translation": tamil_translation,
                            "image": extract_image(article_soup, site['image_selector']),
                            "location": extract_location(content),
                            "category": detect_category(title, content),
                            "source": site['name'],
                            "timestamp": firestore.SERVER_TIMESTAMP
                        })
                        print(f"✅ Saved: {title[:50]}... from {site['name']}")
                    except Exception as e:
                        print(f"⚠️ Firebase Save Error for {article_url}: {e}")
                        continue
                    
                    time.sleep(random.randint(3, 7))  # Avoid Blocking
                
                except Exception as e:
                    print(f"⚠️ Article Error in {site['name']} ({article_url}): {e}")
                    continue
                    
        except Exception as e:
            print(f"🚨 Site Error ({site['name']}): {e}")

# Run once
if __name__ == "__main__":
    scrape_and_save()
