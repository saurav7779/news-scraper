import requests
import json
import random
import time
import re
import os
from bs4 import BeautifulSoup
from transformers import pipeline
from googletrans import Translator
import firebase_admin
from firebase_admin import credentials, firestore

# Debug: Print current directory and files
print("Current directory:", os.getcwd())
print("Files in directory:", os.listdir('.'))

# Handle Firebase key from env
firebase_key_content = os.environ.get('FIREBASE_KEY')
if firebase_key_content:
    with open('firebase-key.json', 'w') as f:
        f.write(firebase_key_content)
    print("Firebase key written from env")
else:
    print("Warning: FIREBASE_KEY env not set")

# Initialize Firebase
try:
    cred = credentials.Certificate("firebase-key.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("Firebase initialized successfully")
except Exception as e:
    print(f"Firebase init error: {e}")

# Load Configs
try:
    with open('news_sites.json') as f:
        news_sites = json.load(f)
    print("Loaded news_sites.json")
except Exception as e:
    print(f"Error loading news_sites.json: {e}")

# Initialize Models
try:
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    print("Summarizer initialized")
except Exception as e:
    print(f"Summarizer init error: {e}")
translator = Translator()

# Helper Functions
def get_random_headers():
    return {'User-Agent': random.choice([
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15'
    ])}

def get_proxy():
    return None  # Proxies disabled

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip() if text else ""

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
        'politics': ['election', 'modi', 'bjp', 'congress', 'government', 'parliament', 'policy'],
        'sports': ['cricket', 'football', 'ipl', 'olympics', 'tennis', 'hockey'],
        'entertainment': ['bollywood', 'hollywood', 'movie', 'actor', 'actress', 'film'],
        'technology': ['ai', 'smartphone', 'tech', 'gadget', 'software', 'internet'],
        'business': ['market', 'stock', 'economy', 'trade', 'industry', 'company'],
        'finance': ['bank', 'investment', 'finance', 'money', 'budget', 'loan'],
        'health': ['health', 'covid', 'hospital', 'medicine', 'disease', 'vaccine'],
        'education': ['school', 'college', 'university', 'exam', 'education', 'student'],
        'lifestyle': ['lifestyle', 'fashion', 'beauty', 'wellness', 'home', 'living'],
        'science': ['science', 'research', 'space', 'physics', 'biology', 'discovery'],
        'world': ['international', 'global', 'world', 'foreign', 'diplomacy'],
        'crime': ['crime', 'murder', 'theft', 'arrest', 'police', 'law'],
        'environment': ['climate', 'environment', 'pollution', 'wildlife', 'forest'],
        'travel': ['travel', 'tourism', 'destination', 'vacation', 'adventure'],
        'fashion': ['fashion', 'clothing', 'designer', 'style', 'trend'],
        'festival': ['festival', 'diwali', 'holi', 'christmas', 'eid', 'celebration'],
        'job': ['job', 'employment', 'career', 'recruitment', 'hiring'],
        'food': ['food', 'recipe', 'cuisine', 'cooking', 'restaurant'],
        'culture': ['culture', 'tradition', 'heritage', 'art', 'history'],
        'music': ['music', 'song', 'album', 'concert', 'singer'],
        'religion': ['religion', 'temple', 'mosque', 'church', 'prayer'],
        'agriculture': ['agriculture', 'farming', 'crop', 'harvest', 'farmer'],
        'automobile': ['car', 'bike', 'vehicle', 'automobile', 'auto'],
        'gaming': ['game', 'gaming', 'esports', 'console', 'video game'],
        'trending': ['trending', 'viral', 'popular', 'social media', 'buzz']
    }
    title = title.lower() if title else ""
    content = content.lower() if content else ""
    for cat, keywords in categories.items():
        if any(kw in title or kw in content for kw in keywords):
            return cat
    return 'all'  # Default to 'all' if no specific category matches

# Main Scraping Function
def scrape_and_save():
    for site in news_sites:
        print(f"Scraping site: {site['name']} ({site['url']})")
        try:
            response = requests.get(
                site['url'],
                headers=get_random_headers(),
                timeout=10
            )
            print(f"Response status for {site['name']}: {response.status_code}")
            soup = BeautifulSoup(response.text, 'html.parser')
            article_links = soup.select(site['article_link_selector'])[:5]
            print(f"Found {len(article_links)} article links for {site['name']}")
            
            for link in article_links:
                try:
                    article_url = link['href'] if link['href'].startswith('http') else site['url'].rstrip('/') + '/' + link['href'].lstrip('/')
                    print(f"Fetching article: {article_url}")
                    article_res = requests.get(article_url, headers=get_random_headers(), timeout=10)
                    article_soup = BeautifulSoup(article_res.text, 'html.parser')
                    
                    title_elem = article_soup.select_one(site['title_selector'])
                    if not title_elem:
                        print(f"No title found for {article_url}")
                        continue
                    title = clean_text(title_elem.text)
                    print(f"Title: {title[:50]}...")
                    
                    content_paras = article_soup.select(site['content_selector'])[:3]
                    content = ' '.join([p.text for p in content_paras if p.text])
                    if not content:
                        print(f"No content found for {article_url}")
                        continue
                    
                    summary = summarizer(content, max_length=100, min_length=60, do_sample=False)[0]['summary_text']
                    print(f"Summary generated for {title[:50]}...")
                    
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
                    time.sleep(random.uniform(3, 7))  # Avoid blocking
                    
                except Exception as e:
                    print(f"⚠️ Article Error in {site['name']}: {e}")
                    continue
                    
        except Exception as e:
            print(f"🚨 Site Error ({site['name']}): {e}")
            continue

# Run once
if __name__ == "__main__":
    scrape_and_save()
