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
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from langdetect import detect
from dateutil.parser import parse

# Handle Firebase key from env (for GitHub Actions)
firebase_key_content = os.environ.get('FIREBASE_KEY')
firebase_key_file = 'firebase_key.json'
if firebase_key_content:
    with open(firebase_key_file, 'w') as f:
        f.write(firebase_key_content)
else:
    print("Warning: FIREBASE_KEY env not set, using local file if exists.")
    if os.path.exists('firebase_key.json'):
        firebase_key_content = 'firebase_key.json'
    else:
        print("🚨 No Firebase key found!")
        exit(1)

# Initialize Firebase
try:
    cred = credentials.Certificate(firebase_key_file)
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
summarizer = None
try:
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
except Exception as e:
    print(f"⚠️ Model Initialization Error: {e}. Proceeding without summarization.")
translator = Translator()

# Helper Functions
def get_random_headers():
    return {'User-Agent': random.choice([
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36'
    ])}

def get_page_content(url):
    try:
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        driver = webdriver.Chrome(options=options)
        driver.get(url)
        time.sleep(3)  # Wait for JS to load
        html = driver.page_source
        driver.quit()
        return html
    except Exception as e:
        print(f"⚠️ Selenium Error for {url}: {e}")
        return None

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip() if text else ''

def detect_language(text):
    try:
        return detect(text)
    except:
        return 'en'

def translate_text(text, lang='hi'):
    try:
        return translator.translate(text, dest=lang).text
    except Exception as e:
        print(f"Translation error for {lang}: {e}")
        return text

def extract_image(soup, selector):
    img = soup.select_one(selector)
    if img and 'src' in img.attrs:
        return img['src']
    for alt_selector in ['img.featured-image', 'img.main-img', 'img.article-image', 'img']:
        img = soup.select_one(alt_selector)
        if img and 'src' in img.attrs:
            return img['src']
    return None

def extract_date(soup, selector):
    date_elem = soup.select_one(selector)
    if date_elem and date_elem.text:
        try:
            return parse(clean_text(date_elem.text)).isoformat()
        except:
            pass
    return datetime.now().isoformat()

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
            # Fetch main page with Selenium
            html = get_page_content(site['url'])
            if not html:
                print(f"🚨 Failed to fetch {site['name']}")
                continue
            soup = BeautifulSoup(html, 'html.parser')
            
            # Get article links (including pagination/archive)
            article_links = set()
            main_links = [link['href'] for link in soup.select(site['article_link_selector'])]
            article_links.update(main_links)
            article_links = list(article_links)[:5]  # Limit to 5 articles
            print(f"Found {len(article_links)} unique articles for {site['name']}")
            
            for article_url in article_links:
                try:
                    # Ensure full URL
                    article_url = article_url if article_url.startswith('http') else site['url'].rstrip('/') + '/' + article_url.lstrip('/')
                    print(f"Scraping Article: {article_url}")
                    article_res = get_page_content(article_url)
                    if not article_res:
                        print(f"⚠️ Failed to fetch article {article_url}")
                        continue
                    article_soup = BeautifulSoup(article_res, 'html.parser')
                    
                    # Extract Data
                    title_elem = article_soup.select_one(site['title_selector'])
                    if not title_elem:
                        with open(f"failed_{site['name']}_{time.time()}.html", "w", encoding="utf-8") as f:
                            f.write(article_res)
                        print(f"No Title Found for {article_url}, saved HTML for debugging")
                        continue
                    title = clean_text(title_elem.text)
                    print(f"Title: {title[:50]}...")
                    
                    content_paras = article_soup.select(site['content_selector'])[:3]
                    content = ' '.join([clean_text(p.text) for p in content_paras])
                    if not content:
                        with open(f"failed_{site['name']}_{time.time()}.html", "w", encoding="utf-8") as f:
                            f.write(article_res)
                        print(f"No Content Found for {article_url}, saved HTML for debugging")
                        continue
                    print(f"Content: {content[:100]}...")
                    
                    # Detect Language
                    lang = detect_language(content)
                    print(f"Detected Language: {lang}")
                    
                    # Summarize
                    summary = content[:200]  # Fallback
                    if lang == 'en' and summarizer:
                        try:
                            summary = summarizer(content, max_length=100, min_length=60, do_sample=False)[0]['summary_text']
                            print(f"Summary: {summary[:100]}...")
                        except Exception as e:
                            print(f"⚠️ Summarization Error for {article_url}: {e}")
                    else:
                        print(f"Non-English content ({lang}), using raw content as summary")
                    
                    # Translate to multiple languages
                    translations = {
                        'en': translate_text(summary, 'en'),
                        'hi': translate_text(summary, 'hi'),
                        'mr': translate_text(summary, 'mr'),
                        'ta': translate_text(summary, 'ta'),
                        'te': translate_text(summary, 'te')
                    }
                    for lang, trans in translations.items():
                        print(f"{lang.upper()} Translation: {trans[:100]}...")
                    
                    # Extract Date
                    publish_date = extract_date(article_soup, site['date_selector'])
                    print(f"Publish Date: {publish_date}")
                    
                    # Save to Firebase
                    for attempt in range(3):
                        try:
                            doc_ref = db.collection("news").document()
                            doc_ref.set({
                                "title": title,
                                "content": content,
                                "language": lang,
                                "summary": summary,
                                "translations": translations,
                                "image": extract_image(article_soup, site['image_selector']),
                                "location": extract_location(content),
                                "category": detect_category(title, content),
                                "source": site['name'],
                                "publish_date": publish_date,
                                "timestamp": firestore.SERVER_TIMESTAMP,
                                "url": article_url
                            })
                            print(f"✅ Saved: {title[:50]}... from {site['name']}")
                            break
                        except Exception as e:
                            print(f"⚠️ Firebase Save Error (Attempt {attempt+1}/3) for {article_url}: {e}")
                            if attempt < 2:
                                time.sleep(5)
                            else:
                                with open("failed_articles.json", "a") as f:
                                    json.dump({"title": title, "url": article_url, "source": site['name']}, f)
                                print(f"🚨 Failed to save {article_url} after 3 attempts")
                    
                    time.sleep(random.randint(3, 7))
                
                except Exception as e:
                    print(f"⚠️ Article Error in {site['name']} ({article_url}): {e}")
                    continue
                    
        except Exception as e:
            print(f"🚨 Site Error ({site['name']}): {e}")
            continue

if __name__ == "__main__":
    scrape_and_save()

