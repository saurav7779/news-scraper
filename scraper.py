import requests
import json
import random
import time
import re
import os
from bs4 import BeautifulSoup
from transformers import pipeline
from googletrans import Translator
from datetime import datetime, date
import firebase_admin
from firebase_admin import credentials, firestore
import langdetect
from dateutil.parser import parse
from playwright.sync_api import sync_playwright

# Handle Firebase key from env (for GitHub Actions)
firebase_key_file = 'serviceAccountKey.json'  # Sync with workflow
firebase_key_content = os.environ.get('FIREBASE_KEY')

if firebase_key_content:
    try:
        with open(firebase_key_file, 'w') as f:
            f.write(firebase_key_content)
        print(f"Firebase key written to {firebase_key_file}")
    except Exception as e:
        print(f"🚨 Error writing Firebase key to file: {e}")
        exit(1)
else:
    print(f"Warning: FIREBASE_KEY env not set, checking for existing {firebase_key_file}")
    if not os.path.exists(firebase_key_file):
        print(f"🚨 No Firebase key file found at {firebase_key_file}!")
        exit(1)

# Initialize Firebase with offline persistence
try:
    cred = credentials.Certificate(firebase_key_file)
    firebase_admin.initialize_app(cred, {'firestorePersistence': True})
    db = firestore.client()
    print("Firebase initialized successfully.")
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
    # Using t5-small for faster processing
    summarizer = pipeline("summarization", model="t5-small")
except Exception as e:
    print(f"⚠️ Model Initialization Error: {e}. Proceeding without summarization.")
translator = Translator()

# Helper Functions
def get_random_headers():
    return {
        'User-Agent': random.choice([
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36'
        ])
    }

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip() if text else ''

def detect_language(text):
    try:
        return langdetect.detect(text)
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
    date_element = soup.select_one(selector)
    if date_element:
        date_text = date_element.get('content') or date_element.get_text(strip=True)
        try:
            parsed_date = parse(date_text, fuzzy=True)
            return parsed_date.isoformat()
        except:
            return datetime.now().isoformat()
    return datetime.now().isoformat()

def is_valid_date(publish_date, scrape_date):
    try:
        pub_date = datetime.fromisoformat(publish_date.replace('Z', '+00:00'))
        return pub_date.date() <= scrape_date
    except:
        return True

def extract_location(content):
    locations = ['Delhi', 'Mumbai', 'Bihar', 'Uttar Pradesh', 'Kolkata', 'Bangalore', 'Chennai']
    for loc in locations:
        if loc.lower() in content.lower():
            return loc
    return 'Unknown'

def detect_category(title, content):
    categories = {
        'politics': ['election', 'government', 'minister', 'congress', 'bjp'],
        'business': ['economy', 'stock', 'market', 'finance', 'business'],
        'sports': ['cricket', 'football', 'hockey', 'olympics', 'athlete'],
        'technology': ['tech', 'gadgets', 'software', 'ai', 'internet'],
        'health': ['health', 'disease', 'medicine', 'hospital', 'doctor'],
        'science': ['science', 'research', 'climate', 'space', 'environment'],
        'world': ['international', 'global', 'world', 'foreign', 'diplomacy'],
        'entertainment': ['bollywood', 'hollywood', 'movie', 'tv', 'celebrity']
    }
    text = (title + ' ' + content).lower()
    for category, keywords in categories.items():
        if any(keyword in text for keyword in keywords):
            return category
    return 'general'

def summarize_text(text, is_title=False):
    if not text or len(text.strip()) < 20:  # Stricter validation
        return text[:50] if is_title else text[:150]
    if not summarizer:
        return text[:50] if is_title else text[:150]
    try:
        max_len = 15 if is_title else 150
        min_len = 5 if is_title else 60
        summary = summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)
        return clean_text(summary[0]['summary_text'])
    except Exception as e:
        print(f"⚠️ Summarization Error: {e}")
        return text[:50] if is_title else text[:150]

def fetch_page(url, retries=5):  # Increased retries
    # Try Playwright first
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(bypass_csp=True, ignore_https_errors=True)
        page = context.new_page()
        for attempt in range(retries):
            try:
                page.goto(url, timeout=150000)  # Increased to 150s
                page.wait_for_load_state('networkidle', timeout=150000)  # Changed to networkidle
                html = page.content()
                context.close()
                browser.close()
                return html
            except Exception as e:
                print(f"⚠️ Playwright Error for {url} (Attempt {attempt+1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(random.randint(5, 15))  # Increased delay
        context.close()
        browser.close()
    
    # Fallback to requests.get
    print(f"⚠️ Falling back to requests.get for {url}")
    try:
        response = requests.get(url, headers=get_random_headers(), timeout=90)  # Increased to 90s
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"⚠️ Requests Error for {url}: {e}")
        return None

def is_article_duplicate(url):
    try:
        query = db.collection("news").where(filter=firestore.FieldFilter("url", "==", url)).limit(1).get(timeout=900)  # Increased timeout
        return len(query) > 0
    except Exception as e:
        print(f"⚠️ Firebase Duplicate Check Error: {e}")
        return False

def is_valid_article_url(url, site_name):
    invalid_patterns = [r'\/category\/', r'\/news\/[^\/]+$', r'\/tags\/', r'\/topics\/', r'\/breakingnews\/']
    valid_domain = url.startswith(('https://www.aajtak.in', 'https://www.abplive.com', 'https://www.gnttv.com')) if site_name in ["Aaj Tak", "ABP News", "GNTTV"] else url.startswith(site['url'])
    if site_name == "ABP News":
        return bool(re.search(r'abpp-\d+', url)) and not any(re.search(pattern, url) for pattern in invalid_patterns) and valid_domain
    elif site_name == "Aaj Tak":
        return bool(re.search(r'\/story\/', url)) and not any(re.search(pattern, url) for pattern in invalid_patterns) and valid_domain
    elif site_name == "GNTTV":
        return bool(re.search(r'\/story\/', url)) and not any(re.search(pattern, url) for pattern in invalid_patterns) and valid_domain
    return not any(re.search(pattern, url) for pattern in invalid_patterns) and valid_domain

def scrape_and_save():
    scrape_date = date.today()
    for site in news_sites:
        print(f"Scraping {site['name']}...")
        try:
            html = fetch_page(site['url'])
            if not html:
                print(f"🚨 Failed to fetch {site['name']} homepage")
                continue
            soup = BeautifulSoup(html, 'html.parser')
            article_links = soup.select(site['article_link_selector'])
            random.shuffle(article_links)
            article_links = article_links[:10]
            print(f"Found {len(article_links)} articles for {site['name']}: {[link.get('href', 'No href') for link in article_links]}")
            
            articles_scraped = 0
            for link in article_links:
                if articles_scraped >= 10:
                    break
                try:
                    article_url = link.get('href', '')
                    if not article_url:
                        print(f"⚠️ No href found for link in {site['name']}")
                        continue
                    if not article_url.startswith('http'):
                        article_url = site['url'].rstrip('/') + '/' + article_url.lstrip('/')
                    
                    if not is_valid_article_url(article_url, site['name']):
                        print(f"⚠️ Skipping non-article URL: {article_url}")
                        continue
                    
                    if is_article_duplicate(article_url):
                        print(f"⚠️ Skipping duplicate article: {article_url}")
                        continue
                    
                    article_html = fetch_page(article_url)
                    if not article_html:
                        print(f"🚨 No Content Found for {article_url}")
                        continue
                    article_soup = BeautifulSoup(article_html, 'html.parser')

                    # Extract Title
                    title = None
                    for selector in site['title_selector'].split(','):
                        title_element = article_soup.select_one(selector.strip())
                        if title_element:
                            if title_element.name == 'meta':
                                title = clean_text(title_element.get('content'))
                            else:
                                title = clean_text(title_element.get_text(strip=True))
                            break
                    if not title:
                        print(f"🚨 No Title Found for {article_url}. Selectors tried: {site['title_selector']}")
                        continue

                    # Summarize Title
                    summarized_title = summarize_text(title, is_title=True)
                    if not summarized_title:
                        print(f"⚠️ Title Summarization failed for {article_url}")
                        summarized_title = title[:50]

                    # Translate Summarized Title
                    title_translations = {
                        'hi': translate_text(summarized_title, 'hi'),
                        'mr': translate_text(summarized_title, 'mr'),
                        'ta': translate_text(summarized_title, 'ta'),
                        'te': translate_text(summarized_title, 'te')
                    }
                    for lang, trans in title_translations.items():
                        print(f"{lang.upper()} Title Translation: {trans[:100]}...")

                    # Extract Content with Debug
                    content = ''
                    for selector in site['content_selector'].split(','):
                        content_elements = article_soup.select(selector.strip())
                        if content_elements:
                            content = ' '.join(clean_text(elem.get_text(strip=True)) for elem in content_elements if elem.get_text(strip=True))
                            print(f"✅ Content found with selector: {selector} for {article_url}")
                            print(f"Content snippet: {content[:100]}...")
                            break
                    if not content:
                        print(f"🚨 No Content Found for {article_url}. Selectors tried: {site['content_selector']}")
                        all_p_tags = article_soup.select('p')
                        print(f"Debug: Found {len(all_p_tags)} <p> tags: {[clean_text(p.get_text(strip=True))[:100] for p in all_p_tags[:3]]}")
                        continue

                    # Extract Date and Validate
                    publish_date = extract_date(article_soup, site['date_selector'])
                    if not is_valid_date(publish_date, scrape_date):
                        print(f"⚠️ Skipping article {article_url}: Publish date {publish_date} is in future")
                        continue

                    # Detect Language
                    lang = detect_language(title + ' ' + content)

                    # Summarize Content
                    summary = summarize_text(content)
                    if not summary:
                        print(f"⚠️ Summarization failed for {article_url}")
                        continue

                    # Translate Content Summary
                    translations = {
                        'hi': translate_text(summary, 'hi'),
                        'mr': translate_text(summary, 'mr'),
                        'ta': translate_text(summary, 'ta'),
                        'te': translate_text(summary, 'te')
                    }
                    for lang, trans in translations.items():
                        print(f"{lang.upper()} Content Translation: {trans[:100]}...")

                    print(f"Publish Date: {publish_date}")

                    # Save to Firebase
                    for attempt in range(7):  # Increased retries
                        try:
                            doc_ref = db.collection("news").document()
                            doc_ref.set({
                                "original_title": title,
                                "summarized_title": summarized_title,
                                "title_translations": title_translations,
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
                            }, timeout=900)  # Increased timeout
                            print(f"✅ Saved: {summarized_title[:50]}... from {site['name']}")
                            articles_scraped += 1
                            break
                        except Exception as e:
                            print(f"⚠️ Firebase Save Error (Attempt {attempt+1}/7) for {article_url}: {e}")
                            if attempt < 6:
                                time.sleep(random.randint(5, 15))
                            else:
                                print(f"🚨 Failed to save {article_url} after 7 attempts")
                    
                    time.sleep(random.randint(3, 7))
                
                except Exception as e:
                    print(f"⚠️ Article Error in {site['name']} ({article_url}): {e}")
                    continue
                    
        except Exception as e:
            print(f"🚨 Site Error ({site['name']}): {e}")

if __name__ == "__main__":
    scrape_and_save()
