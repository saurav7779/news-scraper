import requests
import json
import random
import time
import re
import os
from bs4 import BeautifulSoup
from transformers import pipeline, MBart50TokenizerFast, MBartForConditionalGeneration
from googletrans import Translator
from datetime import datetime, date
import firebase_admin
from firebase_admin import credentials, firestore
import langid
from dateutil.parser import parse
from playwright.sync_api import sync_playwright

# Firebase setup
firebase_key_content = os.environ.get('FIREBASE_KEY')
if firebase_key_content:
    with open('firebase_key.json', 'w') as f:
        f.write(firebase_key_content)
try:
    cred = credentials.Certificate("D:\\Saurav\\NEW MKS\\news_scraper\\firebase_key.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
except Exception as e:
    print(f"🚨 Firebase Error: {e}")
    exit(1)

# Load news sites
try:
    with open('news_sites.json') as f:
        news_sites = json.load(f)
except Exception as e:
    print(f"🚨 Error loading news_sites.json: {e}")
    exit(1)

# Initialize summarizer
summarizer = None
tokenizer = None
try:
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=-1)
except Exception as e:
    print(f"⚠️ Model Error: {e}. No summarization.")

# Initialize translator
translator = Translator()

# Helper functions
def get_random_headers():
    return {'User-Agent': random.choice([
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Safari/14.1.2',
        'Mozilla/5.0 (X11; Linux x86_64) Chrome/92.0.4515.107'
    ])}

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip() if text else ''

def detect_language(text):
    try:
        return langid.classify(text)[0]
    except:
        return 'en'

def translate_text(text, lang='hi', retries=3):
    if not text or len(text.strip()) < 5:
        return text
    for attempt in range(retries):
        try:
            time.sleep(random.uniform(1, 3))
            return translator.translate(text, dest=lang).text
        except Exception as e:
            print(f"⚠️ Translation error for {lang} (Attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(random.uniform(5, 10))
    return text

def extract_image(soup, selector):
    for sel in selector.split(','):
        img = soup.select_one(sel.strip())
        if img and 'src' in img.attrs:
            return img['src']
    return None

def extract_date(soup, selector):
    for sel in selector.split(','):
        date_element = soup.select_one(sel.strip())
        if date_element:
            date_text = date_element.get('content') or date_element.get_text(strip=True)
            try:
                return parse(date_text, fuzzy=True).isoformat()
            except:
                continue
    return datetime.now().isoformat()

def is_valid_date(publish_date, scrape_date):
    try:
        pub_date = datetime.fromisoformat(publish_date.replace('Z', '+00:00'))
        return pub_date.date() <= scrape_date
    except:
        return True

def extract_location(content):
    locations = ['Delhi', 'Mumbai', 'Bihar', 'Uttar Pradesh', 'Kolkata', 'Bangalore', 'Chennai', 
                 'दिल्ली', 'मुंबई', 'बिहार', 'उत्तर प्रदेश', 'कोलकाता', 'बेंगलुरु', 'चेन्नई']
    for loc in locations:
        if loc.lower() in content.lower():
            return loc
    return 'Unknown'

def detect_category(title, content):
    categories = {
        'politics': ['election', 'government', 'minister', 'congress', 'bjp', 'प्रधानमंत्री', 'सरकार', 'चुनाव'],
        'business': ['economy', 'stock', 'market', 'finance', 'business', 'व्यवसाय', 'टैक्स', 'जीएसटी', 'वित्त'],
        'sports': ['cricket', 'football', 'hockey', 'olympics', 'athlete', 'क्रिकेट', 'खेल'],
        'technology': ['tech', 'gadgets', 'software', 'ai', 'internet', 'तकनीक', 'गैजेट'],
        'health': ['health', 'disease', 'medicine', 'hospital', 'doctor', 'स्वास्थ्य', 'दवा'],
        'science': ['science', 'research', 'climate', 'space', 'environment', 'विज्ञान', 'पर्यावरण'],
        'world': ['international', 'global', 'world', 'foreign', 'diplomacy', 'अंतरराष्ट्रीय', 'विश्व'],
        'entertainment': ['bollywood', 'hollywood', 'movie', 'tv', 'celebrity', 'बॉलीवुड', 'फिल्म']
    }
    text = (title + ' ' + content).lower()
    for category, keywords in categories.items():
        if any(keyword in text for keyword in keywords):
            return category
    return 'general'

def summarize_text(text, is_title=False):
    if not text or len(text.strip()) < 20:
        return text[:50] if is_title else text[:150]
    if not summarizer:
        print("⚠️ No summarizer, returning truncated text")
        return text[:50] if is_title else text[:150]
    try:
        text = text[:1500]  # Shortened for speed
        inputs = tokenizer(text, max_length=512, truncation=True, return_tensors="pt")
        decoded_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
        summary = summarizer(decoded_text, max_new_tokens=15 if is_title else 100, min_length=5 if is_title else 40, do_sample=False)
        return clean_text(summary[0]['summary_text'])
    except Exception as e:
        print(f"⚠️ Summarization Error: {e}")
        return text[:50] if is_title else text[:150]

def fetch_page(url, retries=3):
    with sync_playwright() as p:
        browser = None
        try:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            for attempt in range(retries):
                try:
                    print(f"Fetching {url} (Attempt {attempt+1}/{retries})")
                    page.goto(url, timeout=300000, wait_until='networkidle')
                    return page.content()
                except Exception as e:
                    print(f"⚠️ Playwright Error: {e}")
                    if attempt < retries - 1:
                        time.sleep(random.randint(5, 15))
            print(f"🚨 Playwright failed for {url}")
        finally:
            if browser:
                browser.close()
    
    print(f"⚠️ Falling back to requests.get for {url}")
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=get_random_headers(), timeout=60)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"⚠️ Requests Error: {e}")
            if attempt < retries - 1:
                time.sleep(random.randint(5, 15))
    print(f"🚨 Failed to fetch {url}")
    return None

def is_article_duplicate(url):
    try:
        query = db.collection("news").where("url", "==", url).limit(1).get(timeout=600)
        return len(query) > 0
    except Exception as e:
        print(f"⚠️ Firebase Duplicate Error: {e}")
        return False

def is_valid_article_url(url, site_name):
    invalid_patterns = [r'\/category\/', r'\/news\/[^\/]+$', r'\/tags\/', r'\/topics\/', r'\/breakingnews\/']
    valid_domain = url.startswith(('https://www.aajtak.in', 'https://www.abplive.com', 'https://www.gnttv.com')) if site_name in ["Aaj Tak", "ABP News", "GNTTV"] else url.startswith(site['url'])
    if site_name == "Aaj Tak":
        return bool(re.search(r'\/story\/', url)) and not any(re.search(pattern, url) for pattern in invalid_patterns) and valid_domain
    return not any(re.search(pattern, url) for pattern in invalid_patterns) and valid_domain

def scrape_and_save():
    scrape_date = date.today()
    for site in news_sites:
        print(f"Scraping {site['name']}...")
        html = fetch_page(site['url'])
        if not html:
            print(f"🚨 Failed to fetch {site['name']}")
            continue
        soup = BeautifulSoup(html, 'html.parser')
        article_links = soup.select(site['article_link_selector'])[:10]
        random.shuffle(article_links)
        print(f"Found {len(article_links)} articles")
        
        for link in article_links:
            try:
                article_url = link.get('href', '')
                if not article_url:
                    print(f"⚠️ No href in {site['name']}")
                    continue
                if not article_url.startswith('http'):
                    article_url = site['url'].rstrip('/') + '/' + article_url.lstrip('/')
                
                if not is_valid_article_url(article_url, site['name']):
                    print(f"⚠️ Skipping non-article: {article_url}")
                    continue
                
                if is_article_duplicate(article_url):
                    print(f"⚠️ Skipping duplicate: {article_url}")
                    continue
                
                article_html = fetch_page(article_url)
                if not article_html:
                    print(f"🚨 No Content for {article_url}")
                    continue
                article_soup = BeautifulSoup(article_html, 'html.parser')

                # Extract Title
                title = None
                for selector in site['title_selector'].split(','):
                    title_element = article_soup.select_one(selector.strip())
                    if title_element:
                        title = clean_text(title_element.get('content') or title_element.get_text(strip=True))
                        break
                if not title:
                    print(f"🚨 No Title for {article_url}")
                    continue
                print(f"Title: {title[:100]}")

                # Summarize Title
                summarized_title = summarize_text(title, is_title=True) or title[:50]
                print(f"Summarized title: {summarized_title[:100]}")

                # Translate Title
                title_translations = {
                    'hi': translate_text(summarized_title, 'hi'),
                    'mr': translate_text(summarized_title, 'mr'),
                    'ta': translate_text(summarized_title, 'ta'),
                    'te': translate_text(summarized_title, 'te')
                }

                # Extract Content
                content = ''
                for selector in site['content_selector'].split(','):
                    content_elements = article_soup.select(selector.strip())
                    if content_elements:
                        content = ' '.join(clean_text(elem.get_text(strip=True)) for elem in content_elements if elem.get_text(strip=True))
                        break
                if not content:
                    print(f"🚨 No Content for {article_url}")
                    continue
                print(f"Content len: {len(content)}")

                # Extract Date
                publish_date = extract_date(article_soup, site['date_selector'])
                if not is_valid_date(publish_date, scrape_date):
                    print(f"⚠️ Skipping future article: {article_url}")
                    continue

                # Detect Language
                lang = detect_language(title + ' ' + content)

                # Summarize Content
                summary = summarize_text(content) or content[:150]

                # Translate Summary
                translations = {
                    'hi': translate_text(summary, 'hi'),
                    'mr': translate_text(summary, 'mr'),
                    'ta': translate_text(summary, 'ta'),
                    'te': translate_text(summary, 'te')
                }

                # Save to Firebase
                try:
                    db.collection("news").document().set({
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
                    }, timeout=600)
                    print(f"✅ Saved: {summarized_title[:50]}... from {site['name']}")
                except Exception as e:
                    print(f"⚠️ Firebase Save Error: {e}")
                
                time.sleep(random.randint(3, 7))
            
            except Exception as e:
                print(f"⚠️ Article Error in {site['name']} ({article_url}): {e}")
                continue
    
if __name__ == "__main__":
    scrape_and_save()
