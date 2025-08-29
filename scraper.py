import requests
import json
import random
import time
import re
import os
from bs4 import BeautifulSoup
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import translators as ts
from datetime import datetime, date, timedelta
import firebase_admin
from firebase_admin import credentials, firestore
import langid
from dateutil.parser import parse
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
import logging
from sentence_transformers import SentenceTransformer, util
import pickle
from collections import defaultdict
import hashlib

# Logging setup with emojis
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def log_error(msg):
    logger.error(f"🚨 🚨 {msg}")

def log_warning(msg):
    logger.warning(f"⚠️ {msg}")

def log_info(msg):
    logger.info(f"✅ {msg}")

def log_debug(msg):
    logger.debug(f"📝 {msg}")

# Firebase setup
firebase_key_content = os.environ.get('FIREBASE_KEY')
if firebase_key_content:
    with open('firebase_key.json', 'w') as f:
        f.write(firebase_key_content)
try:
    cred = credentials.Certificate("firebase_key.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
except Exception as e:
    log_error(f"Firebase Error: {e}")
    exit(1)

# Load news sites
try:
    with open('news_sites.json') as f:
        news_sites = json.load(f)
except Exception as e:
    log_error(f"Error loading news_sites.json: {e}")
    exit(1)

# Initialize summarizer
summarizer = None
tokenizer = None
try:
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=-1)
except Exception as e:
    log_warning(f"Model Error: {e}. No summarization.")
    summarizer = None

# Initialize sentence transformer
try:
    similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    log_warning(f"Sentence Transformer Error: {e}. Similarity checks disabled.")
    similarity_model = None

# Cache setup
cache_file = 'article_cache.pkl'
translation_cache_file = 'translation_cache.pkl'
article_cache = {}
translation_cache = {}
try:
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            article_cache = pickle.load(f)
    if os.path.exists(translation_cache_file):
        with open(translation_cache_file, 'rb') as f:
            translation_cache = pickle.load(f)
except Exception as e:
    log_warning(f"Cache Load Error: {e}")

# Helper functions
def get_random_headers():
    return {
        'User-Agent': random.choice([
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/120.0'
        ]),
        'Accept-Language': 'en-US,en;q=0.9,hi;q=0.8',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Referer': 'https://www.google.com/',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }

def clean_text(text):
    if not text:
        return ''
    text = re.sub(r'(summarize|generate|in|concise|factual|manner|key points|excluding|opinions|minor details|words|80-150|60-80|10-15).*?:', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text.replace('OPINION |', '').strip())
    return text

def detect_language(text, site_name):
    site_lang_map = {
        'Aaj Tak': 'hi', 'ABP News': 'hi', 'GNTTV': 'hi', 'Times of India': 'en',
        'Marathi Abplive': 'mr', 'Zee Bihar Jharkhand': 'hi', 'First Bihar': 'hi',
        'Zee News': 'hi', 'CNN': 'en', 'The Guardian': 'en', 'BBC': 'en',
        'India TV': 'en', 'News 18': 'hi', 'Amar Ujala': 'hi', 'The Hindu': 'en',
        'Hindustan': 'hi', 'Jagran': 'hi'
    }
    if site_name in site_lang_map:
        return site_lang_map[site_name]
    if text:
        try:
            lang, confidence = langid.classify(text)
            if confidence > 0.7:
                return lang
        except:
            pass
    return 'en'

def is_valid_translation(text, target_lang):
    if not text or len(text.strip()) < 5:
        return False
    invalid_patterns = ['MYMEMORY WARNING', 'Unable to translate', 'Translation failed']
    return not any(pattern.lower() in text.lower() for pattern in invalid_patterns)

def translate_text(text, target_lang='hi', source_lang='auto', retries=3):
    if not text or len(text.strip()) < 5 or source_lang == target_lang:
        return text
    cache_key = hashlib.md5(f"{text}_{source_lang}_{target_lang}".encode()).hexdigest()
    if cache_key in translation_cache:
        log_info(f"Using cached translation for {cache_key}: {translation_cache[cache_key][:50]}...")
        return translation_cache[cache_key]
    
    for attempt in range(retries):
        try:
            time.sleep(random.uniform(5, 10))
            translated = ts.translate_text(text, from_language=source_lang, to_language=target_lang)
            if (translated and 
                len(translated.strip()) > 3 and 
                translated.strip().lower() != text.strip().lower() and
                is_valid_translation(translated, target_lang)):
                log_info(f"📝 Translated to {target_lang}: {translated[:50]}...")
                translation_cache[cache_key] = translated
                try:
                    with open(translation_cache_file, 'wb') as f:
                        pickle.dump(translation_cache, f)
                except Exception as e:
                    log_warning(f"Translation Cache Save Error: {e}")
                return translated
            log_warning(f"Invalid translation to {target_lang}: '{translated}' (Original: '{text}')")
        except Exception as e:
            log_warning(f"Translation error for {target_lang} (Attempt {attempt+1}/{retries}): {e}")
        if attempt < retries - 1:
            time.sleep(random.uniform(5, 10))
    log_error(f"Translation failed for {target_lang}, returning original text")
    return text

def extract_image(soup, selector):
    for sel in selector.split(','):
        sel = sel.strip()
        img = soup.select_one(sel)
        if img:
            if sel.startswith('meta'):
                return img.get('content', '')
            return img.get('src') or img.get('data-src', '')
    return ''

def extract_date(soup, selector):
    for sel in selector.split(','):
        sel = sel.strip()
        date_elem = soup.select_one(sel)
        if date_elem:
            date_text = date_elem.get('content') or date_elem.get_text(strip=True)
            try:
                parsed_date = parse(date_text, fuzzy=True).strftime('%Y-%m-%d')
                return parsed_date
            except:
                continue
    return datetime.now().strftime('%Y-%m-%d')

def is_valid_date(date_str, scrape_date):
    try:
        article_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        return article_date <= scrape_date and article_date >= scrape_date - timedelta(days=2)  # 48 hours
    except:
        return False

def is_valid_article_url(url, site_name, site):
    if not url or "404" in url.lower() or "page not found" in url.lower():
        return False
    exclude_patterns = ['/video/', '/live/', '/category/', '/photos/', '/gallery/', '/login', '/web-stories', '/photo-gallery', '/live-tv', '/tags/', '/bhojpuri-cinema']
    if site_name == 'Zee Bihar Jharkhand':
        url = re.sub(r'/hindi/india/bihar-jharkhand/hindi/india/bihar-jharkhand/', '/hindi/india/bihar-jharkhand/', url)
        if not url.startswith('http'):
            url = site['url'].rstrip('/') + '/' + url.lstrip('/')
    elif site_name == 'BBC':
        url = url.replace('/news/news/', '/news/')
        if not url.startswith('http'):
            url = 'https://www.bbc.com' + url
    return not any(pat in url.lower() for pat in exclude_patterns) and site['url'].split('/')[2] in url

def is_article_duplicate(url):
    if url in article_cache:
        return True
    try:
        query = db.collection("news").where(filter=firestore.FieldFilter("url", "==", url)).limit(1).get()
        return len(query) > 0
    except Exception as e:
        log_warning(f"Duplicate Check Error: {e}")
        return False

def is_content_duplicate(summary):
    if not summary or not similarity_model:
        return False
    try:
        summary_embedding = similarity_model.encode(summary, convert_to_tensor=True)
        for cached_url, cached_data in article_cache.items():
            if 'summary_embedding' in cached_data:
                cached_embedding = cached_data['summary_embedding']
                similarity = util.cos_sim(summary_embedding, cached_embedding).item()
                if similarity > 0.85:
                    log_info(f"Duplicate content detected: {summary[:50]}... similar to cached article {cached_url} (Similarity: {similarity:.2f})")
                    return True
        try:
            query = db.collection("news").limit(50).get()
            for doc in query:
                doc_summary = doc.to_dict().get('summary', '')
                if doc_summary:
                    doc_embedding = similarity_model.encode(doc_summary, convert_to_tensor=True)
                    similarity = util.cos_sim(summary_embedding, doc_embedding).item()
                    if similarity > 0.85:
                        log_info(f"Duplicate content detected: {summary[:50]}... similar to existing article {doc.to_dict().get('url', 'unknown')} (Similarity: {similarity:.2f})")
                        return True
        except Exception as e:
            log_warning(f"Firebase Duplicate Check Error: {e}")
        return False
    except Exception as e:
        log_warning(f"Content Duplicate Check Error: {e}")
        return False

def count_words(text):
    return len(text.split())

def detect_repetition(summary):
    if not summary:
        return False
    sentences = summary.split('. ')
    ngrams = set()
    for sentence in sentences:
        words = sentence.lower().split()
        for i in range(len(words) - 1):
            ngram = ' '.join(words[i:i+2])
            if ngram in ngrams:
                return True
            ngrams.add(ngram)
    return False

def check_title_similarity(title, content):
    if not similarity_model or not title or not content:
        return True
    try:
        title_embedding = similarity_model.encode(title, convert_to_tensor=True)
        content_sentences = content.split('. ')[:3]
        content_embedding = similarity_model.encode(' '.join(content_sentences), convert_to_tensor=True)
        similarity = util.cos_sim(title_embedding, content_embedding).item()
        log_debug(f"Title similarity score: {similarity:.2f} for title: {title[:50]}...")
        return similarity > 0.3
    except:
        return True

def extract_state_district(content):
    states = {
        'Andhra Pradesh': ['andhra pradesh', 'amaravati', 'visakhapatnam', 'vijayawada'],
        'Bihar': ['bihar', 'patna', 'gaya', 'muzaffarpur'],
        'Delhi': ['delhi', 'new delhi'],
        'Uttar Pradesh': ['uttar pradesh', 'lucknow', 'kanpur', 'varanasi'],
        'Maharashtra': ['maharashtra', 'mumbai', 'pune', 'nagpur'],
        'Karnataka': ['karnataka', 'bangalore', 'bengaluru', 'mysore'],
        'Tamil Nadu': ['tamil nadu', 'chennai', 'coimbatore', 'madurai'],
        'West Bengal': ['west bengal', 'kolkata', 'howrah', 'durgapur']
    }
    content_lower = content.lower()
    state, district = 'Unknown', 'Unknown'
    for st, keywords in states.items():
        for keyword in keywords:
            if keyword in content_lower:
                state = st
                district = keyword if keyword not in [st.lower(), 'new delhi'] else 'Unknown'
                break
        if state != 'Unknown':
            break
    return state, district

def detect_category(title, content):
    categories = {
        'defence': ['defence', 'military', 'fighter', 'jet', 'aircraft', 'tejas', 'hal', 'aerospace', 'safran', 'missile', 'army', 'navy', 'air force'],
        'international': ['international', 'global', 'foreign', 'diplomacy', 'visa', 'immigration', 'united nations', 'summit', 'trade agreement'],
        'politics': ['election', 'government', 'minister', 'pm', 'cm', 'parliament', 'congress', 'bjp', 'political', 'policy', 'law', 'bill', 'act', 'prime minister', 'president'],
        'sports': ['cricket', 'football', 'match', 'sport', 'player', 'tournament', 'olympics', 'medal', 'coach', 'team', 'ipl', 'world cup', 'championship'],
        'business': ['market', 'stock', 'business', 'economy', 'company', 'industry', 'trade', 'commerce', 'gst', 'tax', 'investment', 'finance', 'bank', 'rupee', 'dollar'],
        'entertainment': ['movie', 'film', 'actor', 'bollywood', 'hollywood', 'celebrity', 'music', 'song', 'album', 'director', 'entertainment', 'actor', 'actress'],
        'crime': ['crime', 'arrest', 'police', 'murder', 'theft', 'robbery', 'accident', 'investigation', 'court', 'judge', 'lawyer', 'case', 'illegal', 'violation', 'homicide', 'fraud'],
        'technology': ['technology', 'tech', 'computer', 'mobile', 'app', 'software', 'internet', 'digital', 'ai', 'artificial intelligence', 'smartphone', 'social media', 'iphone', 'foxconn', 'apple'],
        'health': ['health', 'medical', 'hospital', 'doctor', 'disease', 'medicine', 'vaccine', 'covid', 'corona', 'healthcare', 'treatment', 'patient'],
        'education': ['education', 'school', 'college', 'university', 'student', 'teacher', 'exam', 'result', 'education', 'board', 'degree', 'internship']
    }
    text = (title.lower() * 2 + ' ' + content.lower())
    category_scores = {cat: 0 for cat in categories}
    for cat, keywords in categories.items():
        for keyword in keywords:
            if keyword in text:
                category_scores[cat] += 2 if keyword in title.lower() else 1
    max_score = max(category_scores.values())
    if max_score == 0:
        return 'general'
    return max(category_scores, key=category_scores.get)

def summarize_text(text, is_title=False, source_lang='en'):
    if not summarizer or not text or len(text.strip()) < 10:
        log_warning("Summarizer not available or text too short, using fallback")
        sentences = text.split('. ')
        relevant_sentences = [s for s in sentences if len(s.split()) > 3 and not any(x in s.lower() for x in ['advertisement', 'sponsored', 'read more'])]
        if is_title:
            if relevant_sentences:
                fallback_title = clean_text(' '.join(relevant_sentences[0].split()[:15]))
                if count_words(fallback_title) < 10:
                    extra_words = relevant_sentences[1].split()[:15-count_words(fallback_title)] if len(relevant_sentences) > 1 else text.split()[:15-count_words(fallback_title)]
                    fallback_title = clean_text(fallback_title + ' ' + ' '.join(extra_words))
                return fallback_title[:100], count_words(fallback_title) >= 10 and count_words(fallback_title) <= 15
            return clean_text(' '.join(text.split()[:15]))[:100], False
        fallback_summary = ' '.join(relevant_sentences[:3])[:500]
        summary_words = fallback_summary.split()
        if len(summary_words) > 80:
            return clean_text(' '.join(summary_words[:80])) + '.', False
        elif len(summary_words) < 60:
            extra_sentences = relevant_sentences[3:5]
            if extra_sentences:
                fallback_summary = f"{fallback_summary} {' '.join(extra_sentences)}"[:500]
        return clean_text(fallback_summary[:500]) + '.', False

    try:
        english_text = text
        if source_lang != 'en':
            english_text = translate_text(text, 'en', source_lang)
            if not english_text or english_text == text:
                log_warning("Translation to English failed, using fallback")
                sentences = text.split('. ')
                relevant_sentences = [s for s in sentences if len(s.split()) > 3 and not any(x in s.lower() for x in ['advertisement', 'sponsored', 'read more'])]
                if is_title:
                    if relevant_sentences:
                        fallback_title = clean_text(' '.join(relevant_sentences[0].split()[:15]))
                        if count_words(fallback_title) < 10:
                            extra_words = relevant_sentences[1].split()[:15-count_words(fallback_title)] if len(relevant_sentences) > 1 else text.split()[:15-count_words(fallback_title)]
                            fallback_title = clean_text(fallback_title + ' ' + ' '.join(extra_words))
                        return fallback_title[:100], False
                    return clean_text(' '.join(text.split()[:15]))[:100], False
                fallback_summary = ' '.join(relevant_sentences[:3])[:500]
                summary_words = fallback_summary.split()
                if len(summary_words) > 80:
                    return clean_text(' '.join(summary_words[:80])) + '.', False
                elif len(summary_words) < 60:
                    extra_sentences = relevant_sentences[3:5]
                    if extra_sentences:
                        fallback_summary = f"{fallback_summary} {' '.join(extra_sentences)}"[:500]
                return clean_text(fallback_summary[:500]) + '.', False

        max_length = 15 if is_title else 80
        min_length = 10 if is_title else 60
        input_tokens = len(tokenizer.encode(english_text, truncation=True))
        log_debug(f"BART input length: {input_tokens} tokens")
        summary = summarizer(
            english_text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
            num_beams=6,
            length_penalty=1.0,
            early_stopping=True,
            truncation=True
        )[0]['summary_text']
        summary = clean_text(summary)
        
        if any(x in summary.lower() for x in ['summarize', 'in 60-80', 'in 10-15']) or (is_title and not check_title_similarity(summary, english_text)):
            log_warning(f"Invalid summary: {summary[:50]}... Using fallback (Words: {count_words(summary)}, Similarity: {check_title_similarity(summary, english_text)})")
            sentences = english_text.split('. ')
            relevant_sentences = [s for s in sentences if len(s.split()) > 3 and not any(x in s.lower() for x in ['advertisement', 'sponsored', 'read more'])]
            if is_title:
                if relevant_sentences:
                    summary = clean_text(' '.join(relevant_sentences[0].split()[:15]))
                    if count_words(summary) < 10:
                        extra_words = relevant_sentences[1].split()[:15-count_words(summary)] if len(relevant_sentences) > 1 else english_text.split()[:15-count_words(summary)]
                        summary = clean_text(summary + ' ' + ' '.join(extra_words))
                else:
                    summary = clean_text(' '.join(english_text.split()[:15]))
                return summary[:100], False
            else:
                summary = ' '.join(relevant_sentences[:3])[:500]
                summary_words = summary.split()
                if len(summary_words) > 80:
                    summary = ' '.join(summary_words[:80])
                elif len(summary_words) < 60:
                    extra_sentences = relevant_sentences[3:5]
                    if extra_sentences:
                        summary = f"{summary} {' '.join(extra_sentences)}"[:500]
                summary += '.'
                return summary, False
        
        if source_lang != 'en':
            summary = translate_text(summary, source_lang, 'en')
        
        summary_words = summary.split()
        word_count = len(summary_words)
        is_valid = True
        if is_title:
            if word_count > 15:
                summary = ' '.join(summary_words[:15])
                is_valid = False
            elif word_count < 10:
                sentences = english_text.split('. ')
                relevant_sentences = [s for s in sentences if len(s.split()) > 3 and not any(x in s.lower() for x in ['advertisement', 'sponsored', 'read more'])]
                extra_words = relevant_sentences[0].split()[:15-word_count] if relevant_sentences else english_text.split()[:15-word_count]
                summary = clean_text(' '.join(summary_words + extra_words))
                is_valid = False
            log_info(f"Generated title (Words: {len(summary.split())})")
            return clean_text(summary[:100]), word_count >= 10 and word_count <= 15
        else:
            if word_count > 80:
                summary = ' '.join(summary_words[:80])
                is_valid = False
            elif word_count < 60:
                sentences = english_text.split('. ')
                relevant_sentences = [s for s in sentences if len(s.split()) > 3 and not any(x in s.lower() for x in ['advertisement', 'sponsored', 'read more'])]
                extra_sentences = []
                current_count = word_count
                for sentence in relevant_sentences:
                    sentence_words = sentence.split()
                    if current_count + len(sentence_words) <= 80:
                        extra_sentences.append(sentence)
                        current_count += len(sentence_words)
                    if current_count >= 60:
                        break
                if extra_sentences:
                    summary = f"{summary} {' '.join(extra_sentences)}"[:500]
                    is_valid = False
                summary += '.'
            
            if detect_repetition(summary):
                log_warning(f"Repetitive summary detected: {summary[:50]}... Using fallback")
                sentences = english_text.split('. ')
                relevant_sentences = [s for s in sentences if len(s.split()) > 3 and not any(x in s.lower() for x in ['advertisement', 'sponsored', 'read more'])]
                summary = ' '.join(relevant_sentences[:3])[:500]
                summary_words = summary.split()
                if len(summary_words) > 80:
                    summary = ' '.join(summary_words[:80])
                elif len(summary_words) < 60:
                    extra_sentences = relevant_sentences[3:5]
                    if extra_sentences:
                        summary = f"{summary} {' '.join(extra_sentences)}"[:500]
                summary += '.'
                is_valid = False
        
        log_info(f"Generated summary (Words: {len(summary.split())})")
        return clean_text(summary), is_valid

def extract_location(content):
    locations = {
        'India': ['india', 'delhi', 'new delhi', 'mumbai', 'bangalore', 'chennai', 'kolkata', 'hyderabad', 'pune', 'ahmedabad'],
        'USA': ['united states', 'us', 'usa', 'u.s.', 'u.s.a.', 'california', 'new york', 'florida', 'texas', 'washington'],
        'UK': ['united kingdom', 'uk', 'britain', 'london'],
        'France': ['france', 'paris', 'safran'],
        'Russia': ['russia', 'moscow'],
        'China': ['china', 'beijing'],
        'Pakistan': ['pakistan', 'islamabad', 'karachi']
    }
    content_lower = content.lower()
    for loc, keywords in locations.items():
        for keyword in keywords:
            if keyword in content_lower:
                return loc
    return 'Unknown'

def fetch_page(url, retries=3):
    backoff = 5
    for attempt in range(retries):
        try:
            log_info(f"Fetching {url} (Attempt {attempt+1}/{retries})")
            response = requests.get(url, headers=get_random_headers(), timeout=30)
            response.raise_for_status()
            if response.text and len(response.text) > 1000:
                log_debug(f"HTML Content (first 2000 chars):\n{response.text[:2000]}")
                return response.text
        except Exception as e:
            log_warning(f"Requests Error: {e}")
            if attempt < retries - 1:
                time.sleep(random.uniform(backoff, backoff * 2))
                backoff *= 2
    log_info(f"Requests failed for {url}, trying Playwright")
    with sync_playwright() as p:
        browser = None
        try:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_viewport_size({"width": 1280, "height": 720})
            for attempt in range(retries):
                try:
                    log_info(f"Fetching with Playwright {url} (Attempt {attempt+1}/{retries})")
                    page.goto(url, timeout=60000, wait_until='domcontentloaded')
                    page.wait_for_timeout(5000)
                    page.evaluate("window.scrollTo(0, document.body.scrollHeight/2)")
                    page.wait_for_timeout(2000)
                    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    page.wait_for_timeout(2000)
                    content = page.content()
                    if content and len(content) > 1000:
                        log_debug(f"Playwright HTML (first 2000 chars):\n{content[:2000]}")
                        return content
                except PlaywrightTimeoutError as e:
                    log_warning(f"Playwright Timeout Error: {e}")
                    if attempt < retries - 1:
                        time.sleep(random.uniform(backoff, backoff * 2))
                        backoff *= 2
                except Exception as e:
                    log_warning(f"Playwright Error: {e}")
                    if attempt < retries - 1:
                        time.sleep(random.uniform(backoff, backoff * 2))
                        backoff *= 2
        except Exception as e:
            log_error(f"Playwright failed for {url}: {e}")
            return None
        finally:
            if browser:
                browser.close()
    return None

def scrape_and_save():
    scrape_date = date.today()
    report = defaultdict(list)
    for site in news_sites:
        log_info(f"Scraping {site['name']}...")
        time.sleep(random.uniform(10, 20))
        html = fetch_page(site['url'])
        if not html:
            log_error(f"Failed to fetch {site['name']}")
            report['failures'].append(f"{site['name']}: Failed to fetch main page")
            continue
        soup = BeautifulSoup(html, 'html.parser')
        
        article_links = list(set(link.get('href', '') for link in soup.select(site['article_link_selector'])[:15]))
        articles_with_dates = []
        for article_url in article_links:
            if not article_url:
                continue
            if not article_url.startswith('http'):
                article_url = site['url'].rstrip('/') + '/' + article_url.lstrip('/')
            if not is_valid_article_url(article_url, site['name'], site) or is_article_duplicate(article_url):
                continue
            article_html = fetch_page(article_url)
            if not article_html:
                log_warning(f"Failed to fetch article for date extraction: {article_url}")
                report['failures'].append(f"{site['name']}: Failed to fetch {article_url}")
                continue
            article_soup = BeautifulSoup(article_html, 'html.parser')
            publish_date = extract_date(article_soup, site['date_selector'])
            if not is_valid_date(publish_date, scrape_date):
                log_warning(f"Skipping old article: {article_url} (Date: {publish_date})")
                continue
            articles_with_dates.append({
                'url': article_url,
                'publish_date': publish_date
            })
        
        try:
            articles_with_dates.sort(key=lambda x: datetime.strptime(x['publish_date'], '%Y-%m-%d'), reverse=True)
        except Exception as e:
            log_warning(f"Sorting error: {e}. Using unsorted order.")
        
        selected_articles = [article['url'] for article in articles_with_dates[:2]]  # Max 2 articles
        log_info(f"Selected {len(selected_articles)} latest articles from {site['name']}: {selected_articles}")
        
        if not selected_articles:
            log_warning(f"No valid latest articles found for {site['name']}, skipping")
            report['failures'].append(f"{site['name']}: No valid articles found")
            continue
        
        for article_url in selected_articles:
            try:
                time.sleep(random.uniform(3, 7))
                article_html = fetch_page(article_url)
                if not article_html:
                    log_error(f"Failed to fetch article: {article_url}")
                    report['failures'].append(f"{site['name']}: Failed to fetch {article_url}")
                    continue
                article_soup = BeautifulSoup(article_html, 'html.parser')
                
                title = ''
                for selector in site['title_selector'].split(','):
                    title_element = article_soup.select_one(selector.strip())
                    if title_element:
                        title = clean_text(title_element.get_text(strip=True))
                        break
                if not title or "404" in title.lower() or "page not found" in title.lower():
                    log_error(f"Invalid Title or 404 for {article_url}")
                    report['failures'].append(f"{site['name']}: Invalid title for {article_url}")
                    continue
                log_info(f"Title: {title[:100]}")

                content = ''
                content_selectors = site['content_selector'].split(',') + ['p', '.content p', '.article p', '.story-content p', '.articlebody p']
                for selector in content_selectors:
                    selector = selector.strip()
                    content_elements = article_soup.select(selector)
                    if content_elements:
                        filtered_content = []
                        for elem in content_elements:
                            elem_text = clean_text(elem.get_text(strip=True))
                            if (len(elem_text) > 50 and 
                                not any(x in elem_text.lower() for x in [
                                    'advertisement', 'sponsored', 'read more', 'toi tech desk',
                                    'click here', 'follow us', 'share this', 'comment now',
                                    'latest news', 'trending now', 'recommended', 'related news',
                                    'copyright', 'privacy policy', 'terms of use', 'cookie policy'
                                ]) and
                                not re.search(r'₹\d+,\d+', elem_text) and
                                not re.search(r'^\d+\s*\.?\s*$', elem_text)):
                                filtered_content.append(elem_text)
                        if filtered_content:
                            content = ' '.join(filtered_content)
                            log_info(f"Content found using selector: {selector}")
                            break
                
                if not content or len(content) < 100:
                    main_content = article_soup.find('div', {'class': re.compile(r'content|article|story|main|body')})
                    if main_content:
                        paragraphs = main_content.find_all('p')
                        if paragraphs:
                            content = ' '.join(clean_text(p.get_text(strip=True)) for p in paragraphs if len(clean_text(p.get_text(strip=True))) > 50)
                
                if not content or len(content) < 100 or "404" in content.lower() or "page not found" in content.lower():
                    log_error(f"Invalid Content or 404 for {article_url}")
                    report['failures'].append(f"{site['name']}: Invalid content for {article_url}")
                    continue
                
                log_info(f"Content len: {len(content)}")

                publish_date = extract_date(article_soup, site['date_selector'])
                if not is_valid_date(publish_date, scrape_date):
                    log_warning(f"Skipping old article: {article_url} (Date: {publish_date})")
                    continue

                image = extract_image(article_soup, site['image_selector'])
                lang = detect_language(title + ' ' + content, site['name'])
                log_info(f"Detected language: {lang}")

                summarized_title, is_title_valid = summarize_text(title, is_title=True, source_lang=lang)
                if not is_title_valid:
                    log_warning(f"Invalid title summary: {summarized_title[:50]}... (Words: {count_words(summarized_title)}, Similarity: {check_title_similarity(summarized_title, content)})")
                    sentences = content.split('. ')
                    relevant_sentences = [s for s in sentences if len(s.split()) > 3 and not any(x in s.lower() for x in ['advertisement', 'sponsored', 'read more'])]
                    summarized_title = clean_text(' '.join(relevant_sentences[0].split()[:15]) if relevant_sentences else ' '.join(title.split()[:15]))
                    if count_words(summarized_title) < 10:
                        extra_words = relevant_sentences[1].split()[:15-count_words(summarized_title)] if len(relevant_sentences) > 1 else title.split()[:15-count_words(summarized_title)]
                        summarized_title = clean_text(summarized_title + ' ' + ' '.join(extra_words))
                    is_title_valid = count_words(summarized_title) >= 10 and count_words(summarized_title) <= 15
                log_info(f"Summarized title: {summarized_title[:100]} (Words: {count_words(summarized_title)})")

                summary, is_summary_valid = summarize_text(content, is_title=False, source_lang=lang)
                if not is_summary_valid:
                    log_warning(f"Summary issue: {summary[:50]}... (Words: {count_words(summary)}, Repetitive: {detect_repetition(summary)}, Ends with period: {summary.endswith('.')})")
                    sentences = content.split('. ')
                    relevant_sentences = [s for s in sentences if len(s.split()) > 3 and not any(x in s.lower() for x in ['advertisement', 'sponsored', 'read more'])]
                    summary = ' '.join(relevant_sentences[:3])[:500]
                    summary_words = summary.split()
                    if len(summary_words) > 80:
                        summary = ' '.join(summary_words[:80])
                    elif len(summary_words) < 60:
                        extra_sentences = relevant_sentences[3:5]
                        if extra_sentences:
                            summary = f"{summary} {' '.join(extra_sentences)}"[:500]
                    summary += '.'
                    is_summary_valid = count_words(summary) >= 60 and count_words(summary) <= 80 and not detect_repetition(summary)
                    log_info(f"Corrected summary (Words: {len(summary.split())})")
                
                log_info(f"Summary: {summary[:100]} (Words: {count_words(summary)})")

                if not is_title_valid or not is_summary_valid:
                    log_error(f"Skipping article due to invalid title or summary: {article_url}")
                    report['failures'].append(f"{site['name']}: Invalid title/summary for {article_url}")
                    continue

                if is_content_duplicate(summary):
                    log_info(f"Skipping duplicate article: {article_url} (Summary: {summary[:50]}...)")
                    report['skipped'].append(f"{site['name']}: Skipped duplicate {article_url}")
                    continue

                title_translations = {}
                summary_translations = {}
                target_languages = ['hi', 'en', 'mr']  # Reduced languages
                all_translations_valid = True
                for target_lang in target_languages:
                    translated_title = summarized_title if lang == target_lang else translate_text(summarized_title, target_lang, lang)
                    translated_summary = summary if lang == target_lang else translate_text(summary, target_lang, lang)
                    title_translations[target_lang] = translated_title
                    summary_translations[target_lang] = translated_summary
                    if not is_valid_translation(translated_title, target_lang) or not is_valid_translation(translated_summary, target_lang):
                        all_translations_valid = False
                        log_warning(f"Invalid translation for {target_lang}: Title: {translated_title[:50]}..., Summary: {translated_summary[:50]}...")
                
                if not all_translations_valid:
                    log_error(f"Skipping article due to invalid translations: {article_url}")
                    report['failures'].append(f"{site['name']}: Invalid translations for {article_url}")
                    continue

                log_info(f"📝 Title translations: {title_translations}")
                log_info(f"📝 Summary translations: {summary_translations}")

                state, district = extract_state_district(content)
                category = detect_category(title, content)

                try:
                    article_data = {
                        "original_title": title,
                        "summarized_title": summarized_title,
                        "title_translations": title_translations,
                        "content": content,
                        "language": lang,
                        "summary": summary,
                        "translations": summary_translations,
                        "image": image,
                        "location": extract_location(content),
                        "state": state,
                        "district": district,
                        "category": category,
                        "categories": [category, "all", "trending"],
                        "source": site['name'],
                        "publish_date": publish_date,
                        "timestamp": firestore.SERVER_TIMESTAMP,
                        "url": article_url,
                        "status": "approved",
                        "likes": [],
                        "dislikes": [],
                        "views": 0,
                        "shares": 0,
                        "comments": []
                    }
                    db.collection("news").document().set(article_data, timeout=600)
                    if similarity_model:
                        summary_embedding = similarity_model.encode(summary, convert_to_tensor=True)
                        article_cache[article_url] = {
                            'publish_date': publish_date,
                            'title': title,
                            'summary': summary,
                            'summary_embedding': summary_embedding
                        }
                    else:
                        article_cache[article_url] = {
                            'publish_date': publish_date,
                            'title': title,
                            'summary': summary
                        }
                    report['successes'].append(f"{site['name']}: {summarized_title[:50]}... (Date: {publish_date})")
                    log_info(f"✅ ✅ Saved: {summarized_title[:50]}... from {site['name']} (Date: {publish_date})")
                except Exception as e:
                    log_error(f"Firebase Save Error: {e}")
                    report['failures'].append(f"{site['name']}: Firebase save failed for {article_url}")
                
                time.sleep(random.uniform(3, 7))
            
            except Exception as e:
                log_error(f"Article Error in {site['name']} ({article_url}): {e}")
                report['failures'].append(f"{site['name']}: Error processing {article_url}")
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(article_cache, f)
        except Exception as e:
            log_warning(f"Cache Save Error: {e}")
    
    log_info("=== Scraping Report ===")
    log_info(f"Total Sites Processed: {len(news_sites)}")
    log_info(f"Successful Articles: {len(report['successes'])}")
    for success in report['successes']:
        log_info(f"Success: {success}")
    log_info(f"Skipped Duplicates: {len(report['skipped'])}")
    for skipped in report['skipped']:
        log_info(f"Skipped: {skipped}")
    log_info(f"Failures: {len(report['failures'])}")
    for failure in report['failures']:
        log_info(f"Failure: {failure}")
    log_info("======================")

if __name__ == "__main__":
    scrape_and_save()
