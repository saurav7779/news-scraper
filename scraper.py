import requests
import json
import random
import time
import re
import os
from bs4 import BeautifulSoup
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from deep_translator import GoogleTranslator, MyMemoryTranslator, LibreTranslator
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

# Logging setup with emojis for professional debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom logging with emojis
def log_error(msg):
    logger.error(f"🚨 🚨 {msg}")

def log_warning(msg):
    logger.warning(f"⚠️ {msg}")

def log_info(msg):
    logger.info(f"✅ {msg}")

def log_debug(msg):
    logger.debug(f"🔍 {msg}")

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

# Initialize summarizer with BART-large-CNN
summarizer = None
tokenizer = None
try:
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=-1)
except Exception as e:
    log_warning(f"Model Error: {e}. No summarization.")
    summarizer = None

# Initialize sentence transformer for semantic similarity
try:
    similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    log_warning(f"Sentence Transformer Error: {e}. Similarity checks disabled.")
    similarity_model = None

# Cache setup for article URLs and content embeddings
cache_file = 'article_cache.pkl'
article_cache = {}
translation_cache = {}
try:
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
            article_cache = cache_data.get('articles', {})
            translation_cache = cache_data.get('translations', {})
except Exception as e:
    log_warning(f"Cache Load Error: {e}")

# Multiple translation services for fallback
TRANSLATION_SERVICES = [
    ('google', GoogleTranslator),
    ('mymemory', MyMemoryTranslator),
    ('libre', LibreTranslator)
]

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
    # Remove summarization artifacts
    text = re.sub(r'(summarize|generate|in|concise|factual|manner|key points|excluding|opinions|minor details|words|80-150|15-30).*?:', '', text, flags=re.IGNORECASE)
    # Clean unwanted patterns
    text = re.sub(r'\s+', ' ', text.replace('OPINION |', '').strip())
    # Remove special characters that cause issues
    text = re.sub(r'[^\w\s.,!?;:\-()\'\"]+', ' ', text)
    return text.strip()

def detect_language(text, site_name):
    site_lang_map = {
        'Aaj Tak': 'hi', 'ABP News': 'hi', 'GNTTV': 'hi', 'Times of India': 'en',
        'Marathi Abplive': 'mr', 'Zee Bihar Jharkhand': 'hi', 'First Bihar': 'hi',
        'Zee News': 'hi', 'CNN': 'en', 'The Guardian': 'en', 'BBC': 'en',
        'India TV': 'hi', 'News 18': 'hi', 'Amar Ujala': 'hi', 'The Hindu': 'en',
        'Hindustan': 'hi', 'Jagran': 'hi', 'Prabhat Khabar': 'hi',
        'Top Bihar News': 'hi', 'Main Media': 'hi', 'Bihar Links': 'hi',
        'NBT': 'hi', 'UP Tak': 'hi', 'UP 18 News': 'hi', 'Khabar Up': 'hi',
        'Dainik Bhaskar': 'hi', 'NDTV': 'en', 'Bharat Samachar': 'hi',
        'The Financial Express': 'en', 'Khabar Lahariya': 'hi'
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
    return 'hi'  # Default to Hindi for Indian sites

def get_translation_cache_key(text, target_lang, source_lang):
    """Generate cache key for translation"""
    text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
    return f"{source_lang}_{target_lang}_{text_hash}"

def is_valid_translation(text, target_lang):
    if not text or len(text.strip()) < 3:
        return False
    invalid_patterns = [
        'MYMEMORY WARNING', 'Unable to translate', 'Translation failed',
        'API limit exceeded', 'Service unavailable', 'Connection error',
        'Request failed', 'Too many requests'
    ]
    return not any(pattern.lower() in text.lower() for pattern in invalid_patterns)

def translate_text_fallback(text, target_lang='hi', source_lang='auto', max_retries=3):
    """Enhanced translation with multiple services and caching"""
    if not text or len(text.strip()) < 3:
        return text
    
    # Normalize languages
    if source_lang == 'auto':
        source_lang = detect_language(text, "")
    
    if source_lang == target_lang:
        return text
    
    # Check cache first
    cache_key = get_translation_cache_key(text, target_lang, source_lang)
    if cache_key in translation_cache:
        log_debug(f"Using cached translation for: {text[:30]}...")
        return translation_cache[cache_key]
    
    # Try different translation services
    for service_name, service_class in TRANSLATION_SERVICES:
        for attempt in range(max_retries):
            try:
                log_info(f"Trying {service_name} translation (attempt {attempt+1})")
                time.sleep(random.uniform(2, 4))  # Rate limiting
                
                if service_name == 'google':
                    translator = service_class(source=source_lang, target=target_lang)
                elif service_name == 'mymemory':
                    translator = service_class(source=source_lang, target=target_lang)
                elif service_name == 'libre':
                    # LibreTranslator might need different parameters
                    try:
                        translator = service_class(source=source_lang, target=target_lang, base_url="https://libretranslate.de")
                    except:
                        continue
                else:
                    continue
                
                translated = translator.translate(text)
                
                if (translated and 
                    len(translated.strip()) > 2 and 
                    translated.strip().lower() != text.strip().lower() and
                    is_valid_translation(translated, target_lang)):
                    
                    # Clean the translation
                    translated = clean_text(translated)
                    
                    # Cache successful translation
                    translation_cache[cache_key] = translated
                    
                    log_info(f"✅ Successfully translated using {service_name}: {translated[:50]}...")
                    return translated
                
                log_warning(f"Invalid translation from {service_name}: '{translated}'")
                
            except Exception as e:
                log_warning(f"{service_name} translation error (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(random.uniform(3, 6))
    
    # If all translation services fail, return original text
    log_error(f"All translation services failed for {target_lang}, returning original")
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
        return article_date <= scrape_date and article_date >= scrape_date - timedelta(days=7)
    except:
        return True

def is_valid_article_url(url, site_name, site):
    if not url or "404" in url.lower() or "page not found" in url.lower():
        return False
    
    exclude_patterns = ['/video/', '/live/', '/category/', '/photos/', '/gallery/', '/login', '/web-stories', '/photo-gallery', '/live-tv', '/tags/', '/bhojpuri-cinema']
    
    # Site-specific URL cleaning
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
        
        # Check cached articles
        for cached_url, cached_data in article_cache.items():
            if 'summary_embedding' in cached_data:
                cached_embedding = cached_data['summary_embedding']
                similarity = util.cos_sim(summary_embedding, cached_embedding).item()
                if similarity > 0.85:
                    log_info(f"Duplicate content detected: {summary[:50]}... similar to cached article (Similarity: {similarity:.2f})")
                    return True
        
        # Check recent articles in database
        try:
            query = db.collection("news").limit(50).get()
            for doc in query:
                doc_summary = doc.to_dict().get('summary', '')
                if doc_summary:
                    doc_embedding = similarity_model.encode(doc_summary, convert_to_tensor=True)
                    similarity = util.cos_sim(summary_embedding, doc_embedding).item()
                    if similarity > 0.85:
                        log_info(f"Duplicate content detected: {summary[:50]}... similar to existing article (Similarity: {similarity:.2f})")
                        return True
        except Exception as e:
            log_warning(f"Firebase Duplicate Check Error: {e}")
        
        return False
    except Exception as e:
        log_warning(f"Content Duplicate Check Error: {e}")
        return False

def count_words(text):
    if not text:
        return 0
    return len(text.split())

def detect_repetition(summary):
    if not summary:
        return False
    
    sentences = summary.split('. ')
    if len(sentences) < 2:
        return False
    
    # Check for sentence repetition
    for i, sentence1 in enumerate(sentences):
        for j, sentence2 in enumerate(sentences[i+1:], i+1):
            if len(sentence1.split()) > 3 and len(sentence2.split()) > 3:
                words1 = set(sentence1.lower().split())
                words2 = set(sentence2.lower().split())
                overlap = len(words1.intersection(words2)) / len(words1.union(words2))
                if overlap > 0.7:  # 70% word overlap indicates repetition
                    return True
    
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
        return similarity > 0.25  # Relaxed threshold
    except:
        return True

def create_comprehensive_summary(text, is_title=False):
    """Create comprehensive 80-word summary that covers complete story"""
    if not text:
        return ""
    
    # Clean and split into sentences
    sentences = text.split('. ')
    relevant_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if (len(sentence.split()) > 4 and 
            not any(x in sentence.lower() for x in [
                'advertisement', 'sponsored', 'read more', 'click here',
                'follow us', 'share this', 'comment now', 'latest news',
                'trending now', 'recommended', 'related news', 'copyright',
                'privacy policy', 'terms of use', 'cookie policy', 'subscribe'
            ])):
            relevant_sentences.append(sentence)
    
    if not relevant_sentences:
        relevant_sentences = [s.strip() for s in sentences[:10] if len(s.strip()) > 10]
    
    if is_title:
        # For titles, extract key information in 20-25 words
        if relevant_sentences:
            title_words = relevant_sentences[0].split()[:25]
            return clean_text(' '.join(title_words))
        return clean_text(' '.join(text.split()[:25]))
    
    # For summaries, create exactly 80-word comprehensive summary
    target_words = 80
    
    # Extract key information from different parts of article
    beginning_sentences = relevant_sentences[:3]  # Lead/intro
    middle_sentences = relevant_sentences[3:8] if len(relevant_sentences) > 6 else []
    end_sentences = relevant_sentences[-2:] if len(relevant_sentences) > 5 else []
    
    # Combine key sentences to get complete picture
    key_sentences = beginning_sentences + middle_sentences + end_sentences
    
    # Remove duplicates while preserving order
    seen = set()
    unique_sentences = []
    for sentence in key_sentences:
        if sentence not in seen:
            unique_sentences.append(sentence)
            seen.add(sentence)
    
    # Build 80-word summary
    summary_words = []
    current_count = 0
    
    # Prioritize beginning sentences for context
    for sentence in unique_sentences:
        sentence_words = sentence.split()
        
        if current_count + len(sentence_words) <= target_words:
            summary_words.extend(sentence_words)
            current_count += len(sentence_words)
        else:
            # Add partial sentence to reach exactly 80 words
            remaining_words = target_words - current_count
            if remaining_words > 5:  # Only add if meaningful words remain
                summary_words.extend(sentence_words[:remaining_words])
                current_count = target_words
            break
    
    # If we haven't reached 80 words, add more content
    if current_count < target_words and len(relevant_sentences) > len(unique_sentences):
        remaining_sentences = [s for s in relevant_sentences if s not in seen]
        for sentence in remaining_sentences:
            sentence_words = sentence.split()
            remaining_words = target_words - current_count
            
            if remaining_words <= 0:
                break
                
            if len(sentence_words) <= remaining_words:
                summary_words.extend(sentence_words)
                current_count += len(sentence_words)
            else:
                summary_words.extend(sentence_words[:remaining_words])
                current_count = target_words
                break
    
    # Ensure we have exactly 80 words (or close to it)
    if len(summary_words) > target_words:
        summary_words = summary_words[:target_words]
    elif len(summary_words) < target_words - 5:  # If significantly under 80
        # Add more words from original text
        all_words = text.split()
        word_position = 0
        for word in all_words:
            if word not in summary_words and len(summary_words) < target_words:
                summary_words.append(word)
    
    summary = ' '.join(summary_words)
    
    # Ensure proper ending
    if summary and not summary.endswith(('.', '!', '?')):
        summary += '.'
    
    final_summary = clean_text(summary)
    log_info(f"Created comprehensive summary: {len(final_summary.split())} words")
    return final_summary

def summarize_text(text, is_title=False, source_lang='hi'):
    """Enhanced summarization creating exactly 80-word comprehensive summaries"""
    if not text or len(text.strip()) < 20:
        return ""
    
    # For titles, use simplified approach
    if is_title:
        return create_comprehensive_summary(text, is_title=True)
    
    # For summaries, always use comprehensive approach regardless of language
    # This ensures complete story understanding in exactly 80 words
    log_info(f"Creating comprehensive 80-word summary for {source_lang} content")
    
    # Enhanced text preprocessing for better summarization
    processed_text = text
    
    # Remove repeated phrases and clean up
    processed_text = re.sub(r'\b(\w+)\s+\1\b', r'\1', processed_text)  # Remove word repetitions
    processed_text = re.sub(r'\.{2,}', '.', processed_text)  # Fix multiple dots
    processed_text = re.sub(r'\s{2,}', ' ', processed_text)  # Fix multiple spaces
    
    # Try BART only for English and if available, otherwise use comprehensive fallback
    if source_lang == 'en' and summarizer:
        try:
            # Prepare text for BART (limit input length)
            input_words = processed_text.split()
            if len(input_words) > 800:
                processed_text = ' '.join(input_words[:800])
            
            input_tokens = len(tokenizer.encode(processed_text, truncation=True))
            log_debug(f"BART processing {input_tokens} tokens")
            
            # Generate initial summary with BART
            bart_summary = summarizer(
                processed_text,
                max_length=120,  # Allow some flexibility for post-processing
                min_length=70,
                do_sample=True,
                top_k=50,
                top_p=0.9,
                num_beams=4,
                length_penalty=1.0,
                early_stopping=True,
                truncation=True
            )[0]['summary_text']
            
            bart_summary = clean_text(bart_summary)
            bart_word_count = count_words(bart_summary)
            
            # Check if BART summary is good quality
            if (bart_word_count >= 60 and bart_word_count <= 120 and
                not any(x in bart_summary.lower() for x in ['summarize', 'generate', 'words']) and
                not detect_repetition(bart_summary)):
                
                # Adjust BART summary to exactly 80 words
                bart_words = bart_summary.split()
                if len(bart_words) > 80:
                    # Trim to 80 words while keeping meaningful ending
                    trimmed_summary = ' '.join(bart_words[:80])
                    if not trimmed_summary.endswith(('.', '!', '?')):
                        # Try to find a good stopping point
                        for i in range(79, 70, -1):
                            if bart_words[i].endswith(('.', '!', '?')):
                                trimmed_summary = ' '.join(bart_words[:i+1])
                                break
                        else:
                            trimmed_summary += '.'
                    
                    log_info(f"BART summary adjusted to 80 words")
                    return clean_text(trimmed_summary)
                
                elif len(bart_words) < 75:
                    # Enhance BART summary with additional context to reach 80 words
                    remaining_words = 80 - len(bart_words)
                    
                    # Get additional context from original text
                    original_sentences = processed_text.split('. ')
                    additional_info = []
                    
                    for sentence in original_sentences:
                        sentence_words = sentence.split()
                        # Find sentences not already covered in summary
                        if len(sentence_words) > 5 and not any(word in bart_summary.lower() for word in sentence.lower().split()[:3]):
                            additional_info.extend(sentence_words[:remaining_words])
                            remaining_words -= len(sentence_words[:remaining_words])
                            if remaining_words <= 0:
                                break
                    
                    if additional_info:
                        enhanced_summary = bart_summary + ' ' + ' '.join(additional_info[:remaining_words])
                        if not enhanced_summary.endswith(('.', '!', '?')):
                            enhanced_summary += '.'
                        
                        log_info(f"BART summary enhanced to 80 words")
                        return clean_text(enhanced_summary)
                
                # If BART summary is already around 80 words, use as is
                if not bart_summary.endswith(('.', '!', '?')):
                    bart_summary += '.'
                
                log_info(f"BART summary used: {len(bart_words)} words")
                return clean_text(bart_summary)
            
            else:
                log_warning(f"BART summary quality issues, using comprehensive fallback")
                
        except Exception as e:
            log_warning(f"BART error: {e}, using comprehensive fallback")
    
    # Use comprehensive summary approach for all cases
    # This ensures exactly 80 words with complete story coverage
    return create_comprehensive_summary(processed_text, is_title=False)

def extract_location(content):
    locations = {
        'India': ['india', 'delhi', 'new delhi', 'mumbai', 'bangalore', 'chennai', 'kolkata', 'hyderabad', 'pune', 'ahmedabad', 'bihar', 'patna'],
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
    return 'India'  # Default to India for Indian news sites

def detect_category(title, content):
    categories = {
        'defence': ['defence', 'military', 'fighter', 'jet', 'aircraft', 'tejas', 'hal', 'aerospace', 'safran', 'missile', 'army', 'navy', 'air force'],
        'international': ['international', 'global', 'foreign', 'diplomacy', 'visa', 'immigration', 'united nations', 'summit', 'trade agreement'],
        'politics': ['election', 'government', 'minister', 'pm', 'cm', 'parliament', 'congress', 'bjp', 'political', 'policy', 'law', 'bill', 'act', 'prime minister', 'president'],
        'sports': ['cricket', 'football', 'match', 'sport', 'player', 'tournament', 'olympics', 'medal', 'coach', 'team', 'ipl', 'world cup', 'championship'],
        'business': ['market', 'stock', 'business', 'economy', 'company', 'industry', 'trade', 'commerce', 'gst', 'tax', 'investment', 'finance', 'bank', 'rupee', 'dollar'],
        'entertainment': ['movie', 'film', 'actor', 'bollywood', 'hollywood', 'celebrity', 'music', 'song', 'album', 'director', 'entertainment'],
        'crime': ['crime', 'arrest', 'police', 'murder', 'theft', 'robbery', 'accident', 'investigation', 'court', 'judge', 'lawyer', 'case', 'illegal', 'violation'],
        'technology': ['technology', 'tech', 'computer', 'mobile', 'app', 'software', 'internet', 'digital', 'ai', 'artificial intelligence', 'smartphone'],
        'health': ['health', 'medical', 'hospital', 'doctor', 'disease', 'medicine', 'vaccine', 'covid', 'corona', 'healthcare', 'treatment', 'patient'],
        'education': ['education', 'school', 'college', 'university', 'student', 'teacher', 'exam', 'result', 'board', 'degree']
    }
    
    text = (title + ' ' + content).lower()
    category_scores = {cat: 0 for cat in categories}
    
    for cat, keywords in categories.items():
        for keyword in keywords:
            if keyword in text:
                category_scores[cat] += 1
    
    max_score = max(category_scores.values())
    if max_score == 0:
        return 'general'
    
    return max(category_scores, key=category_scores.get)

def fetch_page(url, retries=3):
    """Enhanced page fetching with better error handling"""
    backoff = 5
    
    # Try requests first
    for attempt in range(retries):
        try:
            log_info(f"Fetching {url} (Attempt {attempt+1}/{retries})")
            response = requests.get(url, headers=get_random_headers(), timeout=30)
            response.raise_for_status()
            if response.text and len(response.text) > 1000:
                return response.text
        except Exception as e:
            log_warning(f"Requests Error: {e}")
            if attempt < retries - 1:
                time.sleep(random.uniform(backoff, backoff * 2))
                backoff *= 2
    
    # Fallback to Playwright
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

def save_cache():
    """Save both article and translation caches"""
    try:
        cache_data = {
            'articles': article_cache,
            'translations': translation_cache
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        log_info(f"Cache saved with {len(article_cache)} articles and {len(translation_cache)} translations")
    except Exception as e:
        log_warning(f"Cache Save Error: {e}")

def scrape_and_save():
    scrape_date = date.today()
    report = defaultdict(list)
    
    for site in news_sites:
        log_info(f"🔄 Scraping {site['name']}...")
        time.sleep(random.uniform(10, 20))
        
        html = fetch_page(site['url'])
        if not html:
            log_error(f"Failed to fetch {site['name']}")
            report['failures'].append(f"{site['name']}: Failed to fetch main page")
            continue
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # Get unique article links
        article_links = list(set(link.get('href', '') for link in soup.select(site['article_link_selector'])[:15]))
        
        # Pre-filter and sort articles by date
        articles_with_dates = []
        for article_url in article_links:
            if not article_url:
                continue
                
            if not article_url.startswith('http'):
                article_url = site['url'].rstrip('/') + '/' + article_url.lstrip('/')
            
            if not is_valid_article_url(article_url, site['name'], site) or is_article_duplicate(article_url):
                continue
            
            # Quick date extraction
            article_html = fetch_page(article_url)
            if not article_html:
                continue
                
            article_soup = BeautifulSoup(article_html, 'html.parser')
            publish_date = extract_date(article_soup, site['date_selector'])
            
            if not is_valid_date(publish_date, scrape_date):
                continue
            
            articles_with_dates.append({
                'url': article_url,
                'publish_date': publish_date
            })
        
        # Sort by date (newest first)
        try:
            articles_with_dates.sort(key=lambda x: datetime.strptime(x['publish_date'], '%Y-%m-%d'), reverse=True)
        except Exception as e:
            log_warning(f"Sorting error: {e}")
        
        # Process top 5 latest articles
        selected_articles = articles_with_dates[:5]
        log_info(f"Selected {len(selected_articles)} latest articles from {site['name']}")
        
        if not selected_articles:
            log_warning(f"No valid latest articles found for {site['name']}")
            report['failures'].append(f"{site['name']}: No valid articles found")
            continue
        
        for article_data in selected_articles:
            article_url = article_data['url']
            try:
                time.sleep(random.uniform(3, 7))
                
                article_html = fetch_page(article_url)
                if not article_html:
                    log_error(f"Failed to fetch article: {article_url}")
                    report['failures'].append(f"{site['name']}: Failed to fetch {article_url}")
                    continue
                
                article_soup = BeautifulSoup(article_html, 'html.parser')
                
                # Extract title
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

                # Extract content with multiple fallback strategies
                content = ''
                content_selectors = site['content_selector'].split(',') + [
                    'p', '.content p', '.article p', '.story-content p', 
                    '.articlebody p', '.article-text p', '.post-content p'
                ]
                
                for selector in content_selectors:
                    selector = selector.strip()
                    content_elements = article_soup.select(selector)
                    if content_elements:
                        filtered_content = []
                        for elem in content_elements:
                            elem_text = clean_text(elem.get_text(strip=True))
                        if (len(elem_text) > 30 and 
                            not any(x in elem_text.lower() for x in [
                                'advertisement', 'sponsored', 'read more', 'toi tech desk',
                                'click here', 'follow us', 'share this', 'comment now',
                                'latest news', 'trending now', 'recommended', 'related news',
                                'copyright', 'privacy policy', 'terms of use', 'cookie policy',
                                'subscribe', 'newsletter', 'download app'
                            ]) and
                            not re.search(r'₹\d+,\d+', elem_text) and
                            not re.search(r'^\d+\s*\.?\s*', elem_text)):
                            filtered_content.append(elem_text)
                        
                        if filtered_content and len(' '.join(filtered_content)) > 200:
                            content = ' '.join(filtered_content)
                            log_info(f"Content found using selector: {selector}")
                            break
                
                # Additional fallback for content extraction
                if not content or len(content) < 200:
                    main_selectors = [
                        'div[class*="content"]', 'div[class*="article"]', 
                        'div[class*="story"]', 'div[class*="body"]',
                        'article', 'main'
                    ]
                    for main_sel in main_selectors:
                        main_content = article_soup.select_one(main_sel)
                        if main_content:
                            paragraphs = main_content.find_all('p')
                            if paragraphs:
                                para_texts = []
                                for p in paragraphs:
                                    p_text = clean_text(p.get_text(strip=True))
                                    if len(p_text) > 30:
                                        para_texts.append(p_text)
                                if para_texts and len(' '.join(para_texts)) > 200:
                                    content = ' '.join(para_texts)
                                    break
                
                if not content or len(content) < 200:
                    log_error(f"Insufficient content for {article_url}")
                    report['failures'].append(f"{site['name']}: Insufficient content for {article_url}")
                    continue
                
                log_info(f"Content extracted: {len(content)} characters")

                # Extract other metadata
                publish_date = extract_date(article_soup, site['date_selector'])
                if not is_valid_date(publish_date, scrape_date):
                    log_warning(f"Skipping old article: {article_url} (Date: {publish_date})")
                    continue

                image = extract_image(article_soup, site['image_selector'])
                lang = detect_language(title + ' ' + content, site['name'])
                log_info(f"Detected language: {lang}")

                # Generate summaries with enhanced error handling
                log_info("Generating title summary...")
                summarized_title = summarize_text(title, is_title=True, source_lang=lang)
                
                # Validate title summary
                title_word_count = count_words(summarized_title)
                if (title_word_count < 10 or title_word_count > 35 or
                    any(x in summarized_title.lower() for x in ['summarize', '15-30', 'words']) or
                    not check_title_similarity(summarized_title, content)):
                    
                    log_warning(f"Title summary failed validation, creating fallback")
                    summarized_title = create_fallback_summary(title, is_title=True, target_length=25)
                
                log_info(f"Title summary: {summarized_title} (Words: {count_words(summarized_title)})")

                log_info("Generating content summary...")
                summary = summarize_text(content, is_title=False, source_lang=lang)
                
                # Validate content summary - must be exactly 80 words
                summary_word_count = count_words(summary)
                if (summary_word_count < 75 or summary_word_count > 85 or 
                    detect_repetition(summary) or
                    any(x in summary.lower() for x in ['summarize', '80-150', 'words'])):
                    
                    log_warning(f"Content summary failed validation (Words: {summary_word_count}), recreating...")
                    summary = create_comprehensive_summary(content, is_title=False)
                    
                    # Double-check the recreated summary
                    final_word_count = count_words(summary)
                    if final_word_count != 80:
                        # Force exactly 80 words
                        summary_words = summary.split()
                        if len(summary_words) > 80:
                            summary = ' '.join(summary_words[:80])
                        elif len(summary_words) < 80:
                            # Add more content to reach 80 words
                            content_words = content.split()
                            additional_words = []
                            for word in content_words:
                                if word not in summary and len(summary.split()) + len(additional_words) < 80:
                                    additional_words.append(word)
                                if len(summary.split()) + len(additional_words) >= 80:
                                    break
                            summary = summary + ' ' + ' '.join(additional_words[:80-len(summary.split())])
                        
                        # Ensure proper ending
                        if not summary.endswith(('.', '!', '?')):
                            summary += '.'
                
                final_word_count = count_words(summary)
                log_info(f"Final summary: {summary[:100]}... (Words: {final_word_count})")
                
                # Ensure we have exactly 80 words before proceeding
                if final_word_count < 75 or final_word_count > 85:
                    log_warning(f"Summary still not 80 words ({final_word_count}), forcing correction...")
                    words = summary.split()
                    if len(words) != 80:
                        if len(words) > 80:
                            summary = ' '.join(words[:80]) + '.'
                        else:
                            # Get more content from the article
                            remaining_content = content.replace(summary, '').split()
                            needed_words = 80 - len(words)
                            additional = remaining_content[:needed_words]
                            summary = summary.rstrip('.') + ' ' + ' '.join(additional) + '.'
                    
                    log_info(f"Corrected to exactly 80 words: {count_words(summary)} words")

                # Check for duplicate content
                if is_content_duplicate(summary):
                    log_info(f"Skipping duplicate content: {article_url}")
                    report['skipped'].append(f"{site['name']}: Duplicate content {article_url}")
                    continue

                # Enhanced translation with better error handling
                log_info("Starting translations...")
                title_translations = {}
                summary_translations = {}
                target_languages = ['hi', 'mr', 'ta', 'te', 'en']
                
                # Add original language versions
                title_translations[lang] = summarized_title
                summary_translations[lang] = summary
                
                for target_lang in target_languages:
                    if target_lang == lang:
                        continue
                        
                    try:
                        log_info(f"Translating to {target_lang}...")
                        
                        # Translate title
                        translated_title = translate_text_fallback(
                            summarized_title, target_lang, lang
                        )
                        title_translations[target_lang] = translated_title
                        
                        # Translate summary  
                        translated_summary = translate_text_fallback(
                            summary, target_lang, lang
                        )
                        summary_translations[target_lang] = translated_summary
                        
                        log_info(f"Completed {target_lang} translation")
                        
                    except Exception as e:
                        log_warning(f"Translation error for {target_lang}: {e}")
                        # Fallback to original text if translation fails
                        title_translations[target_lang] = summarized_title
                        summary_translations[target_lang] = summary

                # Log translation results
                log_info(f"Title translations completed: {list(title_translations.keys())}")
                log_info(f"Summary translations completed: {list(summary_translations.keys())}")

                # Prepare article data
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
                        "category": detect_category(title, content),
                        "categories": [detect_category(title, content), "all", "trending"],
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
                    
                    # Save to Firebase
                    db.collection("news").document().set(article_data, timeout=600)
                    
                    # Update cache
                    cache_data = {
                        'publish_date': publish_date,
                        'title': title,
                        'summary': summary
                    }
                    
                    if similarity_model:
                        summary_embedding = similarity_model.encode(summary, convert_to_tensor=True)
                        cache_data['summary_embedding'] = summary_embedding
                    
                    article_cache[article_url] = cache_data
                    
                    # Save caches periodically
                    save_cache()
                    
                    report['successes'].append(f"{site['name']}: {summarized_title[:50]}... (Date: {publish_date})")
                    log_info(f"Successfully saved: {summarized_title[:50]}... from {site['name']}")
                    
                except Exception as e:
                    log_error(f"Firebase Save Error: {e}")
                    report['failures'].append(f"{site['name']}: Firebase save failed for {article_url}")
                
                # Rate limiting between articles
                time.sleep(random.uniform(3, 7))
                
            except Exception as e:
                log_error(f"Article processing error in {site['name']} ({article_url}): {e}")
                report['failures'].append(f"{site['name']}: Processing error for {article_url}")
        
        # Rate limiting between sites
        time.sleep(random.uniform(10, 20))
    
    # Final cache save
    save_cache()
    
    # Generate final report
    log_info("=== SCRAPING COMPLETED ===")
    log_info(f"Total Sites Processed: {len(news_sites)}")
    log_info(f"Successful Articles: {len(report['successes'])}")
    log_info(f"Skipped Duplicates: {len(report['skipped'])}")  
    log_info(f"Failures: {len(report['failures'])}")
    
    for success in report['successes']:
        log_info(f"SUCCESS: {success}")
    
    for failure in report['failures']:
        log_info(f"FAILURE: {failure}")
        
    for skipped in report['skipped']:
        log_info(f"SKIPPED: {skipped}")
    
    log_info("=========================")

if __name__ == "__main__":
    scrape_and_save()
