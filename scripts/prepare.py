import json
import re
import html
import unicodedata
import argparse
from pathlib import Path
import ftfy
import emoji
import contractions
from bs4 import BeautifulSoup
from langdetect import detect
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import spacy
import string

class TextPreprocessor:
   def __init__(self, config=None):
       self.config = config or {}
       self.stemmer = PorterStemmer()
       self.lemmatizer = WordNetLemmatizer()
       try:
           self.nlp = spacy.load("en_core_web_sm")
       except:
           self.nlp = None
       
   def clean_html(self, text):
       if not text:
           return text
       soup = BeautifulSoup(text, 'html.parser')
       return soup.get_text()
   
   def fix_encoding(self, text):
       if not text:
           return text
       return ftfy.fix_text(text)
   
   def normalize_unicode(self, text):
       if not text:
           return text
       return unicodedata.normalize('NFKC', text)
   
   def normalize_whitespace(self, text):
       if not text:
           return text
       text = re.sub(r'\s+', ' ', text)
       text = text.strip()
       return text
   
   def normalize_quotes(self, text):
       if not text:
           return text
       quote_map = {
           '"': '"', '"': '"',
           ''': "'", ''': "'",
           '`': "'", '´': "'",
           '«': '"', '»': '"'
       }
       for old, new in quote_map.items():
           text = text.replace(old, new)
       return text
   
   def normalize_dashes(self, text):
       if not text:
           return text
       dash_map = {
           '–': '-',
           '—': '-',
           '―': '-',
           '−': '-'
       }
       for old, new in dash_map.items():
           text = text.replace(old, new)
       return text
   
   def remove_extra_punctuation(self, text):
       if not text:
           return text
       text = re.sub(r'\.{3,}', '...', text)
       text = re.sub(r'!{2,}', '!', text)
       text = re.sub(r'\?{2,}', '?', text)
       text = re.sub(r'-{2,}', '--', text)
       return text
   
   def fix_common_typos(self, text):
       if not text:
           return text
       
       typo_map = {
           r'\bteh\b': 'the',
           r'\band\s+and\b': 'and',
           r'\bthe\s+the\b': 'the',
           r'\bis\s+is\b': 'is',
           r'\bof\s+of\b': 'of',
           r'\bto\s+to\b': 'to',
           r'\bin\s+in\b': 'in',
           r'\bfor\s+for\b': 'for',
           r'\bwith\s+with\b': 'with',
           r'\bon\s+on\b': 'on',
           r'\bat\s+at\b': 'at',
           r'\bbut\s+but\b': 'but',
       }
       
       for pattern, replacement in typo_map.items():
           text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
       
       return text
   
   def remove_urls(self, text):
       if not text:
           return text
       url_pattern = r'https?://[^\s<>"\'{}|\\^`\[\]]+|www\.[^\s<>"\'{}|\\^`\[\]]+'
       return re.sub(url_pattern, '', text)
   
   def normalize_numbers(self, text):
       if not text:
           return text
       text = re.sub(r'\b(\d+),(\d{3})\b', r'\1\2', text)
       text = re.sub(r'(\d+)\s*k\b', r'\1000', text, flags=re.IGNORECASE)
       text = re.sub(r'(\d+)\s*million\b', r'\1000000', text, flags=re.IGNORECASE)
       return text
   
   def expand_contractions(self, text):
       if not text:
           return text
       return contractions.fix(text)
   
   def remove_special_characters(self, text, keep_basic=True):
       if not text:
           return text
       
       if keep_basic:
           text = re.sub(r'[^\w\s.,!?;:()\'"/-]', '', text)
       else:
           text = re.sub(r'[^\w\s]', '', text)
       
       return text
   
   def fix_case_issues(self, text):
       if not text:
           return text
       
       text = re.sub(r'\b[A-Z]{2,}\b', lambda m: m.group().capitalize() if len(m.group()) > 3 else m.group(), text)
       
       sentences = re.split(r'(?<=[.!?])\s+', text)
       fixed_sentences = []
       for sentence in sentences:
           if sentence:
               sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
               fixed_sentences.append(sentence)
       
       return ' '.join(fixed_sentences)
   
   def remove_repeated_characters(self, text):
       if not text:
           return text
       text = re.sub(r'(.)\1{3,}', r'\1\1', text)
       return text
   
   def remove_emojis(self, text):
       if not text:
           return text
       return emoji.demojize(text, delimiters=('', ''))
   
   def demojize_emojis(self, text):
       if not text:
           return text
       return emoji.demojize(text)
   
   def remove_stopwords(self, text, language='english'):
       if not text:
           return text
       try:
           stop_words = set(stopwords.words(language))
           tokens = word_tokenize(text)
           filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
           return ' '.join(filtered_tokens)
       except:
           return text
   
   def stem_text(self, text):
       if not text:
           return text
       tokens = word_tokenize(text)
       stemmed_tokens = [self.stemmer.stem(token) for token in tokens]
       return ' '.join(stemmed_tokens)
   
   def lemmatize_text(self, text):
       if not text:
           return text
       tokens = word_tokenize(text)
       lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
       return ' '.join(lemmatized_tokens)
   
   def remove_punctuation(self, text):
       if not text:
           return text
       return text.translate(str.maketrans('', '', string.punctuation))
   
   def to_lowercase(self, text):
       if not text:
           return text
       return text.lower()
   
   def to_uppercase(self, text):
       if not text:
           return text
       return text.upper()
   
   def detect_language(self, text):
       if not text:
           return None
       try:
           return detect(text)
       except:
           return None
   
   def filter_language(self, text, target_language='en'):
       if not text:
           return text
       detected_lang = self.detect_language(text)
       return text if detected_lang == target_language else ""
   
   def remove_digits(self, text):
       if not text:
           return text
       return re.sub(r'\d+', '', text)
   
   def remove_single_characters(self, text):
       if not text:
           return text
       return re.sub(r'\b\w\b', '', text)
   
   def normalize_accents(self, text):
       if not text:
           return text
       return unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode('utf-8')
   
   def remove_extra_spaces(self, text):
       if not text:
           return text
       return re.sub(r'\s{2,}', ' ', text).strip()
   
   def extract_sentences(self, text):
       if not text:
           return text
       if self.nlp:
           doc = self.nlp(text)
           return ' '.join([sent.text.strip() for sent in doc.sents])
       return text
   
   def remove_short_words(self, text, min_length=2):
       if not text:
           return text
       tokens = text.split()
       filtered_tokens = [word for word in tokens if len(word) >= min_length]
       return ' '.join(filtered_tokens)
   
   def remove_long_words(self, text, max_length=20):
       if not text:
           return text
       tokens = text.split()
       filtered_tokens = [word for word in tokens if len(word) <= max_length]
       return ' '.join(filtered_tokens)
   
   def clean_medical_text(self, text):
       if not text:
           return text
       
       medical_replacements = {
           r'\bmg\b': 'milligrams',
           r'\bml\b': 'milliliters',
           r'\bcc\b': 'cubic centimeters',
           r'\bbid\b': 'twice daily',
           r'\btid\b': 'three times daily',
           r'\bqid\b': 'four times daily',
           r'\bprn\b': 'as needed',
           r'\bpo\b': 'by mouth',
           r'\biv\b': 'intravenous',
           r'\bim\b': 'intramuscular'
       }
       
       for pattern, replacement in medical_replacements.items():
           text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
       
       return text
   
   def process_text(self, text, steps=None):
       if not text or not isinstance(text, str):
           return ""
       
       if steps is None:
           steps = [
               'fix_encoding',
               'clean_html',
               'normalize_unicode',
               'normalize_whitespace',
               'normalize_quotes',
               'normalize_dashes',
               'remove_extra_punctuation',
               'fix_common_typos',
               'expand_contractions',
               'remove_repeated_characters',
               'normalize_whitespace'
           ]
       
       for step in steps:
           if hasattr(self, step):
               text = getattr(self, step)(text)
       
       if self.config.get('medical_domain', False):
           text = self.clean_medical_text(text)
       
       if self.config.get('remove_urls', False):
           text = self.remove_urls(text)
       
       if self.config.get('normalize_numbers', False):
           text = self.normalize_numbers(text)
       
       if self.config.get('remove_special_chars', False):
           text = self.remove_special_characters(text, keep_basic=True)
       
       if self.config.get('fix_case', False):
           text = self.fix_case_issues(text)
       
       if self.config.get('remove_emojis', False):
           text = self.remove_emojis(text)
       
       if self.config.get('demojize_emojis', False):
           text = self.demojize_emojis(text)
       
       if self.config.get('remove_stopwords', False):
           text = self.remove_stopwords(text, self.config.get('stopwords_language', 'english'))
       
       if self.config.get('stem_text', False):
           text = self.stem_text(text)
       
       if self.config.get('lemmatize_text', False):
           text = self.lemmatize_text(text)
       
       if self.config.get('remove_punctuation', False):
           text = self.remove_punctuation(text)
       
       if self.config.get('to_lowercase', False):
           text = self.to_lowercase(text)
       
       if self.config.get('to_uppercase', False):
           text = self.to_uppercase(text)
       
       if self.config.get('filter_language', False):
           text = self.filter_language(text, self.config.get('target_language', 'en'))
       
       if self.config.get('remove_digits', False):
           text = self.remove_digits(text)
       
       if self.config.get('remove_single_characters', False):
           text = self.remove_single_characters(text)
       
       if self.config.get('normalize_accents', False):
           text = self.normalize_accents(text)
       
       if self.config.get('remove_short_words', False):
           text = self.remove_short_words(text, self.config.get('min_word_length', 2))
       
       if self.config.get('remove_long_words', False):
           text = self.remove_long_words(text, self.config.get('max_word_length', 20))
       
       text = self.remove_extra_spaces(text)
       
       return text

def get_nested_field(item, field_path):
   try:
       value = item
       for key in field_path.split('.'):
           if isinstance(value, dict):
               value = value.get(key)
           else:
               return None
       return value
   except:
       return None

def set_nested_field(item, field_path, value):
   keys = field_path.split('.')
   current = item
   
   for key in keys[:-1]:
       if key not in current:
           current[key] = {}
       current = current[key]
   
   current[keys[-1]] = value

def analyze_text_quality(texts):
   if not texts:
       return {}
   
   valid_texts = [t for t in texts if t and isinstance(t, str)]
   
   lengths = [len(t) for t in valid_texts]
   word_counts = [len(t.split()) for t in valid_texts]
   
   special_char_counts = [len(re.findall(r'[^\w\s]', t)) for t in valid_texts]
   url_counts = [len(re.findall(r'https?://\S+', t)) for t in valid_texts]
   
   return {
       'total_items': len(texts),
       'valid_items': len(valid_texts),
       'empty_items': len(texts) - len(valid_texts),
       'avg_length': sum(lengths) / len(lengths) if lengths else 0,
       'avg_words': sum(word_counts) / len(word_counts) if word_counts else 0,
       'avg_special_chars': sum(special_char_counts) / len(special_char_counts) if special_char_counts else 0,
       'items_with_urls': sum(1 for c in url_counts if c > 0),
       'min_length': min(lengths) if lengths else 0,
       'max_length': max(lengths) if lengths else 0
   }

def preprocess_json(input_file, output_file, field_path, config=None, custom_steps=None, backup_original=True):
   with open(input_file, 'r', encoding='utf-8') as f:
       data = json.load(f)
   
   if isinstance(data, dict) and 'results' in data:
       items = data['results']
       container = data
       items_key = 'results'
   elif isinstance(data, list):
       items = data
       container = None
       items_key = None
   else:
       items = [data]
       container = None
       items_key = None
   
   print(f"Found {len(items)} items to process")
   
   preprocessor = TextPreprocessor(config or {})
   
   original_texts = []
   processed_texts = []
   
   for item in items:
       original_text = get_nested_field(item, field_path)
       original_texts.append(original_text)
   
   print("\nOriginal text analysis:")
   original_stats = analyze_text_quality(original_texts)
   for key, value in original_stats.items():
       print(f"  {key}: {value}")
   
   print(f"\nProcessing texts...")
   
   for i, item in enumerate(items):
       if i % 1000 == 0:
           print(f"Processed {i}/{len(items)} items...")
       
       original_text = get_nested_field(item, field_path)
       
       if backup_original and original_text is not None:
           backup_field = f"{field_path}_original"
           set_nested_field(item, backup_field, original_text)
       
       if original_text is not None:
           processed_text = preprocessor.process_text(str(original_text), custom_steps)
           set_nested_field(item, field_path, processed_text)
           processed_texts.append(processed_text)
       else:
           processed_texts.append(None)
   
   print(f"Processed {len(items)}/{len(items)} items complete!")
   
   print("\nProcessed text analysis:")
   processed_stats = analyze_text_quality(processed_texts)
   for key, value in processed_stats.items():
       print(f"  {key}: {value}")
   
   print("\nImprovements:")
   if original_stats['avg_length'] > 0:
       length_change = (processed_stats['avg_length'] - original_stats['avg_length']) / original_stats['avg_length'] * 100
       print(f"  Average length change: {length_change:+.1f}%")
   
   if original_stats['avg_special_chars'] > 0:
       special_change = (processed_stats['avg_special_chars'] - original_stats['avg_special_chars']) / original_stats['avg_special_chars'] * 100
       print(f"  Special characters change: {special_change:+.1f}%")
   
   url_reduction = original_stats['items_with_urls'] - processed_stats['items_with_urls']
   print(f"  URLs removed: {url_reduction} items")
   
   empty_change = processed_stats['empty_items'] - original_stats['empty_items']
   print(f"  Empty items change: {empty_change:+d}")
   
   if container is not None:
       container[items_key] = items
       output_data = container
   else:
       output_data = items if len(items) > 1 else items[0]
   
   with open(output_file, 'w', encoding='utf-8') as f:
       json.dump(output_data, f, indent=2, ensure_ascii=False)
   
   print(f"\nSaved processed data to: {output_file}")
   
   print("\nSample transformations:")
   for i in range(min(3, len(original_texts))):
       if original_texts[i] and processed_texts[i]:
           print(f"\nSample {i+1}:")
           print(f"Original: {original_texts[i][:100]}...")
           print(f"Processed: {processed_texts[i][:100]}...")

def main():
   parser = argparse.ArgumentParser(description='Advanced text preprocessing for JSON data')
   parser.add_argument('input_file', help='Input JSON file')
   parser.add_argument('-o', '--output', help='Output JSON file')
   parser.add_argument('-f', '--field', default='text', help='Field path to process')
   parser.add_argument('--no-backup', action='store_true', help='Do not backup original field')
   
   parser.add_argument('--medical', action='store_true', help='Medical domain cleaning')
   parser.add_argument('--remove-urls', action='store_true', help='Remove URLs')
   parser.add_argument('--normalize-numbers', action='store_true', help='Normalize numbers')
   parser.add_argument('--remove-special', action='store_true', help='Remove special characters')
   parser.add_argument('--fix-case', action='store_true', help='Fix case issues')
   parser.add_argument('--remove-emojis', action='store_true', help='Remove emojis')
   parser.add_argument('--demojize-emojis', action='store_true', help='Convert emojis to text')
   parser.add_argument('--remove-stopwords', action='store_true', help='Remove stopwords')
   parser.add_argument('--stem', action='store_true', help='Apply stemming')
   parser.add_argument('--lemmatize', action='store_true', help='Apply lemmatization')
   parser.add_argument('--remove-punctuation', action='store_true', help='Remove punctuation')
   parser.add_argument('--lowercase', action='store_true', help='Convert to lowercase')
   parser.add_argument('--uppercase', action='store_true', help='Convert to uppercase')
   parser.add_argument('--filter-language', help='Filter by language code')
   parser.add_argument('--remove-digits', action='store_true', help='Remove digits')
   parser.add_argument('--remove-single-chars', action='store_true', help='Remove single characters')
   parser.add_argument('--normalize-accents', action='store_true', help='Remove accents')
   parser.add_argument('--min-word-length', type=int, default=2, help='Minimum word length')
   parser.add_argument('--max-word-length', type=int, default=20, help='Maximum word length')
   parser.add_argument('--remove-short-words', action='store_true', help='Remove short words')
   parser.add_argument('--remove-long-words', action='store_true', help='Remove long words')
   parser.add_argument('--stopwords-language', default='english', help='Stopwords language')
   
   parser.add_argument('--steps', nargs='+', help='Custom preprocessing steps')
   parser.add_argument('--list-steps', action='store_true', help='List available steps')
   
   args = parser.parse_args()
   
   if args.list_steps:
       preprocessor = TextPreprocessor()
       steps = [method for method in dir(preprocessor) if not method.startswith('_') and callable(getattr(preprocessor, method))]
       print("Available preprocessing steps:")
       for step in sorted(steps):
           print(f"  {step}")
       return
   
   if args.output is None:
       input_path = Path(args.input_file)
       args.output = str(input_path.parent / f"{input_path.stem}_cleaned{input_path.suffix}")
   
   config = {
       'medical_domain': args.medical,
       'remove_urls': args.remove_urls,
       'normalize_numbers': args.normalize_numbers,
       'remove_special_chars': args.remove_special,
       'fix_case': args.fix_case,
       'remove_emojis': args.remove_emojis,
       'demojize_emojis': args.demojize_emojis,
       'remove_stopwords': args.remove_stopwords,
       'stem_text': args.stem,
       'lemmatize_text': args.lemmatize,
       'remove_punctuation': args.remove_punctuation,
       'to_lowercase': args.lowercase,
       'to_uppercase': args.uppercase,
       'filter_language': bool(args.filter_language),
       'target_language': args.filter_language,
       'remove_digits': args.remove_digits,
       'remove_single_characters': args.remove_single_chars,
       'normalize_accents': args.normalize_accents,
       'remove_short_words': args.remove_short_words,
       'remove_long_words': args.remove_long_words,
       'min_word_length': args.min_word_length,
       'max_word_length': args.max_word_length,
       'stopwords_language': args.stopwords_language
   }
   
   preprocess_json(
       args.input_file,
       args.output,
       args.field,
       config,
       args.steps,
       not args.no_backup
   )

if __name__ == "__main__":
   main()