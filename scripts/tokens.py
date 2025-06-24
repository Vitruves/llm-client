import json
import numpy as np
from pathlib import Path

def load_tokenizer(model_path):
   try:
       from transformers import AutoTokenizer
       tokenizer = AutoTokenizer.from_pretrained(model_path)
       return tokenizer, 'transformers'
   except Exception as e:
       print(f"Failed to load with transformers: {e}")
       
   try:
       import tiktoken
       if model_path in ['gpt-4', 'gpt-3.5-turbo', 'text-davinci-003', 'text-embedding-ada-002']:
           encoding = tiktoken.encoding_for_model(model_path)
           return encoding, 'tiktoken'
       else:
           encoding = tiktoken.get_encoding("cl100k_base")
           print(f"Using cl100k_base encoding as fallback")
           return encoding, 'tiktoken'
   except Exception as e:
       print(f"Failed to load with tiktoken: {e}")
   
   return None, None

def count_tokens_with_tokenizer(text, tokenizer, tokenizer_type):
   if tokenizer_type == 'transformers':
       return len(tokenizer.encode(text, add_special_tokens=False))
   elif tokenizer_type == 'tiktoken':
       return len(tokenizer.encode(text))
   else:
       return len(text.split())

def get_nested_field(item, field_path):
   """Get nested field value using dot notation (e.g., 'data.ID')"""
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

def count_tokens(input_file, model_path="gpt-4", text_field="text", system_prompt=None, user_template=None, analyze_fields=None):
   tokenizer, tokenizer_type = load_tokenizer(model_path)
   
   if tokenizer is None:
       print("Warning: No tokenizer found, using word count approximation")
       tokenizer_type = 'word_count'
   
   # Default prompts
   if system_prompt is None:
       system_prompt = "You are a helpful and harmless assistant. You should think step-by-step, starting with \"<think>\n\"."
   
   if user_template is None:
       user_template = "Classify medical drug record about a medication into categories: ### 0 ### no weight change mentioned, ### 1 ### weight GAIN drug-induced or weight gain mentioned if no drug mentioned, ### 2 ### weight LOSS drug-induced or weight loss mentioned if no drug mentioned. Analyze step by step then provide classification number. Pay attention to finding weight gain and loss as less common. Testimony: ### {text} ###. Answer with classification number only:"
   
   with open(input_file, 'r', encoding='utf-8') as f:
       data = json.load(f)
   
   # Handle different JSON structures
   if isinstance(data, dict) and 'results' in data:
       annotations = data['results']
       print(f"Found 'results' array with {len(annotations)} items")
   elif isinstance(data, list):
       annotations = data
       print(f"Found array with {len(annotations)} items")
   else:
       print("Unsupported JSON structure")
       return
   
   print(f"Processing {len(annotations)} items...")
   print(f"Text field: '{text_field}'")
   print(f"Tokenizer: {tokenizer_type}")
   
   # Analyze additional fields if specified
   field_stats = {}
   if analyze_fields:
       for field in analyze_fields:
           field_stats[field] = []
   
   system_tokens = count_tokens_with_tokenizer(system_prompt, tokenizer, tokenizer_type)
   template_base = user_template.replace("{text}", "").replace("{" + text_field + "}", "")
   template_tokens = count_tokens_with_tokenizer(template_base, tokenizer, tokenizer_type)
   
   total_tokens_per_request = []
   text_tokens_per_request = []
   missing_field_count = 0
   
   # Process with progress indicator
   for i, item in enumerate(annotations):
       if i % 1000 == 0:
           print(f"Processed {i}/{len(annotations)} items...")
       
       # Get text from specified field
       text = get_nested_field(item, text_field)
       if text is None:
           text = ""
           missing_field_count += 1
       
       text = str(text)
       
       # Format user prompt
       if "{text}" in user_template:
           user_prompt = user_template.format(text=text)
       elif "{" + text_field + "}" in user_template:
           user_prompt = user_template.format(**{text_field: text})
       else:
           user_prompt = user_template + " " + text
       
       if tokenizer_type == 'word_count':
           prompt_tokens = len((system_prompt + " " + user_prompt).split()) * 1.3
           text_tokens = len(text.split()) * 1.3
       else:
           prompt_tokens = system_tokens + count_tokens_with_tokenizer(user_prompt, tokenizer, tokenizer_type)
           text_tokens = count_tokens_with_tokenizer(text, tokenizer, tokenizer_type)
       
       total_tokens_per_request.append(int(prompt_tokens))
       text_tokens_per_request.append(int(text_tokens))
       
       # Analyze additional fields
       for field in (analyze_fields or []):
           field_value = get_nested_field(item, field)
           if field_value is not None:
               field_text = str(field_value)
               field_tokens = count_tokens_with_tokenizer(field_text, tokenizer, tokenizer_type)
               field_stats[field].append(int(field_tokens))
           else:
               field_stats[field].append(0)
   
   print(f"Processed {len(annotations)}/{len(annotations)} items complete!")
   if missing_field_count > 0:
       print(f"Warning: {missing_field_count} items missing '{text_field}' field")
   
   total_tokens = sum(total_tokens_per_request)
   avg_tokens_per_request = np.mean(total_tokens_per_request)
   avg_text_tokens = np.mean(text_tokens_per_request)
   
   percentiles = [5, 10, 25, 50, 75, 90, 95, 99]
   quartiles = [25, 50, 75]
   
   print(f"\nToken Analysis for {len(annotations)} samples:")
   print(f"System prompt: {system_tokens} tokens")
   print(f"Template (without text): {template_tokens} tokens")
   print(f"")
   
   print(f"TEXT FIELD ('{text_field}') STATISTICS:")
   print(f"  Mean: {avg_text_tokens:.1f}")
   print(f"  Std: {np.std(text_tokens_per_request):.1f}")
   print(f"  Min: {np.min(text_tokens_per_request)}")
   print(f"  Max: {np.max(text_tokens_per_request)}")
   print(f"")
   
   print(f"TEXT FIELD PERCENTILES:")
   for p in percentiles:
       value = np.percentile(text_tokens_per_request, p)
       print(f"  P{p}: {value:.0f}")
   print(f"")
   
   print(f"TEXT FIELD QUARTILES:")
   q1, q2, q3 = np.percentile(text_tokens_per_request, quartiles)
   iqr = q3 - q1
   print(f"  Q1 (25th): {q1:.0f}")
   print(f"  Q2 (50th): {q2:.0f}")
   print(f"  Q3 (75th): {q3:.0f}")
   print(f"  IQR: {iqr:.0f}")
   print(f"")
   
   print(f"TOTAL REQUEST TOKENS STATISTICS:")
   print(f"  Mean: {avg_tokens_per_request:.1f}")
   print(f"  Std: {np.std(total_tokens_per_request):.1f}")
   print(f"  Min: {np.min(total_tokens_per_request)}")
   print(f"  Max: {np.max(total_tokens_per_request)}")
   print(f"")
   
   print(f"TOTAL REQUEST TOKENS PERCENTILES:")
   for p in percentiles:
       value = np.percentile(total_tokens_per_request, p)
       print(f"  P{p}: {value:.0f}")
   print(f"")
   
   print(f"TOTAL REQUEST TOKENS QUARTILES:")
   q1, q2, q3 = np.percentile(total_tokens_per_request, quartiles)
   iqr = q3 - q1
   print(f"  Q1 (25th): {q1:.0f}")
   print(f"  Q2 (50th): {q2:.0f}")
   print(f"  Q3 (75th): {q3:.0f}")
   print(f"  IQR: {iqr:.0f}")
   print(f"")
   
   # Additional field analysis
   for field, tokens_list in field_stats.items():
       if tokens_list:
           print(f"FIELD '{field}' STATISTICS:")
           valid_tokens = [t for t in tokens_list if t > 0]
           if valid_tokens:
               print(f"  Mean: {np.mean(valid_tokens):.1f}")
               print(f"  Std: {np.std(valid_tokens):.1f}")
               print(f"  Min: {np.min(valid_tokens)}")
               print(f"  Max: {np.max(valid_tokens)}")
               print(f"  Valid entries: {len(valid_tokens)}/{len(tokens_list)}")
               
               print(f"  Percentiles:")
               for p in [25, 50, 75, 90, 95]:
                   value = np.percentile(valid_tokens, p)
                   print(f"    P{p}: {value:.0f}")
               print(f"")
   
   print(f"SUMMARY:")
   print(f"  Total tokens for all requests: {total_tokens:,}")
   print(f"  Average tokens per request: {avg_tokens_per_request:.1f}")
   print(f"  Requests > 1000 tokens: {sum(1 for t in total_tokens_per_request if t > 1000)}")
   print(f"  Requests > 2000 tokens: {sum(1 for t in total_tokens_per_request if t > 2000)}")
   print(f"  Requests > 4000 tokens: {sum(1 for t in total_tokens_per_request if t > 4000)}")
   
   return {
       'total_tokens': total_tokens,
       'avg_tokens_per_request': avg_tokens_per_request,
       'tokenizer_type': tokenizer_type,
       'field_stats': field_stats
   }

if __name__ == "__main__":
   import argparse
   
   parser = argparse.ArgumentParser(description='Count tokens with flexible field selection')
   parser.add_argument('input_file', help='Input JSON file')
   parser.add_argument('-m', '--model', default='gpt-4', help='Model path or name')
   parser.add_argument('-f', '--field', default='text', help='Field name for text content (supports dot notation like data.text)')
   parser.add_argument('-s', '--system', help='Custom system prompt')
   parser.add_argument('-u', '--user', help='Custom user template (use {text} or {fieldname} placeholder)')
   parser.add_argument('-a', '--analyze', nargs='+', help='Additional fields to analyze (supports dot notation)')
   parser.add_argument('--show-sample', action='store_true', help='Show sample of data structure')
   
   args = parser.parse_args()
   
   # Show sample data structure if requested
   if args.show_sample:
       with open(args.input_file, 'r') as f:
           data = json.load(f)
       
       if isinstance(data, dict) and 'results' in data:
           sample = data['results'][0] if data['results'] else {}
       elif isinstance(data, list):
           sample = data[0] if data else {}
       else:
           sample = data
       
       print("Sample data structure:")
       print(json.dumps(sample, indent=2)[:500] + "...")
       print("\nAvailable fields:")
       
       def get_all_keys(obj, prefix=""):
           keys = []
           if isinstance(obj, dict):
               for key, value in obj.items():
                   full_key = f"{prefix}.{key}" if prefix else key
                   keys.append(full_key)
                   if isinstance(value, dict):
                       keys.extend(get_all_keys(value, full_key))
           return keys
       
       all_keys = get_all_keys(sample)
       for key in sorted(all_keys):
           print(f"  {key}")
       print()
   
   count_tokens(args.input_file, args.model, args.field, args.system, args.user, args.analyze)