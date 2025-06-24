# LLM Client Configuration Reference

This guide provides a comprehensive and user-friendly reference for configuring the LLM Client.
It covers all available options, their purpose, and how to use them effectively.

For a practical, hands-on example with all options demonstrated, please refer to the `config/showcase_config.yaml` file in your project. You can copy and modify this file as a starting point for your own configurations.

## Configuration Structure

The main configuration is organized into five top-level sections:

```yaml
provider:       # Defines how the client connects to the LLM server (e.g., vLLM, Llama.cpp, OpenAI).
model:          # Specifies the LLM and its generation parameters (e.g., temperature, max tokens).
classification: # Configures prompt templates, input field mapping, and how to parse LLM responses.
processing:     # Controls concurrency, performance, and live metrics during processing.
output:         # Determines how processed results are formatted and saved (e.g., JSON, CSV, Parquet, XLSX).
```

---

## 1. Provider Configuration

This section defines the connection details for your chosen Large Language Model (LLM) provider.

### Fields:
- **`name`** (string, **required**):
  -   **Purpose**: Specifies the type of LLM provider the client will connect to.
  -   **Valid Values**: `"vllm"`, `"llamacpp"`, `"openai"`
  -   **Example**: `name: "vllm"`

- **`base_url`** (string, **required for `vllm` and `llamacpp` providers**):
  -   **Purpose**: The base URL of your LLM API endpoint.
  -   **Example**: `base_url: "http://localhost:8000/v1"` (for a local vLLM server)

- **`timeout`** (string, optional):
  -   **Purpose**: Sets the maximum duration to wait for a response from the LLM server.
  -   **Format**: Go duration string (e.g., `"30s"`, `"5m"`, `"1h"`).
  -   **Example**: `timeout: "600s"`

- **`api_key`** (string, **required for `openai` provider**):
  -   **Purpose**: Your authentication key for accessing the LLM API.
  -   **Note**: For `llamacpp` and self-hosted `vllm` instances, an API key might not be required.
  -   **Example**: `api_key: "sk-your-openai-api-key"`

---

## 2. Model Configuration

This section defines the specific Large Language Model to be used and its generation parameters, controlling how the model generates responses.

### Fields:
- **`name`** (string, **required**):
  -   **Purpose**: The identifier for the specific model you want to use. This should match the model deployed on your server or the one you're accessing via API.
  -   **Example**: `name: "qwen-2.5-7b-chat"`

- **`parameters`** (object, optional):
  -   **Purpose**: A collection of parameters that fine-tune the model's generation behavior. These directly influence the style, creativity, and accuracy of the LLM's output.

### Model Parameters:
- **`temperature`** (float, optional, range: 0.0-2.0):
  -   **Purpose**: Controls the randomness of the output. Higher values result in more diverse and creative text, while lower values make the output more deterministic and focused.
  -   **Example**: `temperature: 0.7` (a common balance)

- **`max_tokens`** (int, optional):
  -   **Purpose**: The maximum number of tokens (words or pieces of words) the LLM will generate in its response.
  -   **Example**: `max_tokens: 1024`

- **`min_tokens`** (int, optional):
  -   **Purpose**: The minimum number of tokens the LLM will generate in its response.
  -   **Example**: `min_tokens: 1`

- **`top_p`** (float, optional, range: 0.0-1.0):
  -   **Purpose**: Nucleus sampling. The model considers only the smallest set of tokens whose cumulative probability exceeds this threshold. Useful for controlling diversity while avoiding low-probability tokens.
  -   **Example**: `top_p: 0.9`

- **`top_k`** (int, optional):
  -   **Purpose**: Top-k sampling. The model considers only the `top_k` most likely tokens at each step.
  -   **Example**: `top_k: 50`

- **`min_p`** (float, optional, range: 0.0-1.0):
  -   **Purpose**: Minimum probability threshold for token selection.
  -   **Example**: `min_p: 0.05`

- **`repetition_penalty`** (float, optional):
  -   **Purpose**: Penalizes new tokens based on their existing frequency in the generated text so far. Higher values reduce repetition.
  -   **Example**: `repetition_penalty: 1.1`

- **`presence_penalty`** (float, optional, range: -2.0 to 2.0):
  -   **Purpose**: Penalizes new tokens based on whether they appear in the text generated so far. Positive values discourage topics that have already been discussed.
  -   **Example**: `presence_penalty: 0.0`

- **`frequency_penalty`** (float, optional, range: -2.0 to 2.0):
  -   **Purpose**: Penalizes new tokens based on their frequency in the text generated so far. Similar to `repetition_penalty` but more nuanced.
  -   **Example**: `frequency_penalty: 0.0`

- **`seed`** (int64, optional):
  -   **Purpose**: A random seed for reproducibility. Using the same seed with the same input and model parameters should yield identical outputs.
  -   **Example**: `seed: 42`

- **`n`** (int, optional):
  -   **Purpose**: The number of independent completions to generate for each prompt. Note: This parameter is primarily supported by certain providers like OpenAI.
  -   **Example**: `n: 1`

- **`stop`** ([]string, optional):
  -   **Purpose**: A list of strings that, if generated, will cause the model to stop generating further tokens.
  -   **Example**: `stop: ["<|im_end|>", "Observation:"]`

- **`stop_token_ids`** ([]int, optional):
  -   **Purpose**: A list of token IDs that, if generated, will cause the model to stop generating further tokens. Useful for specific model control.
  -   **Example**: `stop_token_ids: [151645, 151643]`

- **`bad_words`** ([]string, optional):
  -   **Purpose**: A list of words that the model should avoid generating.
  -   **Example**: `bad_words: ["offensive", "inappropriate"]`

- **`include_stop_str_in_output`** (bool, optional):
  -   **Purpose**: If `true`, the stop string (if matched) will be included in the model's output.
  -   **Example**: `include_stop_str_in_output: false`

- **`ignore_eos`** (bool, optional):
  -   **Purpose**: If `true`, the model will ignore the end-of-sequence (EOS) token, potentially generating longer responses than usual if no other stop condition is met.
  -   **Example**: `ignore_eos: false`

- **`logprobs`** (int, optional):
  -   **Purpose**: The number of log probabilities to return for the generated tokens.
  -   **Example**: `logprobs: 0`

- **`prompt_logprobs`** (int, optional):
  -   **Purpose**: The number of log probabilities to return for the tokens in the prompt itself.
  -   **Example**: `prompt_logprobs: 0`

- **`truncate_prompt_tokens`** (int, optional):
  -   **Purpose**: The maximum number of tokens in the prompt before it gets truncated. Useful for managing context window limits.
  -   **Example**: `truncate_prompt_tokens: 2048`

- **`chat_format`** (string, optional):
  -   **Purpose**: Specifies the chat format to be used with llama.cpp models. Essential for correct conversation turn handling.
  -   **Valid Values**: (e.g., `"chatml"`, `"llama-2"`, `"mistral"`)
  -   **Example**: `chat_format: "chatml"`

- **`skip_special_tokens`** (bool, optional):
  -   **Purpose**: If `true`, special tokens (e.g., `<s>`, `</s>`, `<unk>`) will be excluded from the generated output.
  -   **Example**: `skip_special_tokens: false`

- **`spaces_between_special_tokens`** (bool, optional):
  -   **Purpose**: If `true`, adds a space between special tokens in the generated output.
  -   **Example**: `spaces_between_special_tokens: false`

---

## 3. Classification Configuration

This section is crucial for defining how your LLM interacts with your data: how prompts are constructed, how input fields are mapped, and how the LLM's responses are parsed to extract the final answer.

### 3.1. Template Configuration

Defines the structure of the messages sent to the LLM.

#### Fields:
- **`system`** (string, **required**):
  -   **Purpose**: The system-level instruction or role definition for the LLM. This sets the context for the model's behavior.
  -   **Example**: `"You are a helpful assistant. Classify the user's input into one of the following categories: positive, negative, neutral."`

- **`user`** (string, **required**):
  -   **Purpose**: The template for the user-facing prompt. It uses placeholders (`{PLACEHOLDER_NAME}`) that will be dynamically replaced with values from your input data.
  -   **Example**: `"Review: {REVIEW_TEXT}\nClassify this review."`

#### Template Placeholder Grammar:
-   **Syntax**: Placeholders are enclosed in curly braces, e.g., `{PLACEHOLDER_NAME}`.
-   **Built-in Placeholders**:
    -   `{text}`: Maps to the `DataRow.Text` field, which typically holds the primary text content of your input row.
    -   `{index}`: Maps to the `DataRow.Index` (0-indexed integer) of the current input row.
-   **Dynamic Placeholders**: Any other `{FIELD_NAME}` you define will map to a corresponding key in the `DataRow.Data` map (which contains all original columns from your input file).
-   **Case Handling**: The system attempts case-insensitive matching for dynamic placeholders (e.g., `{REVIEW_TEXT}` can match `review_text`, `ReviewText`, etc., in your input data).
-   **Nested Fields**: Supports dot notation for nested JSON objects in your `DataRow.Data`, e.g., `{parent.child}`.

### 3.2. Field Mapping Configuration

This section provides granular control over how input data fields are identified and used by the LLM Client.

#### Fields:
- **`input_text_field`** (string, optional):
  -   **Purpose**: Explicitly specifies which original input column contains the primary text content that should be used for the `Result.InputText` field in the output. This is also used as a fallback for the `{text}` placeholder if `DataRow.Text` is empty.
  -   **Example**: `input_text_field: "product_review_description"`
  -   **Default Behavior**: If not set, the system will attempt to infer the primary text field from common column names like "text", "content", "review", etc.

- **`placeholder_map`** (map[string]string, optional):
  -   **Purpose**: Allows you to create custom mappings between the placeholders used in your `user` prompt template and the actual column names in your input data. This is useful when your template placeholders don't directly match your input column names.
  -   **Example**:
    ```yaml
    placeholder_map:
      REVIEW_TEXT: "customer_feedback" # Maps {REVIEW_TEXT} in template to "customer_feedback" column
      ITEM_ID: "product_identifier"   # Maps {ITEM_ID} in template to "product_identifier" column
    ```

#### Field Resolution Priority:
The LLM Client resolves input fields for templates and `InputText` in the following order:
1.  **`placeholder_map`** custom mappings (if defined).
2.  **Exact case match** in `DataRow.Data` for a dynamic placeholder.
3.  **Case-insensitive match** in `DataRow.Data` for a dynamic placeholder.
4.  **Nested field traversal** for dot notation (e.g., `{parent.child}`).
5.  **`input_text_field`** specified in this section (for `InputText`).
6.  **Built-in placeholder rules** (e.g., `DataRow.Text` for `{text}`).

### 3.3. Parsing Configuration

Defines how to extract the final answer from the LLM's raw response. This is critical for structured output.

#### Core Fields:
- **`find`** ([]string, **required**):
  -   **Purpose**: A prioritized list of exact string values that the parser will look for in the LLM's response. The first string found (in the order listed) will be extracted as the answer.
  -   **Special Value**: Including `"*"` in this list will match any non-empty response if no other `find` strings or `answer_patterns` yield a result.
  -   **Example**: `find: ["positive", "negative", "neutral", "*"]`

- **`default`** (string, **required**):
  -   **Purpose**: The value to be used as the `FinalAnswer` if, after all parsing attempts (patterns, find list, mapping), no valid answer can be extracted from the LLM's response.
  -   **Example**: `default: "unknown"`

- **`fallback`** (string, **required**):
  -   **Purpose**: The value to be used as the `FinalAnswer` if an unexpected error occurs *during* the parsing process itself (e.g., an invalid regex pattern prevents parsing). This helps ensure a result is always recorded, even in case of internal parsing issues.
  -   **Example**: `fallback: "error_parsing"`

- **`map`** (map[string]string, optional):
  -   **Purpose**: A mapping to transform extracted answers. This is useful if the LLM's raw output needs to be standardized or converted to a different representation.
  -   **Application**: Applied *after* an answer has been extracted by `answer_patterns` or `find`. Unmapped values pass through unchanged.
  -   **Example**:
    ```yaml
    map:
      POS: "positive"
      NEG: "negative"
      NEU: "neutral"
    ```

- **`thinking_tags`** (string, optional):
  -   **Purpose**: Defines the start and end tags that delineate "thinking" content within the LLM's response. This content can be extracted separately for analysis.
  -   **Format**: Specify both tags together, separated by `></`. For example, `"<thinking></thinking>"` means content between `<thinking>` and `</thinking>`.
  -   **Example**: `thinking_tags: "<thought></thought>"`

- **`preserve_thinking`** (bool, optional):
  -   **Purpose**: If `true`, the content identified within `thinking_tags` will be preserved in the `ThinkingContent` field of the output `Result`. If `false`, this content will be removed from the `RawResponse` before further parsing.
  -   **Example**: `preserve_thinking: true`

- **`answer_patterns`** ([]string, optional):
  -   **Purpose**: A list of regular expression patterns to extract the final answer from the LLM's raw response. These are attempted *before* the `find` list.
  -   **Syntax**: Go regex (RE2 syntax).
  -   **Capture Requirement**: Each pattern *must* contain exactly one capture group `(...)`. The content of this capture group will be the extracted answer. Patterns without a capture group will be skipped.
  -   **Order**: Patterns are tested sequentially in the order they appear. The first successful match wins.
  -   **Example**:
    ```yaml
    answer_patterns:
      - "The classification is: (.*)"
      - "Category: (positive|negative|neutral)"
    ```

- **`case_sensitive`** (bool, optional):
  -   **Purpose**: Controls case sensitivity for `find` string matching and `answer_patterns` (unless overridden by regex flags like `(?i)`).
  -   **Default**: `false` (case-insensitive)
  -   **Example**: `case_sensitive: false`

- **`exact_match`** (bool, optional):
  -   **Purpose**: If `true`, the strings in the `find` array must match the LLM's response *exactly* (or the extracted part of it). If `false`, substring matching is allowed.
  -   **Default**: `false` (substring matching allowed)
  -   **Example**: `exact_match: false`

#### Parsing Algorithm Execution Order:
The LLM Client processes the LLM's response in a defined sequence to determine the `FinalAnswer`:
1.  **Thinking Extraction**: If `thinking_tags` are specified, the content within these tags is extracted and, if `preserve_thinking` is `false`, removed from the raw response.
2.  **Pattern Matching**: The `answer_patterns` are tested sequentially against the (possibly modified) raw response. The content of the first successful pattern's capture group becomes the candidate answer.
3.  **String Matching (`find` array)**: If no `answer_pattern` yielded a result, the `find` array is then used to search for exact or substring matches (depending on `exact_match`). The first match found becomes the candidate answer.
4.  **Mapping**: If a candidate answer was found, the `map` transformations are applied to it.
5.  **Fallback/Default**:
    -   If an unexpected error occurred *during* any parsing step, the `fallback` value is used.
    -   If no answer was found after all pattern matching, string matching, and mapping attempts, the `default` value is used.

#### Details on Pattern Matching:
-   **Regex Engine**: Uses Go's RE2 syntax, which guarantees linear time complexity but does not support certain advanced regex features like lookaheads/lookbehinds or backreferences.
-   **Capture Requirement**: Crucially, each pattern *must* include exactly one capture group `(...)` to define what part of the match should be extracted as the answer.
    -   **Good Example**: `"answer": "([^"]+)"` (Captures content between quotes)
    -   **Bad Example**: `"answer": "[^"]+"` (No capture group, will be skipped)
    -   **Bad Example**: `(first)(second)` (Multiple capture groups, only the first will be used)
-   **Common RE2 Syntax**:
    -   `.`: Matches any character (except newline).
    -   `*`: Matches zero or more occurrences of the preceding character/group.
    -   `+`: Matches one or more occurrences of the preceding character/group.
    -   `?`: Matches zero or one occurrence of the preceding character/group (makes preceding greedy match non-greedy).
    -   `|`: OR operator (e.g., `(cat|dog)` matches "cat" or "dog").
    -   `[]`: Character set (e.g., `[0-9]` for digits, `[a-zA-Z]` for letters).
    -   `[^]`: Negated character set (e.g., `[^\n]` for any character except newline).
    -   `\d`: Matches a digit (equivalent to `[0-9]`).
    -   `\w`: Matches a word character (alphanumeric + underscore).
    -   `\s`: Matches a whitespace character.
    -   `\b`: Word boundary.
    -   `^`: Start of a string/line.
    -   `$`: End of a string/line.
    -   `(?i)`: Case-insensitive flag (can be used at the start of a pattern, e.g., `(?i)positive`).
    -   `(?s)`: Dot matches newline flag (can be used at the start of a pattern, e.g., `(?s)Start(.*)End`).
-   **Multiline Matching**: You can enable multiline matching for patterns using the `(?s)` flag (e.g., `(?s)Start(.*)End`) if your content spans multiple lines.
-   **Unicode Support**: Full Unicode character support in patterns.
-   **Performance**: All regex patterns are compiled once at startup to optimize performance during processing.

#### Tuning Guide for LLM Parameters:

This section provides guidance on how to effectively tune the various configuration parameters to achieve optimal performance and desired output from your LLM classification tasks.

##### Model Parameters (`model.parameters`):

-   **`temperature`**:
    -   **Low (0.0-0.3)**: Ideal for tasks requiring deterministic, factual, and precise answers (e.g., entity extraction, strict classification). Responses will be less creative and more focused.
    -   **Medium (0.4-0.7)**: A good balance for most general tasks, allowing some creativity while maintaining coherence. Often a good starting point.
    -   **High (0.8-1.0+)**: Use for creative generation, brainstorming, or when you need diverse outputs. May increase the risk of irrelevant or nonsensical responses.
    -   **Tuning Tip**: Start low and gradually increase if you need more diversity. Adjust in small increments (e.g., 0.1).

-   **`max_tokens`**:
    -   **Tuning Tip**: Set this value to slightly more than the maximum expected length of your classification answer or extracted text. Setting it too high can waste tokens/compute, while too low will truncate responses.

-   **`top_p` (Nucleus Sampling) and `top_k` (Top-K Sampling)**:
    -   These parameters control the diversity of generated tokens. Generally, use one or the other, not both simultaneously, as their effects can overlap.
    -   **`top_p`**: Recommended for most cases as it dynamically adjusts the token pool size based on probability distribution. Lower `top_p` values (e.g., 0.7-0.9) are common.
    -   **`top_k`**: Useful if you want to strictly limit the number of token choices.
    -   **Tuning Tip**: Start with `top_p: 0.9` and adjust if you need finer control over token selection. If using `top_k`, common values are 40-100.

-   **`repetition_penalty`, `presence_penalty`, `frequency_penalty`**:
    -   These help prevent repetitive or generic outputs.
    -   **`repetition_penalty` (e.g., 1.05-1.2)**: Good for general reduction of exact phrase repetition.
    -   **`presence_penalty` / `frequency_penalty` (e.g., 0.0-0.5)**: Use if the model tends to repeat concepts or topics. Experiment with small positive values.
    -   **Tuning Tip**: Start with `repetition_penalty: 1.1`. Only adjust `presence_penalty` or `frequency_penalty` if you observe specific issues with topic repetition.

-   **`seed`**:
    -   **Tuning Tip**: Always use a fixed `seed` during development and testing to ensure reproducibility of results. This is crucial for debugging and comparing model changes.

-   **`stop`** and **`stop_token_ids`**:
    -   **Tuning Tip**: Define explicit `stop` sequences (e.g., `["\nObservation:"]`, `"<|im_end|>"]`) that your prompt templates are designed to end with. This prevents the LLM from generating extra content and ensures clean parsing. Using `stop_token_ids` is more robust if you know the specific token IDs.

-   **`chat_format`**:
    -   **Tuning Tip**: Ensure this strictly matches the fine-tuning format of your Llama.cpp model (e.g., `"chatml"`, `"llama-2"`, `"mistral"`). Incorrect format will lead to poor responses.

##### Processing Parameters (`processing`):

-   **`workers`**:
    -   **Tuning Tip**: This controls concurrency. Start with a number appropriate for your CPU cores and the capabilities of your LLM server. Too many workers can overload the server or consume excessive memory. Gradually increase to find the sweet spot that maximizes throughput without introducing errors. Monitor server load if applicable.

-   **`batch_size`**:
    -   **Tuning Tip**: For classification, `1` is often sufficient as many LLM APIs process single prompts. Increase this only if your specific LLM provider or deployment (e.g., vLLM with `max_model_len` configured) demonstrably benefits from larger batches, which can reduce overhead.

-   **`repeat`**:
    -   **Tuning Tip**: Use `repeat > 1` (e.g., `3` or `5`) when you need to increase the reliability of your classifications or quantify the model's confidence/variance. This enables "consensus mode". Analyze the `attempts` and `consensus` fields in the output to evaluate consistency.

-   **`rate_limit`**:
    -   **Tuning Tip**: Set to `true` when `repeat > 1` and you are interacting with public APIs or self-hosted models that have strict rate limits. This prevents `HTTP 429 Too Many Requests` errors.

##### Output Parameters (`output`):

-   **`format`**:
    -   **Tuning Tip**: Choose based on your downstream analysis needs. `json` or `parquet` are generally preferred for retaining full data fidelity (especially with nested `original_data`, `attempts`, `consensus`), while `csv` or `xlsx` are suitable for simpler, flattened outputs.

-   **`include_raw_response`** / **`include_thinking`**:
    -   **Tuning Tip**: Set these to `true` during development and debugging to inspect the full LLM output and its internal "thinking" process. Once confident in your parsing, you can set them to `false` to reduce output file size if not needed for final analysis.

##### General Tuning Workflow:

1.  **Start Simple**: Begin with a basic configuration (e.g., `temperature: 0.7`, `max_tokens: 128`, `workers: 4`, `batch_size: 1`, `repeat: 1`).
2.  **Iterate and Observe**: Make small, incremental changes to one or two parameters at a time.
3.  **Monitor Performance**: Pay attention to `response_time_ms`, success rates, and any error messages in your output.
4.  **Evaluate Output Quality**: Critically review the `final_answer` and `raw_response` (if included) to assess if the LLM is behaving as expected.
5.  **Reproducibility**: Always use a `seed` when comparing parameter changes.

This comprehensive guide should help you effectively configure and tune the LLM Client for your specific use cases.

---

## 4. Processing Configuration

This section controls the execution flow, concurrency, and real-time metrics for your LLM classification tasks.

### Fields:
- **`workers`** (int, **required**):
  -   **Purpose**: The number of concurrent goroutines (lightweight threads) that will process your input data. Higher numbers can speed up processing on multi-core systems, but may also increase memory usage or LLM server load.
  -   **Example**: `workers: 100`

- **`batch_size`** (int, **required**):
  -   **Purpose**: The number of input items to send to the LLM in a single API request. While many LLMs support batching, this is typically set to `1` for most classification tasks unless your provider explicitly supports and benefits from larger batches.
  -   **Example**: `batch_size: 1`

- **`repeat`** (int, **required**, range: 1-10):
  -   **Purpose**: Determines how many times each individual input item will be sent to the LLM. This is useful for "consensus mode," where multiple responses for the same input are generated to improve reliability or analyze answer variance.
  -   **Example**: `repeat: 3` (sends each item 3 times)

- **`rate_limit`** (bool, **required**):
  -   **Purpose**: If `true`, a 1-second delay will be introduced between consecutive requests for the same item when `repeat` is greater than 1. This helps prevent rate limiting issues with some LLM providers.
  -   **Example**: `rate_limit: false`

- **`flashinfer_safe`** (bool, optional):
  -   **Purpose**: If `true`, disables certain features that might be incompatible or cause issues with FlashInfer optimizations on some LLM backends (e.g., specific stop token configurations). Only set this if you encounter issues with FlashInfer.
  -   **Example**: `flashinfer_safe: false`

### Live Metrics Configuration:
This sub-section enables and configures the real-time display and calculation of classification performance metrics during processing.

#### Fields:
- **`enabled`** (bool, **required**):
  -   **Purpose**: If `true`, live metrics are calculated and displayed in the terminal during processing, providing real-time feedback on model performance.
  -   **Example**: `enabled: true`

- **`metric`** (string, **required** when `enabled: true`):
  -   **Purpose**: The type of classification metric to calculate.
  -   **Valid Values**: `"accuracy"`, `"f1"`, `"kappa"`
  -   **Example**: `metric: "accuracy"`

- **`ground_truth`** (string, **required** when `enabled: true`):
  -   **Purpose**: The name of the input column (from your original data) that contains the true, correct labels for comparison against the LLM's predictions.
  -   **Example**: `ground_truth: "actual_category"`

- **`average`** (string, optional, **applicable for `f1` and `kappa` metrics**):
  -   **Purpose**: Specifies the averaging method for multi-class F1-score or Kappa.
  -   **Valid Values**: `"macro"` (unweighted mean), `"micro"` (globally calculates metrics), `"weighted"` (weighted by class support).
  -   **Example**: `average: "macro"`

- **`classes`** ([]string, optional, **recommended for `f1` and `kappa` metrics**):
  -   **Purpose**: A list of all expected unique class labels that the LLM might output or that are present in your `ground_truth` column. Providing this helps ensure accurate metric calculation, especially for F1 and Kappa.
  -   **Example**: `classes: ["positive", "negative", "neutral"]`

---

## 5. Output Configuration

This section specifies how the results of the LLM classification process are saved to files, including the output directory, file format, and what additional information to include.

### Fields:
- **`directory`** (string, **required**):
  -   **Purpose**: The path to the directory where all output files (results, resume states, etc.) will be saved. The directory will be created if it does not exist.
  -   **Example**: `directory: "results/my_classification_run"`

- **`format`** (string, **required**):
  -   **Purpose**: The desired file format for saving the classification results.
  -   **Valid Values**: `"json"`, `"csv"`, `"parquet"`, `"xlsx"`
  -   **Example**: `format: "json"`

- **`include_raw_response`** (bool, optional):
  -   **Purpose**: If `true`, the full, unparsed response string directly from the LLM will be included in the output file for each processed item. Useful for debugging or further analysis.
  -   **Example**: `include_raw_response: false`

- **`include_thinking`** (bool, optional):
  -   **Purpose**: If `true`, any "thinking" content extracted from the LLM's response (as defined by `classification.parsing.thinking_tags`) will be included in the output file.
  -   **Example**: `include_thinking: true`

- **`input_text_field`** (string, **DEPRECATED**):
  -   **Purpose**: This field is deprecated. Please use `classification.field_mapping.input_text_field` instead for more flexible and explicit control over how the primary input text is identified and used.

### Output Content Details:

All output formats (`json`, `csv`, `parquet`, `xlsx`) will include the following core fields for each processed item. The way they are represented (e.g., as nested objects in JSON vs. flattened columns in CSV) will vary by format, but the data will be present.

-   `index`: The 0-indexed row number from the original input file.
-   `input_text`: The primary text content that was sent to the LLM. This is resolved based on your `classification.field_mapping.input_text_field` or inferred from common column names.
-   `original_data`:
    -   **JSON/Parquet**: This will be a nested map/struct containing all original columns and their values from the input file row. This ensures all original context is preserved.
    -   **CSV/XLSX**: The original columns will be flattened and appear as individual columns in the output, prefixed with their original column names.
-   `ground_truth`: The true label for the item, if you provided it in your input data and configured `processing.live_metrics.ground_truth`.
-   `final_answer`: The final classified answer or extracted value that the LLM Client determined from the model's response.
-   `success`: A boolean (`true`/`false`) indicating whether the LLM processing for this specific item completed successfully.
-   `error`: An error message if the processing failed for this item (e.g., network error, LLM returned an error). This field will be empty if `success` is `true`.
-   `response_time_ms`: The duration (in milliseconds) that the LLM request took for this item.
-   `raw_response`: (Optional, included if `include_raw_response` is `true`) The complete, unparsed response string received directly from the LLM.
-   `thinking_content`: (Optional, included if `include_thinking` is `true`) Any specific "thinking" text extracted from the LLM's response using `thinking_tags`.
-   `attempts`: (Only for JSON/Parquet output if `processing.repeat` > 1) A detailed array of each individual attempt made for the item in consensus mode, including their raw responses and parsed answers.
-   `consensus`: (Only for JSON/Parquet output if `processing.repeat` > 1) Provides details about the consensus calculation if multiple attempts were made, including the final consensus answer, agreement ratio, and distribution of answers.
-   `tool_calls`: (Optional) Any tool calls identified in the model's response.
-   `usage`: (Optional) Token usage statistics from the model response (e.g., prompt tokens, completion tokens).

---

## Data Types

Understanding the underlying data structures helps in preparing your input and interpreting the output.

### `DataRow` Structure:
This struct represents a single row (or item) from your input file.

```go
type DataRow struct {
    Index int                    // The 0-indexed row number of the data item from the input file.
    Text  string                 // The primary text content of the row. This field is used as the default input for the LLM if no `input_text_field` is specified.
    Data  map[string]interface{} // A map containing all original fields (columns) and their values from the input JSON, CSV, Excel, or Parquet row. This allows for dynamic access to any original data.
}
```

### `Result` Structure:
This struct encapsulates the outcome of processing a single `DataRow` by the LLM Client.

```go
type Result struct {
    Index           int                    `json:"index"`             // The 0-indexed row number from the input file.
    InputText       string                 `json:"input_text"`        // The primary text content that was sent to the LLM (resolved from DataRow).
    OriginalData    map[string]interface{} `json:"original_data,omitempty"` // All original columns and their values from the input DataRow.
    GroundTruth     string                 `json:"ground_truth"`      // The ground truth label for the item, if provided in the input and configured.
    FinalAnswer     string                 `json:"final_answer"`      // The final classified answer or extracted value from the LLM.
    RawResponse     string                 `json:"raw_response"`      // The complete, raw response string received from the LLM.
    ThinkingContent string                 `json:"thinking_content,omitempty"` // Any extracted "thinking" content from the LLM's response.
    Success         bool                   `json:"success"`           // `true` if the LLM processing for this item was successful, `false` otherwise.
    Error           string                 `json:"error,omitempty"`   // An error message if the processing failed.
    ResponseTime    time.Duration          `json:"response_time"`     // The duration of the LLM request for this item.
    Attempts        []Attempt              `json:"attempts,omitempty"`// (Optional) A slice of individual attempts if `processing.repeat` is greater than 1.
    Consensus       *Consensus             `json:"consensus,omitempty"` // (Optional) Details about the consensus calculation if `processing.repeat` is greater than 1.
    ToolCalls       []ToolCall             `json:"tool_calls,omitempty"` // (Optional) Any tool calls identified in the model's response.
    Usage           *Usage                 `json:"usage,omitempty"`   // (Optional) Token usage statistics from the model response.
}
```

---

## Error Handling

The LLM Client includes robust error handling mechanisms to provide informative feedback during configuration validation, data processing, and LLM interactions.

### Parsing Errors:
These errors relate to issues in extracting the final answer from the LLM's response based on your `classification.parsing` configuration.

-   **Invalid Regex**: If a configured `answer_pattern` is syntactically incorrect (i.e., not a valid RE2 regex), the configuration validation will fail at application startup, preventing the process from beginning with a faulty pattern.
-   **No Capture Group**: If an `answer_pattern` does not contain exactly one capture group `(...)`, that specific pattern will be skipped during parsing. The parser will then proceed to the next pattern or the `find` list.
-   **No Matches**: If, after trying all `answer_patterns` and `find` strings, no valid answer can be extracted from the LLM's response, the `default` value (if configured) will be used as the `FinalAnswer`.
-   **Exception During Parsing**: If an unexpected internal error occurs during the parsing process (e.g., a critical type assertion fails), the `fallback` value (if configured) will be used as the `FinalAnswer`, ensuring a result is always recorded.

### Field Resolution Errors:
These errors pertain to how input data fields are mapped and accessed for prompt templating.

-   **Missing Field**: If a placeholder in your `user` prompt template (e.g., `{MY_FIELD}`) does not correspond to an existing field in the `DataRow.Data` map (your original input columns), the placeholder will remain unreplaced in the prompt sent to the LLM. This can lead to unexpected LLM behavior.
-   **Type Conversion**: Non-string values from `DataRow.Data` (e.g., numbers, booleans) used in templates or for `InputText` resolution are automatically converted to strings using Go's default formatting (`fmt.Sprintf("%v", value)`). While this prevents errors, ensure the default string representation is suitable for your prompt.
-   **Nested Field Not Found**: If a nested field specified using dot notation (e.g., `{parent.child}`) is not found within your `DataRow.Data`, that specific placeholder will also remain unreplaced in the prompt.

### Processing Errors:
These errors cover issues related to network communication with the LLM server or problems with the LLM's response itself.

-   **Network Timeout**: Requests to the LLM server will respect the `provider.timeout` value. If a request exceeds this configured duration, it will be cancelled, and an error will be recorded for that item.
-   **Server Error**: Errors returned directly by the LLM server (e.g., `invalid model`, `rate limiting`, `internal server error`) will be captured and recorded in the `Result.Error` field for the corresponding item.
-   **Invalid Response**: If the LLM returns a malformed or otherwise unparseable response (not related to `classification.parsing` rules, but fundamental structural issues), the `fallback` value will be used, and a relevant error message may be logged.
