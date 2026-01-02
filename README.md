# LLM Client

LLM Client is a versatile, high-performance command-line tool designed for batch processing, classification, and analysis tasks using Large Language Models. It supports multiple LLM serving backends like **vLLM**, **llama.cpp**, and **OpenAI**, providing a unified interface for complex workflows.

## Features

-   **Multiple Providers**: Seamlessly switch between `vllm`, `llamacpp`, and `openai` backends.
-   **Rich Parameter Support**: Extensive control over generation parameters for each provider.
-   **Concurrent Processing**: High-throughput processing with configurable worker pools.
-   **Consensus Mode**: Improve accuracy by making multiple requests per data item and finding the consensus answer.
-   **Advanced Parsing**: Sophisticated response parsing using regex, keywords, and custom mappings.
-   **Flexible I/O**: Supports `csv`, `json`, `xlsx`, and `parquet` for both input and output.
-   **Streaming Output**: Write results to disk as they are completed, ideal for long-running jobs.
-   **Live Metrics**: Monitor classification performance (Accuracy, F1-Score, Kappa) in real-time.
-   **Detailed Reporting**: Generate comprehensive analysis reports from your results.
-   **Resumability**: Automatically save state on cancellation (Ctrl+C) and resume interrupted jobs.
-   **Guided Generation**: Leverage provider-specific features like regex, grammar, and JSON-enforced output with vLLM.

## Installation (Recommended)

```bash
go install github.com/Vitruves/llm-client@latest
```

This will install the `llm-client` binary to your `$GOPATH/bin` directory.

## Manual Installation

```bash
git clone https://github.com/Vitruves/llm-client.git
cd llm-client
make install
```

This will build and install the binary to `~/.local/bin`. Alternatively, use `make build` to build without installing (outputs to `./bin/llm-client`).

## Quick Start

1.  **Create a Configuration File**: Start with the example `config.full.yaml` provided in the repository and adapt it to your needs.
2.  **Prepare Input Data**: Create an input file (e.g., `data.csv`) with the text you want to process.
3.  **Run the Client**:
    ```bash
    llm-client run -c config.yaml -i data.csv
    ```
4.  **Check the Output**: Results will be saved in the directory specified in your configuration (default: `./output`).

## Configuration (`config.yaml`)

The client is controlled by a single `config.yaml` file. Below is a comprehensive guide to all supported parameters.

---

### 1. `provider`

This section configures the connection to your LLM backend.

| Parameter | Type   | Description                                                                                             | Required | Recommended Value                                |
| :-------- | :----- | :------------------------------------------------------------------------------------------------------ | :------- | :----------------------------------------------- |
| `name`    | string | The LLM provider. Supported: `vllm`, `llamacpp`, `openai`.                                              | Yes      | `vllm`                                           |
| `base_url`| string | The base URL of the LLM server API. Not required for `openai`.                                            | Yes      | `http://localhost:8000` (for vLLM)               |
| `timeout` | string | The timeout for API requests (e.g., `60s`, `2m`).                                                       | No       | `120s` (Default: `60s`)                          |
| `api_key` | string | The API key for the provider. Required for `openai`. Can use environment variables (e.g., `${OPENAI_KEY}`). | If `openai` | `sk-...` or `${YOUR_ENV_VAR}`                    |

**Implementation:**

```yaml
provider:
  name: vllm
  base_url: http://localhost:8000
  timeout: 120s
  # api_key: ${OPENAI_API_KEY} # Only for openai provider
```

---

### 2. `model`

This section defines the model to use and its generation parameters.

| Parameter    | Type     | Description                                     | Required | Recommended Value                                         |
| :----------- | :------- | :---------------------------------------------- | :------- | :-------------------------------------------------------- |
| `name`       | string   | The name of the model to use for the requests.  | Yes      | `mistralai/Mistral-7B-Instruct-v0.2`                      |
| `parameters` | object   | A nested object containing all model parameters. | Yes      | See the detailed `parameters` section below.              |

---

### 3. `model.parameters`

This is the most critical section for controlling the LLM's output. Parameters are supported by different providers.

#### Common Sampling Parameters

| Parameter             | Type         | Description                                                                                             | Provider Support        | Recommended Value                               |
| :-------------------- | :----------- | :------------------------------------------------------------------------------------------------------ | :---------------------- | :---------------------------------------------- |
| `temperature`         | float        | Controls randomness. Lower is more deterministic. 0 disables sampling.                                  | `vllm`, `llamacpp`, `openai` | `0.7` (for creative tasks), `0.1` (for factual) |
| `max_tokens`          | int          | Maximum number of tokens to generate. Mapped to `n_predict` for `llamacpp`.                             | `vllm`, `llamacpp`, `openai` | `256`                                           |
| `min_tokens`          | int          | Minimum number of tokens to generate.                                                                   | `vllm`, `llamacpp`      | `0`                                             |
| `top_p`               | float        | Nucleus sampling: considers tokens with cumulative probability >= `top_p`.                              | `vllm`, `llamacpp`, `openai` | `0.9`                                           |
| `top_k`               | int          | Considers the `top_k` most likely tokens. `-1` disables it.                                             | `vllm`, `llamacpp`      | `40`                                            |
| `min_p`               | float        | Minimum probability for a token to be considered.                                                       | `vllm`, `llamacpp`      | `0.0`                                           |
| `repetition_penalty`  | float        | Penalizes repeating tokens. `1.0` means no penalty.                                                     | `vllm`, `llamacpp`      | `1.1`                                           |
| `presence_penalty`    | float        | Penalizes new tokens based on whether they appear in the text so far. (-2.0 to 2.0)                     | `vllm`, `llamacpp`, `openai` | `0.0`                                           |
| `frequency_penalty`   | float        | Penalizes new tokens based on their existing frequency in the text. (-2.0 to 2.0)                       | `vllm`, `llamacpp`, `openai` | `0.0`                                           |
| `seed`                | int          | Random seed for reproducible results.                                                                   | `vllm`, `llamacpp`, `openai` | `42`                                            |
| `n`                   | int          | Number of completions to generate for each prompt. Mapped to `n_choices` for `llamacpp`.              | `vllm`, `llamacpp`, `openai` | `1`                                             |
| `stop`                | list[string] | A list of strings that will stop generation.                                                            | `vllm`, `llamacpp`, `openai` | `["\n", "User:"]`                               |
| `stop_token_ids`      | list[int]    | A list of token IDs that will stop generation.                                                          | `vllm`                  | `[123, 456]`                                    |
| `ignore_eos`          | bool         | Ignore the End-Of-Stream token.                                                                         | `vllm`, `llamacpp`      | `false`                                         |
| `logprobs`            | int          | Number of log probabilities to return.                                                                  | `vllm`, `llamacpp`, `openai` | `null` (disabled)                               |
| `skip_special_tokens` | bool         | Whether to skip special tokens in the output.                                                           | `vllm`                  | `true`                                          |

#### vLLM-Specific Parameters

| Parameter                   | Type           | Description                                                                                             | Recommended Value                               |
| :-------------------------- | :------------- | :------------------------------------------------------------------------------------------------------ | :---------------------------------------------- |
| `enable_thinking`           | bool           | For models like Qwen2, enables the `<|think|>` block for chain-of-thought reasoning before the answer.    | `true`                                          |
| `guided_choice`             | list[string]   | Constrains the output to a predefined list of strings.                                                  | `["positive", "negative", "neutral"]`           |
| `guided_regex`              | string         | Enforces the output to follow a specific regular expression.                                            | `"(positive\|negative\|neutral)"`                |
| `guided_json`               | map[string]any | Enforces the output to be a JSON object matching a Pydantic or JSON schema.                             | `{ "type": "object", "properties": { ... } }`   |
| `guided_grammar`            | string         | Enforces the output to follow a specific context-free grammar (e.g., GBNF format).                      | (Path to or content of a `.gbnf` file)          |
| `guided_whitespace_pattern` | string         | A regex to handle whitespace during guided generation.                                                  | `\s*`                                           |
| `guided_decoding_backend`   | string         | The backend to use for guided generation.                                                               | `outlines`                                      |
| `use_beam_search`           | bool           | Use beam search for generation instead of sampling.                                                     | `false`                                         |
| `best_of`                   | int            | Generates `best_of` sequences and returns the one with the highest log probability.                     | `1`                                             |
| `length_penalty`            | float          | Penalty for longer sequences, used with beam search.                                                    | `1.0`                                           |
| `early_stopping`            | bool           | Controls stopping criteria for beam search.                                                             | `false`                                         |
| `bad_words`                 | list[string]   | A list of words to prevent from being generated.                                                        | `[]`                                            |
| `include_stop_str_in_output`| bool           | Whether to include the stop string in the output.                                                       | `false`                                         |
| `prompt_logprobs`           | int            | Number of log probabilities to return for the prompt.                                                   | `null`                                          |
| `truncate_prompt_tokens`    | int            | If the prompt is too long, it will be truncated to this length.                                         | `null`                                          |
| `spaces_between_special_tokens` | bool       | Whether to add a space between special tokens.                                                          | `true`                                          |
| `max_logprobs`              | int            | Alias for `logprobs`. Sets the number of log probabilities to return.                                     | `null`                                          |
| `echo`                      | bool           | If `true`, the prompt is included in the beginning of the completion.                                   | `false`                                         |

#### llama.cpp-Specific Parameters

| Parameter       | Type    | Description                                                                                             | Recommended Value |
| :-------------- | :------ | :------------------------------------------------------------------------------------------------------ | :---------------- |
| `mirostat`      | int     | Enable Mirostat sampling (0: disabled, 1: Mirostat, 2: Mirostat v2).                                    | `0`               |
| `mirostat_tau`  | float   | Mirostat target entropy.                                                                                | `5.0`             |
| `mirostat_eta`  | float   | Mirostat learning rate.                                                                                 | `0.1`             |
| `tfs_z`         | float   | Tail Free Sampling `z` parameter.                                                                       | `1.0`             |
| `typical_p`     | float   | Locally Typical Sampling `p` parameter.                                                                 | `1.0`             |
| `n_keep`        | int     | Number of tokens from the prompt to keep when the context window is exceeded.                           | `0` (none)        |
| `penalize_nl`   | bool    | Penalize newline characters.                                                                            | `false`           |
| `chat_format`   | string  | The chat template to use (e.g., `chatml`, `llama2`). Enables chat completion endpoint.                  | `chatml`          |

---

### 4. `classification`

This section controls how the prompt is constructed and how the model's response is parsed.

| Parameter         | Type     | Description                                                                                                                               | Required | Recommended Value                                                              |
| :---------------- | :------- | :---------------------------------------------------------------------------------------------------------------------------------------- | :------- | :----------------------------------------------------------------------------- |
| `template.system` | string   | The system prompt.                                                                                                                        | Yes      | `You are an expert sentiment classifier.`                                      |
| `template.user`   | string   | The user prompt template. Use placeholders like `{text}` or `{column_name}` to insert data from your input file.                            | Yes      | `Classify the following review: {review_text}`                                 |
| `parsing`         | object   | A nested object defining how to extract the final answer from the raw LLM response.                                                       | Yes      | See `parsing` section below.                                                   |
| `field_mapping`   | object   | (Optional) Maps input file columns to placeholders in the user template.                                                                  | No       | See `field_mapping` section below.                                             |

#### 4.1 `classification.parsing`

Defines how to extract the final answer from the LLM's raw response. The parsing logic follows this order: **`answer_patterns`** -> **`find`** -> **`map`**.

| Parameter          | Type          | Description                                                                                                                            | Required | Recommended Value                                                              |
| :----------------- | :------------ | :------------------------------------------------------------------------------------------------------------------------------------- | :------- | :----------------------------------------------------------------------------- |
| `find`             | list[string]  | A list of keywords to search for in the response. The first one found is returned as the answer. Case-insensitive by default.            | Yes      | `["positive", "negative", "neutral"]`                                          |
| `default`          | string        | The value to return if no keywords from `find` or patterns are matched.                                                              | Yes      | `unknown`                                                                      |
| `fallback`         | string        | A value to use if the entire parsing process fails due to an internal error (e.g., invalid regex).                                     | No       | `error_parsing`                                                                |
| `map`              | map[string]string | A key-value map to translate a found answer into a final value (e.g., `{"pos": "positive", "neg": "negative"}`).                     | No       | `{"1": "positive", "0": "negative"}`                                           |
| `thinking_tags`    | string        | The start and end tags for the model's chain-of-thought reasoning (e.g., `<think></think>`). This content is extracted separately.        | No       | `<think></think>`                                                               |
| `preserve_thinking`| bool          | If `true`, the thinking block is not removed from the raw response before parsing for the final answer.                                  | No       | `false`                                                                        |
| `answer_patterns`  | list[string]  | A list of regular expressions to find the answer. The first capturing group `(...)` of the first matching pattern is used. Takes precedence over `find`. | No       | `["Answer: (.*)"]`                                                             |
| `case_sensitive`   | bool          | If `true`, the `find` and `answer_patterns` matching will be case-sensitive.                                                           | No       | `false`                                                                        |
| `exact_match`      | bool          | If `true`, requires the entire response to match a keyword in `find` (instead of just containing it).                                    | No       | `false`                                                                        |

#### 4.2 `classification.field_mapping`

Provides explicit control over how input data fields are mapped to prompt template placeholders.

| Parameter          | Type          | Description                                                                                                                            | Required | Recommended Value                                                              |
| :----------------- | :------------ | :------------------------------------------------------------------------------------------------------------------------------------- | :------- | :----------------------------------------------------------------------------- |
| `input_text_field` | string        | The name of the column from the input file that should be considered the primary "input text" for reporting and output.                  | No       | `review_text`                                                                  |
| `placeholder_map`  | map[string]string | Explicitly maps a placeholder in the template (e.g., `{my_placeholder}`) to a column name in the input file (e.g., `column_from_file`). | No       | `{ "product_name": "product" }`                                                |

---

### 5. `processing`

This section configures the execution of the classification job.

| Parameter        | Type   | Description                                                                                                                              | Required | Recommended Value                                                              |
| :--------------- | :----- | :--------------------------------------------------------------------------------------------------------------------------------------- | :------- | :----------------------------------------------------------------------------- |
| `workers`        | int    | The number of concurrent workers making requests to the LLM.                                                                             | No       | `16` (Default: `4`)                                                            |
| `batch_size`     | int    | The size of batches for processing (future use).                                                                                         | No       | `1` (Default: `1`)                                                             |
| `repeat`         | int    | Number of times to repeat the request for each data item to find a consensus answer. Must be 1-10.                                         | No       | `3` (Default: `1`)                                                             |
| `rate_limit`     | bool   | Enables a client-side rate limiter to avoid overwhelming the server. Recommended for `vllm`.                                             | No       | `true`                                                                         |
| `flashinfer_safe`| bool   | If `true`, disables the `seed` parameter when using vLLM to ensure compatibility with FlashInfer.                                          | No       | `false`                                                                        |
| `live_metrics`   | object | Configuration for displaying live performance metrics during processing.                                                                 | No       | See `live_metrics` section below.                                              |
| `minimal_mode`   | bool   | If `true`, skips all parsing and consensus logic. The raw response from the LLM is treated as the final answer. Useful for pure generation. | No       | `false`                                                                        |

#### 5.1 `processing.live_metrics`

| Parameter     | Type         | Description                                                                                                                            | Required | Recommended Value                                                              |
| :------------ | :----------- | :------------------------------------------------------------------------------------------------------------------------------------- | :------- | :----------------------------------------------------------------------------- |
| `enabled`     | bool         | Set to `true` to enable live metrics in the progress bar.                                                                              | Yes      | `true`                                                                         |
| `metric`      | string       | The metric to display. Supported: `accuracy`, `f1`, `kappa`.                                                                           | Yes      | `f1`                                                                           |
| `ground_truth`| string       | The name of the column in your input file that contains the ground truth (correct) labels.                                             | Yes      | `label`                                                                        |
| `average`     | string       | The averaging method for F1-score. Supported: `macro`, `micro`, `weighted`.                                                            | No       | `macro`                                                                        |
| `classes`     | list[string] | An explicit list of all possible classes. If not provided, it's inferred from `classification.parsing.find`.                             | No       | `["positive", "negative", "neutral"]`                                          |

---

### 6. `output`

This section configures how the results are saved. The `input_text_field` here is a fallback for the one in `classification.field_mapping`.

| Parameter            | Type   | Description                                                                                             | Required | Recommended Value                                                              |
| :------------------- | :----- | :------------------------------------------------------------------------------------------------------ | :------- | :----------------------------------------------------------------------------- |
| `directory`          | string | The directory where output files will be saved.                                                         | No       | `./output` (Default)                                                           |
| `format`             | string | The output file format. Supported: `json`, `csv`, `parquet`, `xlsx`.                                      | No       | `json` (Default)                                                               |
| `input_text_field`   | string | A fallback column from the original data to use as `input_text` in the output if not specified elsewhere. | No       | `review`                                                                       |
| `include_raw_response`| bool   | If `true`, the full, unparsed response from the LLM is included in the output.                           | No       | `true`                                                                         |
| `include_thinking`   | bool   | If `true`, the extracted thinking content is included in a separate field in the output.                  | No       | `true`                                                                         |
| `stream_output`      | bool   | If `true`, results are written to the output file as they are completed, instead of all at the end.       | No       | `false`                                                                        |
| `stream_save_every`  | int    | When `stream_output` is true, forces a flush to disk every N results. `1` is recommended for safety.      | No       | `1`                                                                            |

---

### 7. `reference`

(Optional) This section is used by the `report` command to compare results against a ground truth file.

| Parameter     | Type   | Description                                                                                             | Required | Recommended Value                                                              |
| :------------ | :----- | :------------------------------------------------------------------------------------------------------ | :------- | :----------------------------------------------------------------------------- |
| `file`        | string | Path to the reference/ground truth file.                                                                | No       | `data/ground_truth.csv`                                                        |
| `column`      | string | The column in the reference file containing the ground truth labels.                                    | No       | `correct_label`                                                                |
| `format`      | string | The format of the reference file (`csv`, `json`, etc.).                                                 | No       | `csv`                                                                          |
| `index_column`| string | The column in the reference file used to match rows with the input data (e.g., a unique ID).              | No       | `id`                                                                           |

## Command-Line Interface (CLI)

The client provides several commands:

-   `llm-client run`: The main command to process data.
    -   `-c, --config`: Path to the configuration file.
    -   `-i, --input`: Path to the input data file.
    -   `-o, --output`: Override the output directory from the config.
    -   `-w, --workers`: Override the number of workers.
    -   `-r, --repeat`: Override the repeat count for consensus mode.
    -   `-l, --limit`: Limit processing to the first N rows.
    -   `-v, --verbose`: Enable detailed logging.
    -   `--resume`: Resume a previously cancelled job from a state file.
    -   `--temperature`, `--max-tokens`, etc.: Override any model parameter from the command line.
-   `llm-client report analyze [result-file]`: Generate a detailed performance and accuracy report from a results file.
-   `llm-client report compare [file1] [file2]`: Compare two result files side-by-side.
-   `llm-client health`: Check the status of the configured LLM server.
-   `llm-client config validate [config-file]`: Validate the syntax and logic of a configuration file.

## Data Structures

-   **`DataRow`**: Represents one row from your input file. Contains `Index` (int), `Text` (string), and `Data` (map[string]interface{} for all original columns).
-   **`Result`**: The output for one processed row. Contains `Index`, `InputText`, `OriginalData`, `GroundTruth`, `FinalAnswer`, `RawResponse`, `ThinkingContent`, `Success`, `Error`, `ResponseTime`, and optional `Attempts`, `Consensus`, `ToolCalls`, and `Usage` data.

## Error Handling

-   **Configuration Validation**: The client validates the `config.yaml` on startup and will exit with a clear error if any required fields are missing or values are invalid.
-   **Processing Errors**: Network errors, timeouts, or server-side errors are caught per-item. The `Success` field in the result will be `false`, and the `Error` field will contain the error message. The process will continue with the next item.
-   **Parsing Errors**: If a response cannot be parsed according to your rules, the `default` value from the parsing config is used. If the parsing logic itself fails, the `fallback` value is used.