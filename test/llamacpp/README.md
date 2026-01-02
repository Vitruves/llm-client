# LLM Client - llama.cpp Integration Tests

This directory contains comprehensive integration tests for the llama.cpp backend of `llm-client`. These tests require a running llama.cpp server with the Qwen3-8B model.

## Prerequisites

1. **llama.cpp server** running with the Qwen3 model:
   ```bash
   llama-server -m Qwen_Qwen3-8B-Q5_K_M.gguf --port 8080
   ```

2. **Go 1.22+** installed

## Quick Start

```bash
# Run all tests
go test ./test/llamacpp/... -v

# Run with custom server URL
LLAMACPP_URL=http://myserver:8080 go test ./test/llamacpp/... -v

# Skip tests (no server available)
LLAMACPP_SKIP=1 go test ./test/llamacpp/...
```

## Test Structure

```
test/llamacpp/
├── README.md                 # This file
├── llamacpp_test.go          # Client-level integration tests
├── pipeline_test.go          # Full pipeline integration tests
├── configs/                  # Test configuration files
│   ├── chat_basic.yaml
│   ├── chat_deterministic.yaml
│   ├── classification.yaml
│   ├── completion_basic.yaml
│   ├── sampling_advanced.yaml
│   ├── sampling_mirostat.yaml
│   ├── stop_sequences.yaml
│   ├── thinking_mode.yaml
│   ├── pipeline_sentiment.yaml
│   ├── pipeline_topics.yaml
│   ├── pipeline_qa.yaml
│   ├── pipeline_translation.yaml
│   ├── pipeline_summarization.yaml
│   ├── pipeline_code_review.yaml
│   ├── pipeline_consensus.yaml
│   ├── pipeline_streaming.yaml
│   ├── pipeline_thinking.yaml
│   ├── pipeline_parquet_output.yaml
│   └── pipeline_csv_output.yaml
└── data/                     # Test datasets
    ├── sentiment.csv         # 10 sentiment analysis samples
    ├── questions.csv         # 5 Q&A samples
    ├── topics.csv            # 5 topic classification samples
    ├── translation.csv       # 5 translation samples
    ├── summarization.csv     # 3 article summarization samples
    └── code_review.csv       # 5 code explanation samples
```

## Test Categories

### Client-Level Tests (`llamacpp_test.go`)

| Test | Description |
|------|-------------|
| `TestHealthCheck` | Verifies server health endpoint |
| `TestGetServerInfo` | Retrieves server configuration and model info |
| `TestChatBasic` | Basic chat completion request |
| `TestChatDeterministic` | Reproducible outputs with temperature=0 and seed |
| `TestChatMultiTurn` | Multi-turn conversation context retention |
| `TestCompletionBasic` | Non-chat completion endpoint |
| `TestSamplingMirostat` | Mirostat sampling algorithm |
| `TestSamplingAdvanced` | Advanced sampling parameters (top_p, top_k, etc.) |
| `TestSamplingWithExplicitParams` | Override config params per-request |
| `TestClassificationPositive` | Sentiment classification (positive) |
| `TestClassificationNegative` | Sentiment classification (negative) |
| `TestClassificationNeutral` | Sentiment classification (neutral) |
| `TestStopSequences` | Early termination with stop strings |
| `TestThinkingMode` | Qwen3 thinking mode with `<think>` tags |
| `TestResponseTime` | Response time tracking |
| `TestUsageStats` | Token usage statistics |
| `TestFinishReason` | Finish reason reporting |
| `TestMaxTokensLimit` | Token limit truncation |
| `TestContextCancellation` | Request cancellation handling |
| `TestConcurrentRequests` | Parallel request processing |
| `BenchmarkChatRequest` | Performance benchmarking |

### Pipeline Tests (`pipeline_test.go`)

| Test | Description |
|------|-------------|
| `TestPipelineDataLoading` | CSV data loading validation |
| `TestPipelineSentimentClassification` | End-to-end sentiment analysis with metrics |
| `TestPipelineTopicClassification` | Topic classification with F1 score |
| `TestPipelineQuestionAnswering` | Question answering pipeline |
| `TestPipelineTranslation` | Multi-language translation |
| `TestPipelineSummarization` | Text summarization |
| `TestPipelineCodeReview` | Code explanation generation |
| `TestPipelineConsensusMode` | Multiple attempts per item with consensus |
| `TestPipelineCSVOutput` | CSV output format |
| `TestPipelineParquetOutput` | Parquet output format |
| `TestPipelineStreamingOutput` | Real-time streaming output |
| `TestPipelineThinkingMode` | Pipeline with thinking content capture |
| `TestPipelineInvalidConfig` | Error handling for missing config |
| `TestPipelineInvalidInputFile` | Error handling for missing input |
| `TestPipelineContextCancellation` | Pipeline cancellation |
| `TestPipelineConcurrentWorkers` | Multi-worker processing |
| `TestPipelineAccuracyMetrics` | Live accuracy calculation |
| `TestPipelineConfigInfoFile` | Config/server info file generation |
| `BenchmarkPipelineSentiment` | Pipeline performance benchmarking |

## Running Specific Tests

```bash
# Client tests only
go test ./test/llamacpp/... -v -run "^Test[^P]"

# Pipeline tests only
go test ./test/llamacpp/... -v -run Pipeline

# Classification tests only
go test ./test/llamacpp/... -v -run Classification

# Sampling tests only
go test ./test/llamacpp/... -v -run Sampling

# Run benchmarks
go test ./test/llamacpp/... -bench=. -benchtime=5s
```

## Configuration Files

### Client Configs

| Config | Purpose |
|--------|---------|
| `chat_basic.yaml` | Standard chat with temperature=0.7 |
| `chat_deterministic.yaml` | Reproducible output (temp=0, seed=42) |
| `classification.yaml` | Sentiment classification |
| `completion_basic.yaml` | Non-chat completion mode |
| `sampling_mirostat.yaml` | Mirostat v2 sampling |
| `sampling_advanced.yaml` | All advanced sampling params |
| `stop_sequences.yaml` | Custom stop sequences |
| `thinking_mode.yaml` | Qwen3 thinking mode enabled |

### Pipeline Configs

| Config | Purpose |
|--------|---------|
| `pipeline_sentiment.yaml` | Sentiment classification with accuracy metrics |
| `pipeline_topics.yaml` | Topic classification with F1 metrics |
| `pipeline_qa.yaml` | Question answering (minimal mode) |
| `pipeline_translation.yaml` | Translation with field mapping |
| `pipeline_summarization.yaml` | Text summarization |
| `pipeline_code_review.yaml` | Code explanation |
| `pipeline_consensus.yaml` | 3x repeat with consensus voting |
| `pipeline_streaming.yaml` | Stream results to disk |
| `pipeline_thinking.yaml` | Capture thinking content |
| `pipeline_parquet_output.yaml` | Parquet format output |
| `pipeline_csv_output.yaml` | CSV format output |

## Test Datasets

### sentiment.csv
10 labeled samples for sentiment classification:
- 4 positive examples
- 3 negative examples
- 3 neutral examples

### questions.csv
5 question-answer pairs covering:
- Factual questions (capitals, authors)
- Procedural questions (how-to)
- Explanatory questions (why)
- Mathematical questions

### topics.csv
5 text samples for topic classification:
- Business, Science, Sports, Technology categories

### translation.csv
5 English sentences with target languages:
- French, Spanish, German, Italian, Portuguese

### summarization.csv
3 multi-paragraph articles on:
- Artificial Intelligence
- Climate Change
- Neuroscience

### code_review.csv
5 code snippets in different languages:
- Python, JavaScript, Java, Rust, Go

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLAMACPP_URL` | Server URL | `http://localhost:8080` |
| `LLAMACPP_SKIP` | Skip tests if set to `1` | (not set) |

## Expected Test Duration

| Test Suite | Approximate Duration |
|------------|---------------------|
| Client tests | ~2 minutes |
| Pipeline tests | ~5 minutes |
| Full suite | ~7 minutes |

*Durations vary based on model and hardware.*

## Troubleshooting

### Tests skip with "server not available"
- Ensure llama.cpp server is running on port 8080
- Check `LLAMACPP_URL` if using non-default port
- Verify server health: `curl http://localhost:8080/health`

### Classification tests show unexpected answers
- Qwen3 uses thinking mode by default
- Tests use `/no_think` suffix to disable thinking
- Check `max_tokens` is sufficient for complete responses

### Pipeline tests timeout
- Increase timeout in test code if needed
- Reduce `workers` count if server is overloaded
- Check server logs for errors

### Output files not created
- Verify `test_output/` directory permissions
- Check for disk space issues
- Review test logs for write errors

## Adding New Tests

1. **Client test**: Add to `llamacpp_test.go`
   - Use `skipIfNoServer(t)` helper
   - Use `loadConfig(t, "config_name.yaml")` for config

2. **Pipeline test**: Add to `pipeline_test.go`
   - Use `setupPipelineTest(t)` for setup/cleanup
   - Use `loadPipelineConfig(t, "config_name.yaml")`

3. **New config**: Add YAML to `configs/`
   - Follow existing naming conventions
   - Include all required fields

4. **New test data**: Add CSV to `data/`
   - Include header row
   - Use consistent column naming
