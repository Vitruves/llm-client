// Package llamacpp_test provides pipeline integration tests for the llm-client.
// These tests exercise the full pipeline: config loading → data loading → processing → output.
//
// To run these tests:
//   1. Start llama.cpp server: llama-server -m Qwen_Qwen3-8B-Q5_K_M.gguf --port 8080
//   2. Run tests: go test ./test/llamacpp/... -v -run Pipeline
//
// Set LLAMACPP_URL environment variable to override the default server URL.
// Set LLAMACPP_SKIP=1 to skip these tests entirely.
package llamacpp_test

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/Vitruves/llm-client/internal/client"
	"github.com/Vitruves/llm-client/internal/config"
	"github.com/Vitruves/llm-client/internal/loader"
	"github.com/Vitruves/llm-client/internal/models"
	"github.com/Vitruves/llm-client/internal/processor"
)

const (
	dataDir   = "data"
	outputDir = "./test_output"
)

// setupPipelineTest prepares the test environment
func setupPipelineTest(t *testing.T) func() {
	t.Helper()
	skipIfNoServer(t)

	// Create output directory
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		t.Fatalf("Failed to create output directory: %v", err)
	}

	// Return cleanup function
	return func() {
		os.RemoveAll(outputDir)
	}
}

// loadPipelineConfig loads a pipeline config and overrides server URL and output dir
func loadPipelineConfig(t *testing.T, configFile string) *models.Config {
	t.Helper()

	configPath := filepath.Join(configDir, configFile)
	cfg, err := config.Load(configPath)
	if err != nil {
		t.Fatalf("Failed to load config %s: %v", configPath, err)
	}

	cfg.Provider.BaseURL = serverURL
	cfg.Output.Directory = outputDir
	return cfg
}

// =============================================================================
// Data Loading Tests
// =============================================================================

func TestPipelineDataLoading(t *testing.T) {
	// Test CSV loading
	t.Run("LoadSentimentCSV", func(t *testing.T) {
		dataPath := filepath.Join(dataDir, "sentiment.csv")
		data, err := loader.LoadData(dataPath)
		if err != nil {
			t.Fatalf("Failed to load sentiment data: %v", err)
		}

		if len(data) != 10 {
			t.Errorf("Expected 10 rows, got %d", len(data))
		}

		// Verify first row
		if data[0].Data["id"] != "1" && data[0].Data["id"] != 1 {
			t.Errorf("Expected first row id=1, got %v", data[0].Data["id"])
		}

		// Verify label column exists
		if _, ok := data[0].Data["label"]; !ok {
			t.Error("Expected 'label' column in data")
		}

		t.Logf("Loaded %d sentiment rows", len(data))
	})

	t.Run("LoadQuestionsCSV", func(t *testing.T) {
		dataPath := filepath.Join(dataDir, "questions.csv")
		data, err := loader.LoadData(dataPath)
		if err != nil {
			t.Fatalf("Failed to load questions data: %v", err)
		}

		if len(data) != 5 {
			t.Errorf("Expected 5 rows, got %d", len(data))
		}

		t.Logf("Loaded %d question rows", len(data))
	})

	t.Run("LoadTopicsCSV", func(t *testing.T) {
		dataPath := filepath.Join(dataDir, "topics.csv")
		data, err := loader.LoadData(dataPath)
		if err != nil {
			t.Fatalf("Failed to load topics data: %v", err)
		}

		if len(data) != 5 {
			t.Errorf("Expected 5 rows, got %d", len(data))
		}

		t.Logf("Loaded %d topic rows", len(data))
	})
}

// =============================================================================
// Sentiment Classification Pipeline Tests
// =============================================================================

func TestPipelineSentimentClassification(t *testing.T) {
	cleanup := setupPipelineTest(t)
	defer cleanup()

	cfg := loadPipelineConfig(t, "pipeline_sentiment.yaml")
	proc := processor.New(cfg)

	dataPath := filepath.Join(dataDir, "sentiment.csv")
	opts := processor.Options{
		Limit:        5, // Process first 5 items for faster testing
		ShowProgress: false,
		Verbose:      true,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	err := proc.ProcessFile(ctx, dataPath, opts)
	if err != nil {
		t.Fatalf("Pipeline processing failed: %v", err)
	}

	// Verify output files were created
	files, err := os.ReadDir(outputDir)
	if err != nil {
		t.Fatalf("Failed to read output directory: %v", err)
	}

	var resultFile string
	for _, f := range files {
		if strings.HasPrefix(f.Name(), "results_") && strings.HasSuffix(f.Name(), ".json") {
			resultFile = filepath.Join(outputDir, f.Name())
			break
		}
	}

	if resultFile == "" {
		t.Fatal("No results file found in output directory")
	}

	// Load and verify results
	resultData, err := os.ReadFile(resultFile)
	if err != nil {
		t.Fatalf("Failed to read results file: %v", err)
	}

	var results map[string]interface{}
	if err := json.Unmarshal(resultData, &results); err != nil {
		t.Fatalf("Failed to parse results JSON: %v", err)
	}

	// Verify structure
	if _, ok := results["results"]; !ok {
		t.Error("Results file missing 'results' field")
	}

	if _, ok := results["summary"]; !ok {
		t.Error("Results file missing 'summary' field")
	}

	// Check summary
	summary := results["summary"].(map[string]interface{})
	total := int(summary["total"].(float64))
	success := int(summary["success"].(float64))

	t.Logf("Pipeline results: %d total, %d successful", total, success)

	if total != 5 {
		t.Errorf("Expected 5 total results, got %d", total)
	}

	if success == 0 {
		t.Error("No successful results")
	}

	// Verify individual results
	resultsList := results["results"].([]interface{})
	for i, r := range resultsList {
		result := r.(map[string]interface{})
		finalAnswer := result["final_answer"].(string)

		// Check that answers are one of the expected values
		validAnswers := map[string]bool{"positive": true, "negative": true, "neutral": true, "unknown": true}
		if !validAnswers[strings.ToLower(finalAnswer)] {
			t.Logf("Result %d has unexpected answer: %s", i, finalAnswer)
		}
	}
}

// =============================================================================
// Topic Classification Pipeline Tests
// =============================================================================

func TestPipelineTopicClassification(t *testing.T) {
	cleanup := setupPipelineTest(t)
	defer cleanup()

	cfg := loadPipelineConfig(t, "pipeline_topics.yaml")
	proc := processor.New(cfg)

	dataPath := filepath.Join(dataDir, "topics.csv")
	opts := processor.Options{
		Limit:        5,
		ShowProgress: false,
		Verbose:      true,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	err := proc.ProcessFile(ctx, dataPath, opts)
	if err != nil {
		t.Fatalf("Pipeline processing failed: %v", err)
	}

	t.Log("Topic classification pipeline completed successfully")
}

// =============================================================================
// Question Answering Pipeline Tests
// =============================================================================

func TestPipelineQuestionAnswering(t *testing.T) {
	cleanup := setupPipelineTest(t)
	defer cleanup()

	cfg := loadPipelineConfig(t, "pipeline_qa.yaml")
	proc := processor.New(cfg)

	dataPath := filepath.Join(dataDir, "questions.csv")
	opts := processor.Options{
		Limit:        3,
		ShowProgress: false,
		Verbose:      true,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	err := proc.ProcessFile(ctx, dataPath, opts)
	if err != nil {
		t.Fatalf("Pipeline processing failed: %v", err)
	}

	// Find and verify results
	files, _ := os.ReadDir(outputDir)
	for _, f := range files {
		if strings.HasPrefix(f.Name(), "results_") && strings.HasSuffix(f.Name(), ".json") {
			resultFile := filepath.Join(outputDir, f.Name())
			resultData, _ := os.ReadFile(resultFile)

			var results map[string]interface{}
			json.Unmarshal(resultData, &results)

			resultsList := results["results"].([]interface{})
			for i, r := range resultsList {
				result := r.(map[string]interface{})
				answer := result["final_answer"].(string)
				t.Logf("Q%d answer: %s", i+1, truncateString(answer, 100))
			}
			break
		}
	}
}

// =============================================================================
// Translation Pipeline Tests
// =============================================================================

func TestPipelineTranslation(t *testing.T) {
	cleanup := setupPipelineTest(t)
	defer cleanup()

	cfg := loadPipelineConfig(t, "pipeline_translation.yaml")
	proc := processor.New(cfg)

	dataPath := filepath.Join(dataDir, "translation.csv")
	opts := processor.Options{
		Limit:        3,
		ShowProgress: false,
		Verbose:      true,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	err := proc.ProcessFile(ctx, dataPath, opts)
	if err != nil {
		t.Fatalf("Pipeline processing failed: %v", err)
	}

	t.Log("Translation pipeline completed successfully")
}

// =============================================================================
// Summarization Pipeline Tests
// =============================================================================

func TestPipelineSummarization(t *testing.T) {
	cleanup := setupPipelineTest(t)
	defer cleanup()

	cfg := loadPipelineConfig(t, "pipeline_summarization.yaml")
	proc := processor.New(cfg)

	dataPath := filepath.Join(dataDir, "summarization.csv")
	opts := processor.Options{
		Limit:        2,
		ShowProgress: false,
		Verbose:      true,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	err := proc.ProcessFile(ctx, dataPath, opts)
	if err != nil {
		t.Fatalf("Pipeline processing failed: %v", err)
	}

	t.Log("Summarization pipeline completed successfully")
}

// =============================================================================
// Code Review Pipeline Tests
// =============================================================================

func TestPipelineCodeReview(t *testing.T) {
	cleanup := setupPipelineTest(t)
	defer cleanup()

	cfg := loadPipelineConfig(t, "pipeline_code_review.yaml")
	proc := processor.New(cfg)

	dataPath := filepath.Join(dataDir, "code_review.csv")
	opts := processor.Options{
		Limit:        3,
		ShowProgress: false,
		Verbose:      true,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	err := proc.ProcessFile(ctx, dataPath, opts)
	if err != nil {
		t.Fatalf("Pipeline processing failed: %v", err)
	}

	t.Log("Code review pipeline completed successfully")
}

// =============================================================================
// Consensus Mode Pipeline Tests
// =============================================================================

func TestPipelineConsensusMode(t *testing.T) {
	cleanup := setupPipelineTest(t)
	defer cleanup()

	cfg := loadPipelineConfig(t, "pipeline_consensus.yaml")
	proc := processor.New(cfg)

	dataPath := filepath.Join(dataDir, "sentiment.csv")
	opts := processor.Options{
		Limit:        3, // Fewer items since each gets 3 requests
		ShowProgress: false,
		Verbose:      true,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	err := proc.ProcessFile(ctx, dataPath, opts)
	if err != nil {
		t.Fatalf("Pipeline processing failed: %v", err)
	}

	// Find and verify consensus results
	files, _ := os.ReadDir(outputDir)
	for _, f := range files {
		if strings.HasPrefix(f.Name(), "results_") && strings.HasSuffix(f.Name(), ".json") {
			resultFile := filepath.Join(outputDir, f.Name())
			resultData, _ := os.ReadFile(resultFile)

			var results map[string]interface{}
			json.Unmarshal(resultData, &results)

			// Check consensus stats in summary
			summary := results["summary"].(map[string]interface{})
			if consensusStats, ok := summary["consensus_stats"]; ok {
				stats := consensusStats.(map[string]interface{})
				t.Logf("Consensus stats: %v", stats)
			}

			// Check individual results for consensus info
			resultsList := results["results"].([]interface{})
			for i, r := range resultsList {
				result := r.(map[string]interface{})
				if consensus, ok := result["consensus"]; ok && consensus != nil {
					cons := consensus.(map[string]interface{})
					t.Logf("Result %d consensus: %s (%.1f%% ratio)",
						i, cons["final_answer"], cons["ratio"].(float64)*100)
				}
			}
			break
		}
	}
}

// =============================================================================
// Output Format Tests
// =============================================================================

func TestPipelineCSVOutput(t *testing.T) {
	cleanup := setupPipelineTest(t)
	defer cleanup()

	cfg := loadPipelineConfig(t, "pipeline_csv_output.yaml")
	proc := processor.New(cfg)

	dataPath := filepath.Join(dataDir, "sentiment.csv")
	opts := processor.Options{
		Limit:        3,
		ShowProgress: false,
		Verbose:      false,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	err := proc.ProcessFile(ctx, dataPath, opts)
	if err != nil {
		t.Fatalf("Pipeline processing failed: %v", err)
	}

	// Verify CSV output exists
	files, _ := os.ReadDir(outputDir)
	var csvFound bool
	for _, f := range files {
		if strings.HasPrefix(f.Name(), "results_") && strings.HasSuffix(f.Name(), ".csv") {
			csvFound = true
			t.Logf("CSV output created: %s", f.Name())
			break
		}
	}

	if !csvFound {
		t.Error("CSV output file not found")
	}
}

func TestPipelineParquetOutput(t *testing.T) {
	cleanup := setupPipelineTest(t)
	defer cleanup()

	cfg := loadPipelineConfig(t, "pipeline_parquet_output.yaml")
	proc := processor.New(cfg)

	dataPath := filepath.Join(dataDir, "sentiment.csv")
	opts := processor.Options{
		Limit:        3,
		ShowProgress: false,
		Verbose:      false,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	err := proc.ProcessFile(ctx, dataPath, opts)
	if err != nil {
		t.Fatalf("Pipeline processing failed: %v", err)
	}

	// Verify Parquet output exists
	files, _ := os.ReadDir(outputDir)
	var parquetFound bool
	for _, f := range files {
		if strings.HasPrefix(f.Name(), "results_") && strings.HasSuffix(f.Name(), ".parquet") {
			parquetFound = true
			t.Logf("Parquet output created: %s", f.Name())
			break
		}
	}

	if !parquetFound {
		t.Error("Parquet output file not found")
	}
}

// =============================================================================
// Streaming Output Tests
// =============================================================================

func TestPipelineStreamingOutput(t *testing.T) {
	cleanup := setupPipelineTest(t)
	defer cleanup()

	cfg := loadPipelineConfig(t, "pipeline_streaming.yaml")
	proc := processor.New(cfg)

	dataPath := filepath.Join(dataDir, "sentiment.csv")
	opts := processor.Options{
		Limit:        5,
		ShowProgress: false,
		Verbose:      false,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	err := proc.ProcessFile(ctx, dataPath, opts)
	if err != nil {
		t.Fatalf("Pipeline processing failed: %v", err)
	}

	t.Log("Streaming output pipeline completed")
}

// =============================================================================
// Thinking Mode Pipeline Tests
// =============================================================================

func TestPipelineThinkingMode(t *testing.T) {
	cleanup := setupPipelineTest(t)
	defer cleanup()

	cfg := loadPipelineConfig(t, "pipeline_thinking.yaml")
	proc := processor.New(cfg)

	dataPath := filepath.Join(dataDir, "questions.csv")
	opts := processor.Options{
		Limit:        2,
		ShowProgress: false,
		Verbose:      true,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	err := proc.ProcessFile(ctx, dataPath, opts)
	if err != nil {
		t.Fatalf("Pipeline processing failed: %v", err)
	}

	// Verify thinking content is captured
	files, _ := os.ReadDir(outputDir)
	for _, f := range files {
		if strings.HasPrefix(f.Name(), "results_") && strings.HasSuffix(f.Name(), ".json") {
			resultFile := filepath.Join(outputDir, f.Name())
			resultData, _ := os.ReadFile(resultFile)

			var results map[string]interface{}
			json.Unmarshal(resultData, &results)

			resultsList := results["results"].([]interface{})
			for i, r := range resultsList {
				result := r.(map[string]interface{})
				if thinking, ok := result["thinking_content"]; ok && thinking != "" {
					t.Logf("Result %d has thinking content: %s",
						i, truncateString(thinking.(string), 100))
				}
			}
			break
		}
	}
}

// =============================================================================
// Error Handling Tests
// =============================================================================

func TestPipelineInvalidConfig(t *testing.T) {
	_, err := config.Load("nonexistent_config.yaml")
	if err == nil {
		t.Error("Expected error loading nonexistent config")
	}
}

func TestPipelineInvalidInputFile(t *testing.T) {
	skipIfNoServer(t)

	cfg := loadPipelineConfig(t, "pipeline_sentiment.yaml")
	proc := processor.New(cfg)

	opts := processor.Options{
		ShowProgress: false,
	}

	ctx := context.Background()
	err := proc.ProcessFile(ctx, "nonexistent_data.csv", opts)
	if err == nil {
		t.Error("Expected error processing nonexistent file")
	}
}

func TestPipelineContextCancellation(t *testing.T) {
	cleanup := setupPipelineTest(t)
	defer cleanup()

	cfg := loadPipelineConfig(t, "pipeline_sentiment.yaml")
	proc := processor.New(cfg)

	dataPath := filepath.Join(dataDir, "sentiment.csv")
	opts := processor.Options{
		ShowProgress: false,
	}

	// Cancel immediately
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	err := proc.ProcessFile(ctx, dataPath, opts)
	if err == nil {
		t.Log("Note: Processing completed before cancellation took effect")
	} else {
		t.Logf("Got expected cancellation: %v", err)
	}
}

// =============================================================================
// Concurrent Workers Tests
// =============================================================================

func TestPipelineConcurrentWorkers(t *testing.T) {
	cleanup := setupPipelineTest(t)
	defer cleanup()

	cfg := loadPipelineConfig(t, "pipeline_sentiment.yaml")
	cfg.Processing.Workers = 4 // Use multiple workers
	proc := processor.New(cfg)

	dataPath := filepath.Join(dataDir, "sentiment.csv")
	opts := processor.Options{
		Limit:        10,
		ShowProgress: false,
		Verbose:      false,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	start := time.Now()
	err := proc.ProcessFile(ctx, dataPath, opts)
	duration := time.Since(start)

	if err != nil {
		t.Fatalf("Pipeline processing failed: %v", err)
	}

	t.Logf("Processed 10 items with 4 workers in %v", duration)
}

// =============================================================================
// Full Pipeline Accuracy Tests
// =============================================================================

func TestPipelineAccuracyMetrics(t *testing.T) {
	cleanup := setupPipelineTest(t)
	defer cleanup()

	cfg := loadPipelineConfig(t, "pipeline_sentiment.yaml")
	proc := processor.New(cfg)

	dataPath := filepath.Join(dataDir, "sentiment.csv")
	opts := processor.Options{
		Limit:        10,
		ShowProgress: false,
		Verbose:      false,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	err := proc.ProcessFile(ctx, dataPath, opts)
	if err != nil {
		t.Fatalf("Pipeline processing failed: %v", err)
	}

	// Analyze accuracy
	files, _ := os.ReadDir(outputDir)
	for _, f := range files {
		if strings.HasPrefix(f.Name(), "results_") && strings.HasSuffix(f.Name(), ".json") {
			resultFile := filepath.Join(outputDir, f.Name())
			resultData, _ := os.ReadFile(resultFile)

			var results map[string]interface{}
			json.Unmarshal(resultData, &results)

			// Check live metrics
			summary := results["summary"].(map[string]interface{})
			if liveMetrics, ok := summary["live_metrics"]; ok {
				metrics := liveMetrics.(map[string]interface{})
				t.Logf("Accuracy metric (%s): %.2f%%",
					metrics["metric_name"],
					metrics["metric_value"].(float64)*100)
			}

			// Calculate manual accuracy
			resultsList := results["results"].([]interface{})
			correct := 0
			total := 0
			for _, r := range resultsList {
				result := r.(map[string]interface{})
				if result["inference_success"].(bool) {
					predicted := strings.ToLower(result["final_answer"].(string))
					actual := strings.ToLower(result["ground_truth"].(string))
					if predicted == actual {
						correct++
					}
					total++
				}
			}

			if total > 0 {
				accuracy := float64(correct) / float64(total) * 100
				t.Logf("Manual accuracy: %d/%d (%.1f%%)", correct, total, accuracy)
			}
			break
		}
	}
}

// =============================================================================
// Config Info File Tests
// =============================================================================

func TestPipelineConfigInfoFile(t *testing.T) {
	cleanup := setupPipelineTest(t)
	defer cleanup()

	cfg := loadPipelineConfig(t, "pipeline_sentiment.yaml")
	proc := processor.New(cfg)

	dataPath := filepath.Join(dataDir, "sentiment.csv")
	opts := processor.Options{
		Limit:        2,
		ShowProgress: false,
		Verbose:      false,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	err := proc.ProcessFile(ctx, dataPath, opts)
	if err != nil {
		t.Fatalf("Pipeline processing failed: %v", err)
	}

	// Verify config info file was created
	files, _ := os.ReadDir(outputDir)
	var configInfoFound bool
	for _, f := range files {
		if strings.HasPrefix(f.Name(), "config_and_server_info_") && strings.HasSuffix(f.Name(), ".txt") {
			configInfoFound = true
			t.Logf("Config info file created: %s", f.Name())

			// Verify content
			content, _ := os.ReadFile(filepath.Join(outputDir, f.Name()))
			if !strings.Contains(string(content), "PROVIDER CONFIGURATION") {
				t.Error("Config info missing provider section")
			}
			if !strings.Contains(string(content), "MODEL CONFIGURATION") {
				t.Error("Config info missing model section")
			}
			break
		}
	}

	if !configInfoFound {
		t.Error("Config info file not found")
	}
}

// =============================================================================
// Benchmark Tests
// =============================================================================

func BenchmarkPipelineSentiment(b *testing.B) {
	if os.Getenv("LLAMACPP_SKIP") == "1" {
		b.Skip("Skipping benchmark (LLAMACPP_SKIP=1)")
	}

	// Check server availability
	cfg := &models.Config{
		Provider: models.ProviderConfig{
			Name:    "llamacpp",
			BaseURL: serverURL,
			Timeout: "5s",
		},
		Processing: models.ProcessingConfig{Workers: 1},
	}

	c, err := client.NewClient(cfg)
	if err != nil {
		b.Skipf("Failed to create client: %v", err)
	}
	defer c.Close()

	ctx := context.Background()
	if err := c.HealthCheck(ctx); err != nil {
		b.Skipf("Server not available: %v", err)
	}

	// Setup
	os.MkdirAll(outputDir, 0755)
	defer os.RemoveAll(outputDir)

	configPath := filepath.Join(configDir, "pipeline_sentiment.yaml")
	pipelineCfg, _ := config.Load(configPath)
	pipelineCfg.Provider.BaseURL = serverURL
	pipelineCfg.Output.Directory = outputDir

	dataPath := filepath.Join(dataDir, "sentiment.csv")
	opts := processor.Options{
		Limit:        1,
		ShowProgress: false,
		Verbose:      false,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		proc := processor.New(pipelineCfg)
		proc.ProcessFile(ctx, dataPath, opts)
	}
}

// Helper functions
func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen-3] + "..."
}
