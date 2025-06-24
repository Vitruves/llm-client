package processor

import (
	"context"
	"encoding/csv"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/Vitruves/llm-client/internal/client"
	"github.com/Vitruves/llm-client/internal/models"
)

// TestProcessorIntegration is a comprehensive integration test that tests all config parameters
// with a real vLLM server. This test requires a running vLLM server on localhost:8000.
func TestProcessorIntegration(t *testing.T) {
	// Skip this test if we're not running integration tests
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	// Check if we can connect to vLLM server
	if !isVLLMServerAvailable() {
		t.Skip("vLLM server not available on localhost:8000, skipping integration test")
	}

	t.Log("Running comprehensive integration test with real vLLM server")

	// Create temporary directories for test
	tmpDir := t.TempDir()
	outputDir := filepath.Join(tmpDir, "output")

	// Create comprehensive test configuration with all parameters
	config := createComprehensiveConfig(outputDir)

	// Create test input data
	inputFile := filepath.Join(tmpDir, "test_input.csv")
	createComprehensiveTestData(inputFile)

	// Create processor
	processor := New(config)
	processor.SetConfigFile("integration_test_config.yaml")

	// Test 1: Basic processing with all parameters
	t.Run("BasicProcessing", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		defer cancel()

		opts := Options{
			Limit:        3, // Limit to 3 items for faster testing
			ShowProgress: true,
			Verbose:      true,
		}

		err := processor.ProcessFile(ctx, inputFile, opts)
		if err != nil {
			t.Fatalf("Processing failed: %v", err)
		}

		// Verify output files were created
		files, err := os.ReadDir(outputDir)
		if err != nil {
			t.Fatalf("Failed to read output directory: %v", err)
		}

		if len(files) == 0 {
			t.Error("Expected output files to be created")
		}

		t.Logf("Created %d output files", len(files))
		for _, file := range files {
			t.Logf("  - %s", file.Name())
		}
	})

	// Test 2: Streaming mode
	t.Run("StreamingMode", func(t *testing.T) {
		streamConfig := createComprehensiveConfig(filepath.Join(tmpDir, "streaming"))
		streamConfig.Output.StreamOutput = true
		streamConfig.Output.StreamSaveEvery = 1
		streamConfig.Output.Format = "json"

		streamProcessor := New(streamConfig)

		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		defer cancel()

		opts := Options{
			Limit:        2,
			ShowProgress: false,
			Verbose:      false,
		}

		err := streamProcessor.ProcessFile(ctx, inputFile, opts)
		if err != nil {
			t.Fatalf("Streaming processing failed: %v", err)
		}

		// Verify streaming output
		streamFiles, err := os.ReadDir(streamConfig.Output.Directory)
		if err != nil {
			t.Fatalf("Failed to read streaming output directory: %v", err)
		}

		hasStreamingFile := false
		for _, file := range streamFiles {
			if filepath.Ext(file.Name()) == ".json" {
				hasStreamingFile = true
				break
			}
		}

		if !hasStreamingFile {
			t.Error("Expected streaming JSON file to be created")
		}
	})

	// Test 3: Multiple output formats
	formats := []string{"json", "csv", "parquet"}
	for _, format := range formats {
		t.Run(fmt.Sprintf("Format_%s", format), func(t *testing.T) {
			formatConfig := createComprehensiveConfig(filepath.Join(tmpDir, format))
			formatConfig.Output.Format = format
			formatConfig.Output.StreamOutput = false

			formatProcessor := New(formatConfig)

			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
			defer cancel()

			opts := Options{
				Limit:        2,
				ShowProgress: false,
				Verbose:      false,
			}

			err := formatProcessor.ProcessFile(ctx, inputFile, opts)
			if err != nil {
				t.Fatalf("Processing with format %s failed: %v", format, err)
			}

			// Verify format-specific output
			formatFiles, err := os.ReadDir(formatConfig.Output.Directory)
			if err != nil {
				t.Fatalf("Failed to read format output directory: %v", err)
			}

			hasFormatFile := false
			expectedExt := "." + format
			if format == "parquet" {
				expectedExt = ".parquet"
			}

			for _, file := range formatFiles {
				if filepath.Ext(file.Name()) == expectedExt {
					hasFormatFile = true
					break
				}
			}

			if !hasFormatFile {
				t.Errorf("Expected %s file to be created", format)
			}
		})
	}

	// Test 4: Consensus mode (multiple attempts)
	t.Run("ConsensusMode", func(t *testing.T) {
		consensusConfig := createComprehensiveConfig(filepath.Join(tmpDir, "consensus"))
		consensusConfig.Processing.Repeat = 3
		consensusConfig.Processing.RateLimit = true

		consensusProcessor := New(consensusConfig)

		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
		defer cancel()

		opts := Options{
			Limit:        2,
			ShowProgress: true,
			Verbose:      true,
		}

		err := consensusProcessor.ProcessFile(ctx, inputFile, opts)
		if err != nil {
			t.Fatalf("Consensus processing failed: %v", err)
		}
	})

	// Test 5: All vLLM parameters
	t.Run("AllVLLMParameters", func(t *testing.T) {
		vllmConfig := createComprehensiveConfig(filepath.Join(tmpDir, "vllm_params"))
		
		// Set all vLLM parameters
		params := &vllmConfig.Model.Parameters
		params.Temperature = func() *float64 { v := 0.8; return &v }()
		params.MaxTokens = func() *int { v := 50; return &v }()
		params.TopP = func() *float64 { v := 0.9; return &v }()
		params.TopK = func() *int { v := 40; return &v }()
		params.RepetitionPenalty = func() *float64 { v := 1.1; return &v }()
		params.Seed = func() *int64 { v := int64(42); return &v }()
		params.Stop = []string{"\n", "END"}
		params.GuidedChoice = []string{"positive", "negative", "neutral"}

		vllmProcessor := New(vllmConfig)

		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		defer cancel()

		opts := Options{
			Limit:        2,
			ShowProgress: false,
			Verbose:      true,
		}

		err := vllmProcessor.ProcessFile(ctx, inputFile, opts)
		if err != nil {
			t.Fatalf("vLLM parameters processing failed: %v", err)
		}
	})

	// Test 6: Live metrics with reference data
	t.Run("LiveMetrics", func(t *testing.T) {
		metricsConfig := createComprehensiveConfig(filepath.Join(tmpDir, "metrics"))
		metricsConfig.Processing.LiveMetrics.Enabled = true
		metricsConfig.Processing.LiveMetrics.Metric = "accuracy"
		metricsConfig.Processing.LiveMetrics.GroundTruth = "ground_truth"

		metricsProcessor := New(metricsConfig)

		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		defer cancel()

		opts := Options{
			Limit:        3,
			ShowProgress: true,
			Verbose:      false,
		}

		err := metricsProcessor.ProcessFile(ctx, inputFile, opts)
		if err != nil {
			t.Fatalf("Live metrics processing failed: %v", err)
		}
	})

	t.Log("All integration tests completed successfully!")
}

// Helper function to check if vLLM server is available
func isVLLMServerAvailable() bool {
	config := &models.Config{
		Provider: models.ProviderConfig{
			Name:    "vllm",
			BaseURL: "http://localhost:8000",
			Timeout: "10s",
		},
		Model: models.ModelConfig{
			Name: "test-model",
		},
	}

	client, err := createMockClientForCheck(config)
	if err != nil {
		return false
	}
	defer client.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	return client.HealthCheck(ctx) == nil
}

// Create a real client for server check
func createMockClientForCheck(config *models.Config) (models.Client, error) {
	return client.NewClient(config)
}

// Create comprehensive configuration with all parameters
func createComprehensiveConfig(outputDir string) *models.Config {
	return &models.Config{
		Provider: models.ProviderConfig{
			Name:    "vllm",
			BaseURL: "http://localhost:8000",
			Timeout: "60s",
		},
		Model: models.ModelConfig{
			Name: "Qwen/Qwen2.5-Coder-3B-Instruct", // Common vLLM model
			Parameters: models.ModelParameters{
				Temperature:       func() *float64 { v := 0.7; return &v }(),
				MaxTokens:         func() *int { v := 100; return &v }(),
				TopP:              func() *float64 { v := 0.95; return &v }(),
				TopK:              func() *int { v := 50; return &v }(),
				RepetitionPenalty: func() *float64 { v := 1.0; return &v }(),
				Seed:              func() *int64 { v := int64(12345); return &v }(),
				Stop:              []string{"\n\n"},
				EnableThinking:    func() *bool { v := false; return &v }(),
			},
		},
		Classification: models.ClassificationConfig{
			Template: models.TemplateConfig{
				System: "You are a sentiment analysis assistant. Classify the sentiment of the given text.",
				User:   "Classify the sentiment of this text as positive, negative, or neutral:\n\nText: {text}\n\nSentiment:",
			},
			Parsing: models.ParsingConfig{
				Find:           []string{"positive", "negative", "neutral"},
				Default:        "neutral",
				Fallback:       "unknown",
				AnswerPatterns: []string{"positive", "negative", "neutral"},
				CaseSensitive:  func() *bool { v := false; return &v }(),
				ExactMatch:     func() *bool { v := false; return &v }(),
				Map: map[string]string{
					"pos":      "positive",
					"neg":      "negative",
					"neut":     "neutral",
					"good":     "positive",
					"bad":      "negative",
					"okay":     "neutral",
				},
			},
			FieldMapping: &models.FieldMappingConfig{
				InputTextField: "text",
			},
		},
		Processing: models.ProcessingConfig{
			Workers:     2,
			BatchSize:   1,
			Repeat:      1,
			RateLimit:   false,
			MinimalMode: false,
			LiveMetrics: &models.LiveMetrics{
				Enabled:     true,
				Metric:      "accuracy",
				GroundTruth: "ground_truth",
				Classes:     []string{"positive", "negative", "neutral"},
			},
		},
		Output: models.OutputConfig{
			Directory:          outputDir,
			Format:             "json",
			IncludeRawResponse: true,
			IncludeThinking:    true,
			StreamOutput:       false,
			StreamSaveEvery:    5,
		},
		Reference: models.ReferenceConfig{
			File:        "",
			Column:      "ground_truth",
			Format:      "csv",
			IndexColumn: "index",
		},
	}
}

// Create comprehensive test data with various scenarios
func createComprehensiveTestData(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Write header
	header := []string{"index", "text", "ground_truth", "category", "source"}
	if err := writer.Write(header); err != nil {
		return err
	}

	// Write test data with various scenarios
	testData := [][]string{
		{"0", "I love this product! It's amazing and works perfectly.", "positive", "review", "test"},
		{"1", "This is terrible. I hate it and want my money back.", "negative", "review", "test"},
		{"2", "It's okay, not great but not bad either.", "neutral", "review", "test"},
		{"3", "Absolutely fantastic! Best purchase I've ever made.", "positive", "review", "test"},
		{"4", "Worst experience ever. Complete waste of money.", "negative", "review", "test"},
		{"5", "Average product. Does what it's supposed to do.", "neutral", "review", "test"},
		{"6", "Outstanding quality and excellent customer service!", "positive", "review", "test"},
		{"7", "Broke after one day. Very disappointed.", "negative", "review", "test"},
		{"8", "It's fine. Nothing special but works as expected.", "neutral", "review", "test"},
		{"9", "Exceeded my expectations! Highly recommend.", "positive", "review", "test"},
	}

	for _, row := range testData {
		if err := writer.Write(row); err != nil {
			return err
		}
	}

	return nil
}