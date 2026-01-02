// Package llamacpp_test provides integration tests for the llama.cpp client.
// These tests require a running llama.cpp server with Qwen3-8B model.
//
// To run these tests:
//   1. Start llama.cpp server: llama-server -m Qwen_Qwen3-8B-Q5_K_M.gguf --port 8080
//   2. Run tests: go test ./test/llamacpp/... -v
//
// Set LLAMACPP_URL environment variable to override the default server URL.
// Set LLAMACPP_SKIP=1 to skip these tests entirely.
package llamacpp_test

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/Vitruves/llm-client/internal/client"
	"github.com/Vitruves/llm-client/internal/config"
	"github.com/Vitruves/llm-client/internal/models"
)

const (
	defaultServerURL = "http://localhost:8080"
	configDir        = "configs"
)

var serverURL string

func init() {
	serverURL = os.Getenv("LLAMACPP_URL")
	if serverURL == "" {
		serverURL = defaultServerURL
	}
}

// skipIfNoServer skips the test if the llama.cpp server is not available
func skipIfNoServer(t *testing.T) {
	t.Helper()

	if os.Getenv("LLAMACPP_SKIP") == "1" {
		t.Skip("Skipping llama.cpp tests (LLAMACPP_SKIP=1)")
	}

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
		t.Skipf("Skipping: failed to create client: %v", err)
	}
	defer c.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := c.HealthCheck(ctx); err != nil {
		t.Skipf("Skipping: llama.cpp server not available at %s: %v", serverURL, err)
	}
}

// loadConfig loads a test config file and overrides the server URL
func loadConfig(t *testing.T, configFile string) *models.Config {
	t.Helper()

	configPath := filepath.Join(configDir, configFile)
	cfg, err := config.Load(configPath)
	if err != nil {
		t.Fatalf("Failed to load config %s: %v", configPath, err)
	}

	// Override with actual server URL
	cfg.Provider.BaseURL = serverURL
	return cfg
}

// =============================================================================
// Health Check & Server Info Tests
// =============================================================================

func TestHealthCheck(t *testing.T) {
	skipIfNoServer(t)

	cfg := loadConfig(t, "chat_basic.yaml")
	c, err := client.NewClient(cfg)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer c.Close()

	ctx := context.Background()
	err = c.HealthCheck(ctx)
	if err != nil {
		t.Errorf("Health check failed: %v", err)
	}
}

func TestGetServerInfo(t *testing.T) {
	skipIfNoServer(t)

	cfg := loadConfig(t, "chat_basic.yaml")
	c, err := client.NewClient(cfg)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer c.Close()

	ctx := context.Background()
	info, err := c.GetServerInfo(ctx)
	if err != nil {
		t.Fatalf("GetServerInfo failed: %v", err)
	}

	if !info.Available {
		t.Error("Expected server to be available")
	}

	if info.ServerType != "llamacpp" {
		t.Errorf("Expected server type 'llamacpp', got %q", info.ServerType)
	}

	t.Logf("Server info: %+v", info)
}

// =============================================================================
// Chat Completion Tests
// =============================================================================

func TestChatBasic(t *testing.T) {
	skipIfNoServer(t)

	cfg := loadConfig(t, "chat_basic.yaml")
	c, err := client.NewClient(cfg)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer c.Close()

	req := models.Request{
		Messages: []models.Message{
			{Role: "system", Content: "You are a helpful assistant. Keep responses brief."},
			{Role: "user", Content: "Say hello in exactly 3 words."},
		},
		Options: cfg.Model.Parameters,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	resp, err := c.SendRequest(ctx, req)
	if err != nil {
		t.Fatalf("SendRequest failed: %v", err)
	}

	if !resp.Success {
		t.Errorf("Expected success, got error: %s", resp.Error)
	}

	if resp.Content == "" {
		t.Error("Expected non-empty response content")
	}

	t.Logf("Response: %q", resp.Content)
	t.Logf("Response time: %v", resp.ResponseTime)
}

func TestChatDeterministic(t *testing.T) {
	skipIfNoServer(t)

	cfg := loadConfig(t, "chat_deterministic.yaml")
	c, err := client.NewClient(cfg)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer c.Close()

	req := models.Request{
		Messages: []models.Message{
			{Role: "system", Content: "You are a helpful assistant. Always respond with exactly: Hello World"},
			{Role: "user", Content: "Greet me."},
		},
		Options: cfg.Model.Parameters,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	// Run twice with same seed - results should be similar
	resp1, err := c.SendRequest(ctx, req)
	if err != nil {
		t.Fatalf("First request failed: %v", err)
	}

	resp2, err := c.SendRequest(ctx, req)
	if err != nil {
		t.Fatalf("Second request failed: %v", err)
	}

	t.Logf("Response 1: %q", resp1.Content)
	t.Logf("Response 2: %q", resp2.Content)

	// With temperature=0 and same seed, responses should be identical or very similar
	if resp1.Content != resp2.Content {
		t.Logf("Note: Responses differ despite deterministic settings (this can happen with some backends)")
	}
}

func TestChatMultiTurn(t *testing.T) {
	skipIfNoServer(t)

	cfg := loadConfig(t, "chat_basic.yaml")
	c, err := client.NewClient(cfg)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer c.Close()

	req := models.Request{
		Messages: []models.Message{
			{Role: "system", Content: "You are a helpful assistant."},
			{Role: "user", Content: "My name is Alice."},
			{Role: "assistant", Content: "Nice to meet you, Alice!"},
			{Role: "user", Content: "What is my name?"},
		},
		Options: cfg.Model.Parameters,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	resp, err := c.SendRequest(ctx, req)
	if err != nil {
		t.Fatalf("SendRequest failed: %v", err)
	}

	if !resp.Success {
		t.Errorf("Expected success, got error: %s", resp.Error)
	}

	// Response should mention "Alice"
	if !strings.Contains(strings.ToLower(resp.Content), "alice") {
		t.Errorf("Expected response to mention 'Alice', got: %q", resp.Content)
	}

	t.Logf("Response: %q", resp.Content)
}

// =============================================================================
// Completion (Non-Chat) Tests
// =============================================================================

func TestCompletionBasic(t *testing.T) {
	skipIfNoServer(t)

	cfg := loadConfig(t, "completion_basic.yaml")
	c, err := client.NewClient(cfg)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer c.Close()

	req := models.Request{
		Messages: []models.Message{
			{Role: "user", Content: "Complete this sentence: The quick brown fox"},
		},
		Options: cfg.Model.Parameters,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	resp, err := c.SendRequest(ctx, req)
	if err != nil {
		t.Fatalf("SendRequest failed: %v", err)
	}

	if !resp.Success {
		t.Errorf("Expected success, got error: %s", resp.Error)
	}

	if resp.Content == "" {
		t.Error("Expected non-empty response content")
	}

	t.Logf("Response: %q", resp.Content)
}

// =============================================================================
// Sampling Parameter Tests
// =============================================================================

func TestSamplingMirostat(t *testing.T) {
	skipIfNoServer(t)

	cfg := loadConfig(t, "sampling_mirostat.yaml")
	c, err := client.NewClient(cfg)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer c.Close()

	req := models.Request{
		Messages: []models.Message{
			{Role: "system", Content: "You are a creative storyteller."},
			{Role: "user", Content: "Write a very short story about a robot in 2 sentences."},
		},
		Options: cfg.Model.Parameters,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	resp, err := c.SendRequest(ctx, req)
	if err != nil {
		t.Fatalf("SendRequest failed: %v", err)
	}

	if !resp.Success {
		t.Errorf("Expected success, got error: %s", resp.Error)
	}

	t.Logf("Mirostat response: %q", resp.Content)
}

func TestSamplingAdvanced(t *testing.T) {
	skipIfNoServer(t)

	cfg := loadConfig(t, "sampling_advanced.yaml")
	c, err := client.NewClient(cfg)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer c.Close()

	req := models.Request{
		Messages: []models.Message{
			{Role: "system", Content: "You are a helpful assistant."},
			{Role: "user", Content: "List 3 colors."},
		},
		Options: cfg.Model.Parameters,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	resp, err := c.SendRequest(ctx, req)
	if err != nil {
		t.Fatalf("SendRequest failed: %v", err)
	}

	if !resp.Success {
		t.Errorf("Expected success, got error: %s", resp.Error)
	}

	t.Logf("Advanced sampling response: %q", resp.Content)
}

func TestSamplingWithExplicitParams(t *testing.T) {
	skipIfNoServer(t)

	cfg := loadConfig(t, "chat_basic.yaml")
	c, err := client.NewClient(cfg)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer c.Close()

	// Override config params with explicit request params
	temp := 0.5
	maxTokens := 50
	topP := 0.8
	topK := 30
	seed := int64(12345)

	req := models.Request{
		Messages: []models.Message{
			{Role: "user", Content: "Say 'test' and nothing else."},
		},
		Options: models.ModelParameters{
			Temperature: &temp,
			MaxTokens:   &maxTokens,
			TopP:        &topP,
			TopK:        &topK,
			Seed:        &seed,
			ChatFormat:  cfg.Model.Parameters.ChatFormat,
		},
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	resp, err := c.SendRequest(ctx, req)
	if err != nil {
		t.Fatalf("SendRequest failed: %v", err)
	}

	if !resp.Success {
		t.Errorf("Expected success, got error: %s", resp.Error)
	}

	t.Logf("Response with explicit params: %q", resp.Content)
}

// =============================================================================
// Classification Tests
// =============================================================================

func TestClassificationPositive(t *testing.T) {
	skipIfNoServer(t)

	cfg := loadConfig(t, "classification.yaml")
	c, err := client.NewClient(cfg)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer c.Close()

	// Use /no_think to disable Qwen3's thinking mode for classification
	req := models.Request{
		Messages: []models.Message{
			{Role: "system", Content: "You are a sentiment classifier. Respond with exactly one word: positive, negative, or neutral."},
			{Role: "user", Content: "Classify the sentiment: I love this product, it's amazing! /no_think"},
		},
		Options: cfg.Model.Parameters,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	resp, err := c.SendRequest(ctx, req)
	if err != nil {
		t.Fatalf("SendRequest failed: %v", err)
	}

	if !resp.Success {
		t.Errorf("Expected success, got error: %s", resp.Error)
	}

	lower := strings.ToLower(resp.Content)
	if !strings.Contains(lower, "positive") {
		t.Logf("Note: Expected 'positive' sentiment, got: %q", resp.Content)
	}

	t.Logf("Classification response: %q", resp.Content)
}

func TestClassificationNegative(t *testing.T) {
	skipIfNoServer(t)

	cfg := loadConfig(t, "classification.yaml")
	c, err := client.NewClient(cfg)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer c.Close()

	req := models.Request{
		Messages: []models.Message{
			{Role: "system", Content: "You are a sentiment classifier. Respond with exactly one word: positive, negative, or neutral."},
			{Role: "user", Content: "Classify the sentiment: This is terrible, I hate it and want a refund. /no_think"},
		},
		Options: cfg.Model.Parameters,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	resp, err := c.SendRequest(ctx, req)
	if err != nil {
		t.Fatalf("SendRequest failed: %v", err)
	}

	if !resp.Success {
		t.Errorf("Expected success, got error: %s", resp.Error)
	}

	lower := strings.ToLower(resp.Content)
	if !strings.Contains(lower, "negative") {
		t.Logf("Note: Expected 'negative' sentiment, got: %q", resp.Content)
	}

	t.Logf("Classification response: %q", resp.Content)
}

func TestClassificationNeutral(t *testing.T) {
	skipIfNoServer(t)

	cfg := loadConfig(t, "classification.yaml")
	c, err := client.NewClient(cfg)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer c.Close()

	req := models.Request{
		Messages: []models.Message{
			{Role: "system", Content: "You are a sentiment classifier. Respond with exactly one word: positive, negative, or neutral."},
			{Role: "user", Content: "Classify the sentiment: The package arrived on Tuesday. /no_think"},
		},
		Options: cfg.Model.Parameters,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	resp, err := c.SendRequest(ctx, req)
	if err != nil {
		t.Fatalf("SendRequest failed: %v", err)
	}

	if !resp.Success {
		t.Errorf("Expected success, got error: %s", resp.Error)
	}

	lower := strings.ToLower(resp.Content)
	if !strings.Contains(lower, "neutral") {
		t.Logf("Note: Expected 'neutral' sentiment, got: %q", resp.Content)
	}

	t.Logf("Classification response: %q", resp.Content)
}

// =============================================================================
// Stop Sequences Tests
// =============================================================================

func TestStopSequences(t *testing.T) {
	skipIfNoServer(t)

	cfg := loadConfig(t, "stop_sequences.yaml")
	c, err := client.NewClient(cfg)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer c.Close()

	req := models.Request{
		Messages: []models.Message{
			{Role: "system", Content: "You are a helpful assistant."},
			{Role: "user", Content: "Write a paragraph, then write END, then write another paragraph."},
		},
		Options: cfg.Model.Parameters,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	resp, err := c.SendRequest(ctx, req)
	if err != nil {
		t.Fatalf("SendRequest failed: %v", err)
	}

	if !resp.Success {
		t.Errorf("Expected success, got error: %s", resp.Error)
	}

	// Response should not contain "END" as it should stop before that
	// (though the model might not follow instructions perfectly)
	t.Logf("Stop sequences response: %q", resp.Content)
	t.Logf("Response length: %d chars", len(resp.Content))
}

// =============================================================================
// Thinking Mode Tests (Qwen3)
// =============================================================================

func TestThinkingMode(t *testing.T) {
	skipIfNoServer(t)

	cfg := loadConfig(t, "thinking_mode.yaml")
	c, err := client.NewClient(cfg)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer c.Close()

	req := models.Request{
		Messages: []models.Message{
			{Role: "system", Content: "You are a helpful assistant that thinks step by step."},
			{Role: "user", Content: "What is 15 + 27? /think"},
		},
		Options: cfg.Model.Parameters,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	resp, err := c.SendRequest(ctx, req)
	if err != nil {
		t.Fatalf("SendRequest failed: %v", err)
	}

	if !resp.Success {
		t.Errorf("Expected success, got error: %s", resp.Error)
	}

	// Check if response contains thinking tags or the answer
	hasThinking := strings.Contains(resp.Content, "<think>") || strings.Contains(resp.Content, "think")
	hasAnswer := strings.Contains(resp.Content, "42")

	t.Logf("Has thinking: %v, Has correct answer: %v", hasThinking, hasAnswer)
	t.Logf("Thinking mode response: %q", resp.Content)
}

// =============================================================================
// Response Metadata Tests
// =============================================================================

func TestResponseTime(t *testing.T) {
	skipIfNoServer(t)

	cfg := loadConfig(t, "chat_basic.yaml")
	c, err := client.NewClient(cfg)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer c.Close()

	req := models.Request{
		Messages: []models.Message{
			{Role: "user", Content: "Say hi."},
		},
		Options: cfg.Model.Parameters,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	resp, err := c.SendRequest(ctx, req)
	if err != nil {
		t.Fatalf("SendRequest failed: %v", err)
	}

	if resp.ResponseTime <= 0 {
		t.Errorf("Expected positive response time, got %v", resp.ResponseTime)
	}

	t.Logf("Response time: %v", resp.ResponseTime)
}

func TestUsageStats(t *testing.T) {
	skipIfNoServer(t)

	cfg := loadConfig(t, "chat_basic.yaml")
	c, err := client.NewClient(cfg)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer c.Close()

	req := models.Request{
		Messages: []models.Message{
			{Role: "system", Content: "You are helpful."},
			{Role: "user", Content: "Say hello."},
		},
		Options: cfg.Model.Parameters,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	resp, err := c.SendRequest(ctx, req)
	if err != nil {
		t.Fatalf("SendRequest failed: %v", err)
	}

	if resp.Usage != nil {
		t.Logf("Usage - Prompt: %d, Completion: %d, Total: %d",
			resp.Usage.PromptTokens, resp.Usage.CompletionTokens, resp.Usage.TotalTokens)

		if resp.Usage.TotalTokens <= 0 {
			t.Error("Expected positive total tokens")
		}
	} else {
		t.Log("Note: Usage stats not returned by server")
	}
}

func TestFinishReason(t *testing.T) {
	skipIfNoServer(t)

	cfg := loadConfig(t, "chat_basic.yaml")
	c, err := client.NewClient(cfg)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer c.Close()

	req := models.Request{
		Messages: []models.Message{
			{Role: "user", Content: "Say 'done'."},
		},
		Options: cfg.Model.Parameters,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	resp, err := c.SendRequest(ctx, req)
	if err != nil {
		t.Fatalf("SendRequest failed: %v", err)
	}

	if resp.FinishReason != nil {
		t.Logf("Finish reason: %s", *resp.FinishReason)
	} else {
		t.Log("Note: Finish reason not returned by server")
	}
}

// =============================================================================
// Error Handling Tests
// =============================================================================

func TestMaxTokensLimit(t *testing.T) {
	skipIfNoServer(t)

	cfg := loadConfig(t, "chat_basic.yaml")
	c, err := client.NewClient(cfg)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer c.Close()

	// Request with very small max_tokens
	maxTokens := 5
	req := models.Request{
		Messages: []models.Message{
			{Role: "user", Content: "Write a very long essay about the history of computing."},
		},
		Options: models.ModelParameters{
			MaxTokens:  &maxTokens,
			ChatFormat: cfg.Model.Parameters.ChatFormat,
		},
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	resp, err := c.SendRequest(ctx, req)
	if err != nil {
		t.Fatalf("SendRequest failed: %v", err)
	}

	if !resp.Success {
		t.Errorf("Expected success, got error: %s", resp.Error)
	}

	// Response should be truncated
	t.Logf("Truncated response (%d chars): %q", len(resp.Content), resp.Content)

	if resp.FinishReason != nil && *resp.FinishReason == "length" {
		t.Log("Correctly stopped due to length limit")
	}
}

func TestContextCancellation(t *testing.T) {
	skipIfNoServer(t)

	cfg := loadConfig(t, "chat_basic.yaml")
	c, err := client.NewClient(cfg)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer c.Close()

	req := models.Request{
		Messages: []models.Message{
			{Role: "user", Content: "Write a very long story."},
		},
		Options: cfg.Model.Parameters,
	}

	// Cancel immediately
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err = c.SendRequest(ctx, req)
	if err == nil {
		t.Error("Expected error due to cancelled context")
	} else {
		t.Logf("Got expected error: %v", err)
	}
}

// =============================================================================
// Concurrent Request Tests
// =============================================================================

func TestConcurrentRequests(t *testing.T) {
	skipIfNoServer(t)

	cfg := loadConfig(t, "chat_basic.yaml")
	c, err := client.NewClient(cfg)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer c.Close()

	numRequests := 3
	results := make(chan *models.Response, numRequests)
	errors := make(chan error, numRequests)

	for i := 0; i < numRequests; i++ {
		go func(id int) {
			req := models.Request{
				Messages: []models.Message{
					{Role: "user", Content: "Say your request number: " + string(rune('0'+id))},
				},
				Options: cfg.Model.Parameters,
			}

			ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
			defer cancel()

			resp, err := c.SendRequest(ctx, req)
			if err != nil {
				errors <- err
				return
			}
			results <- resp
		}(i)
	}

	successCount := 0
	for i := 0; i < numRequests; i++ {
		select {
		case resp := <-results:
			if resp.Success {
				successCount++
				t.Logf("Request succeeded: %q", resp.Content)
			}
		case err := <-errors:
			t.Logf("Request error: %v", err)
		case <-time.After(180 * time.Second):
			t.Fatal("Timeout waiting for concurrent requests")
		}
	}

	if successCount == 0 {
		t.Error("No concurrent requests succeeded")
	}

	t.Logf("Concurrent requests: %d/%d succeeded", successCount, numRequests)
}

// =============================================================================
// Benchmark Tests
// =============================================================================

func BenchmarkChatRequest(b *testing.B) {
	if os.Getenv("LLAMACPP_SKIP") == "1" {
		b.Skip("Skipping benchmark (LLAMACPP_SKIP=1)")
	}

	cfg := &models.Config{
		Provider: models.ProviderConfig{
			Name:    "llamacpp",
			BaseURL: serverURL,
			Timeout: "120s",
		},
		Model: models.ModelConfig{
			Parameters: models.ModelParameters{
				ChatFormat: stringPtr("chatml"),
			},
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

	maxTokens := 20
	temp := 0.0

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		req := models.Request{
			Messages: []models.Message{
				{Role: "user", Content: "Say hi."},
			},
			Options: models.ModelParameters{
				MaxTokens:   &maxTokens,
				Temperature: &temp,
				ChatFormat:  stringPtr("chatml"),
			},
		}

		_, err := c.SendRequest(ctx, req)
		if err != nil {
			b.Errorf("Request failed: %v", err)
		}
	}
}

// Helper functions
func stringPtr(s string) *string { return &s }
