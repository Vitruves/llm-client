package client

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"llm-client/internal/models"
)

func TestNewClient(t *testing.T) {
	tests := []struct {
		name        string
		config      *models.Config
		expectError bool
		expectType  string
	}{
		{
			name: "vLLM client",
			config: &models.Config{
				Provider: models.ProviderConfig{
					Name:    "vllm",
					BaseURL: "http://localhost:8000/v1",
					Timeout: "60s",
				},
				Processing: models.ProcessingConfig{Workers: 4},
			},
			expectError: false,
			expectType:  "*client.VLLMClient",
		},
		{
			name: "llama.cpp client",
			config: &models.Config{
				Provider: models.ProviderConfig{
					Name:    "llamacpp",
					BaseURL: "http://localhost:8080",
					Timeout: "60s",
				},
				Processing: models.ProcessingConfig{Workers: 2},
			},
			expectError: false,
			expectType:  "*client.LlamaCppClient",
		},
		{
			name: "OpenAI client",
			config: &models.Config{
				Provider: models.ProviderConfig{
					Name:   "openai",
					APIKey: "test-key",
					Timeout: "60s",
				},
				Processing: models.ProcessingConfig{Workers: 1},
			},
			expectError: false,
			expectType:  "*client.OpenAIClient",
		},
		{
			name: "invalid provider",
			config: &models.Config{
				Provider: models.ProviderConfig{
					Name: "invalid",
				},
			},
			expectError: true,
		},
		{
			name: "invalid timeout",
			config: &models.Config{
				Provider: models.ProviderConfig{
					Name:    "vllm",
					BaseURL: "http://localhost:8000",
					Timeout: "invalid",
				},
				Processing: models.ProcessingConfig{Workers: 1},
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client, err := NewClient(tt.config)
			
			if tt.expectError {
				if err == nil {
					t.Error("Expected error but got none")
				}
				return
			}
			
			if err != nil {
				t.Errorf("Unexpected error: %v", err)
				return
			}
			
			if client == nil {
				t.Error("Expected client to be created")
				return
			}
			
			// Clean up
			client.Close()
		})
	}
}

func TestVLLMClientBasicRequest(t *testing.T) {
	// Mock vLLM server
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/chat/completions" {
			response := map[string]interface{}{
				"choices": []map[string]interface{}{
					{
						"message": map[string]interface{}{
							"content": "positive",
						},
						"finish_reason": "stop",
					},
				},
				"usage": map[string]interface{}{
					"prompt_tokens":     10,
					"completion_tokens": 1,
					"total_tokens":      11,
				},
			}
			json.NewEncoder(w).Encode(response)
		} else if r.URL.Path == "/health" {
			w.WriteHeader(http.StatusOK)
		}
	}))
	defer mockServer.Close()

	config := &models.Config{
		Provider: models.ProviderConfig{
			Name:    "vllm",
			BaseURL: mockServer.URL + "/v1",
			Timeout: "10s",
		},
		Processing: models.ProcessingConfig{Workers: 1},
	}

	client, err := NewClient(config)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer client.Close()

	// Test health check
	ctx := context.Background()
	err = client.HealthCheck(ctx)
	if err != nil {
		t.Errorf("Health check failed: %v", err)
	}

	// Test request
	req := models.Request{
		Messages: []models.Message{
			{Role: "system", Content: "You are a classifier"},
			{Role: "user", Content: "This is positive"},
		},
		Options: models.ModelParameters{
			Temperature: floatPtr(0.7),
			MaxTokens:   intPtr(100),
		},
	}

	resp, err := client.SendRequest(ctx, req)
	if err != nil {
		t.Errorf("Send request failed: %v", err)
		return
	}

	if !resp.Success {
		t.Errorf("Expected successful response, got error: %s", resp.Error)
	}

	if resp.Content != "positive" {
		t.Errorf("Expected content 'positive', got '%s'", resp.Content)
	}

	if resp.Usage == nil {
		t.Error("Expected usage information")
	} else {
		if resp.Usage.TotalTokens != 11 {
			t.Errorf("Expected total tokens 11, got %d", resp.Usage.TotalTokens)
		}
	}
}

func TestVLLMClientWithGuidedGeneration(t *testing.T) {
	// Mock server that validates guided generation parameters
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/chat/completions" {
			var requestBody map[string]interface{}
			err := json.NewDecoder(r.Body).Decode(&requestBody)
			if err != nil {
				t.Errorf("Failed to decode request body: %v", err)
				w.WriteHeader(http.StatusBadRequest)
				return
			}

			// Check that guided generation parameters are present
			if guidedChoice, ok := requestBody["guided_choice"].([]interface{}); ok {
				if len(guidedChoice) != 3 {
					t.Errorf("Expected 3 guided choices, got %d", len(guidedChoice))
				}
			} else {
				t.Error("Expected guided_choice parameter")
			}

			if guidedRegex, ok := requestBody["guided_regex"].(string); !ok || guidedRegex != "^(positive|negative|neutral)$" {
				t.Errorf("Expected guided regex, got %v", guidedRegex)
			}

			response := map[string]interface{}{
				"choices": []map[string]interface{}{
					{
						"message": map[string]interface{}{
							"content": "positive",
						},
					},
				},
			}
			json.NewEncoder(w).Encode(response)
		}
	}))
	defer mockServer.Close()

	config := &models.Config{
		Provider: models.ProviderConfig{
			Name:    "vllm",
			BaseURL: mockServer.URL + "/v1",
			Timeout: "10s",
		},
		Processing: models.ProcessingConfig{Workers: 1},
	}

	client, err := NewClient(config)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer client.Close()

	req := models.Request{
		Messages: []models.Message{
			{Role: "user", Content: "Classify this"},
		},
		Options: models.ModelParameters{
			GuidedChoice: []string{"positive", "negative", "neutral"},
			GuidedRegex:  stringPtr("^(positive|negative|neutral)$"),
		},
	}

	ctx := context.Background()
	resp, err := client.SendRequest(ctx, req)
	if err != nil {
		t.Errorf("Send request failed: %v", err)
		return
	}

	if !resp.Success {
		t.Errorf("Expected successful response")
	}
}

func TestVLLMClientWithThinkingMode(t *testing.T) {
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/chat/completions" {
			response := map[string]interface{}{
				"choices": []map[string]interface{}{
					{
						"message": map[string]interface{}{
							"content":           "positive",
							"reasoning_content": "This is clearly a positive statement because...",
						},
					},
				},
			}
			json.NewEncoder(w).Encode(response)
		}
	}))
	defer mockServer.Close()

	config := &models.Config{
		Provider: models.ProviderConfig{
			Name:    "vllm",
			BaseURL: mockServer.URL + "/v1",
			Timeout: "10s",
		},
		Processing: models.ProcessingConfig{Workers: 1},
	}

	client, err := NewClient(config)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer client.Close()

	req := models.Request{
		Messages: []models.Message{
			{Role: "user", Content: "Classify this"},
		},
		Options: models.ModelParameters{
			EnableThinking: boolPtr(true),
		},
	}

	ctx := context.Background()
	resp, err := client.SendRequest(ctx, req)
	if err != nil {
		t.Errorf("Send request failed: %v", err)
		return
	}

	if !resp.Success {
		t.Errorf("Expected successful response")
	}

	// Should contain thinking content
	expectedContent := "<think>\nThis is clearly a positive statement because...\n</think>\n\npositive"
	if resp.Content != expectedContent {
		t.Errorf("Expected formatted thinking content, got '%s'", resp.Content)
	}
}

func TestLlamaCppClientAdvancedSampling(t *testing.T) {
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/chat/completions" {
			var requestBody map[string]interface{}
			err := json.NewDecoder(r.Body).Decode(&requestBody)
			if err != nil {
				t.Errorf("Failed to decode request body: %v", err)
				w.WriteHeader(http.StatusBadRequest)
				return
			}

			// Check advanced sampling parameters
			if mirostat, ok := requestBody["mirostat"].(float64); !ok || mirostat != 2 {
				t.Errorf("Expected mirostat 2, got %v", mirostat)
			}
			if mirostatTau, ok := requestBody["mirostat_tau"].(float64); !ok || mirostatTau != 5.0 {
				t.Errorf("Expected mirostat_tau 5.0, got %v", mirostatTau)
			}

			response := map[string]interface{}{
				"choices": []map[string]interface{}{
					{
						"message": map[string]interface{}{
							"content": "negative",
						},
					},
				},
			}
			json.NewEncoder(w).Encode(response)
		} else if r.URL.Path == "/health" {
			w.WriteHeader(http.StatusOK)
		}
	}))
	defer mockServer.Close()

	config := &models.Config{
		Provider: models.ProviderConfig{
			Name:    "llamacpp",
			BaseURL: mockServer.URL,
			Timeout: "10s",
		},
		Processing: models.ProcessingConfig{Workers: 1},
		Model: models.ModelConfig{
			Parameters: models.ModelParameters{
				ChatFormat: stringPtr("chatml"),
			},
		},
	}

	client, err := NewClient(config)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer client.Close()

	req := models.Request{
		Messages: []models.Message{
			{Role: "user", Content: "Classify this"},
		},
		Options: models.ModelParameters{
			Mirostat:    intPtr(2),
			MirostatTau: floatPtr(5.0),
			MirostatEta: floatPtr(0.1),
			TfsZ:        floatPtr(1.0),
			TypicalP:    floatPtr(1.0),
		},
	}

	ctx := context.Background()
	resp, err := client.SendRequest(ctx, req)
	if err != nil {
		t.Errorf("Send request failed: %v", err)
		return
	}

	if !resp.Success {
		t.Errorf("Expected successful response")
	}
}

func TestClientErrorHandling(t *testing.T) {
	tests := []struct {
		name           string
		serverResponse func(w http.ResponseWriter, r *http.Request)
		expectError    bool
		errorContains  string
	}{
		{
			name: "server error response",
			serverResponse: func(w http.ResponseWriter, r *http.Request) {
				response := map[string]interface{}{
					"error": map[string]interface{}{
						"message": "Model not found",
					},
				}
				json.NewEncoder(w).Encode(response)
			},
			expectError:   true,
			errorContains: "Model not found",
		},
		{
			name: "empty choices",
			serverResponse: func(w http.ResponseWriter, r *http.Request) {
				response := map[string]interface{}{
					"choices": []interface{}{},
				}
				json.NewEncoder(w).Encode(response)
			},
			expectError:   true,
			errorContains: "no response choices returned",
		},
		{
			name: "malformed JSON",
			serverResponse: func(w http.ResponseWriter, r *http.Request) {
				w.Write([]byte(`{"invalid": json`))
			},
			expectError: true,
		},
		{
			name: "network timeout",
			serverResponse: func(w http.ResponseWriter, r *http.Request) {
				// Simulate slow response
				time.Sleep(2 * time.Second)
				json.NewEncoder(w).Encode(map[string]interface{}{})
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockServer := httptest.NewServer(http.HandlerFunc(tt.serverResponse))
			defer mockServer.Close()

			config := &models.Config{
				Provider: models.ProviderConfig{
					Name:    "vllm",
					BaseURL: mockServer.URL + "/v1",
					Timeout: "1s", // Short timeout for timeout test
				},
				Processing: models.ProcessingConfig{Workers: 1},
			}

			client, err := NewClient(config)
			if err != nil {
				t.Fatalf("Failed to create client: %v", err)
			}
			defer client.Close()

			req := models.Request{
				Messages: []models.Message{
					{Role: "user", Content: "Test"},
				},
			}

			ctx := context.Background()
			resp, err := client.SendRequest(ctx, req)

			if tt.expectError {
				if err == nil && (resp == nil || resp.Success) {
					t.Error("Expected error but got none")
				}
				if tt.errorContains != "" && resp != nil && resp.Error != "" {
					if !contains(resp.Error, tt.errorContains) {
						t.Errorf("Expected error to contain '%s', got '%s'", tt.errorContains, resp.Error)
					}
				}
			} else {
				if err != nil {
					t.Errorf("Unexpected error: %v", err)
				}
			}
		})
	}
}

func TestGetServerInfo(t *testing.T) {
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			w.WriteHeader(http.StatusOK)
		case "/v1/models":
			response := map[string]interface{}{
				"data": []map[string]interface{}{
					{
						"id":           "test-model",
						"object":       "model",
						"created":      1699564800,
						"owned_by":     "test",
						"max_model_len": 4096,
					},
				},
			}
			json.NewEncoder(w).Encode(response)
		case "/version":
			response := map[string]interface{}{
				"version": "0.6.0",
			}
			json.NewEncoder(w).Encode(response)
		}
	}))
	defer mockServer.Close()

	config := &models.Config{
		Provider: models.ProviderConfig{
			Name:    "vllm",
			BaseURL: mockServer.URL + "/v1",
			Timeout: "10s",
		},
		Processing: models.ProcessingConfig{Workers: 4},
		Model: models.ModelConfig{
			Name: "test-model",
			Parameters: models.ModelParameters{
				Temperature: floatPtr(0.7),
				MaxTokens:   intPtr(100),
			},
		},
	}

	client, err := NewClient(config)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer client.Close()

	ctx := context.Background()
	serverInfo, err := client.GetServerInfo(ctx)
	if err != nil {
		t.Errorf("GetServerInfo failed: %v", err)
		return
	}

	if !serverInfo.Available {
		t.Error("Expected server to be available")
	}

	if serverInfo.ServerType != "vllm" {
		t.Errorf("Expected server type 'vllm', got '%s'", serverInfo.ServerType)
	}

	if serverInfo.Models["model_name"] != "test-model" {
		t.Errorf("Expected model name 'test-model', got %v", serverInfo.Models["model_name"])
	}

	if serverInfo.Config["temperature"] != 0.7 {
		t.Errorf("Expected temperature 0.7 in config, got %v", serverInfo.Config["temperature"])
	}
}

// Helper functions
func floatPtr(f float64) *float64 { return &f }
func intPtr(i int) *int           { return &i }
func stringPtr(s string) *string  { return &s }
func boolPtr(b bool) *bool        { return &b }

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(substr) == 0 || 
		(len(s) > len(substr) && (s[:len(substr)] == substr || s[len(s)-len(substr):] == substr || 
		 len(s) > len(substr)*2 && s[len(s)/2-len(substr)/2:len(s)/2+len(substr)/2+len(substr)%2] == substr)))
}