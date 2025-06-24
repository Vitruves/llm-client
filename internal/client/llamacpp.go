package client

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/Vitruves/llm-client/internal/models"
)

type LlamaCppClient struct {
	baseURL    string
	httpClient *http.Client
	timeout    time.Duration
	config     *models.Config
}

func NewLlamaCppClient(cfg *models.Config) (*LlamaCppClient, error) {
	timeout, err := time.ParseDuration(cfg.Provider.Timeout)
	if err != nil {
		return nil, fmt.Errorf("invalid timeout: %w", err)
	}

	return &LlamaCppClient{
		baseURL:    cfg.Provider.BaseURL,
		httpClient: &http.Client{Timeout: timeout},
		timeout:    timeout,
		config:     cfg,
	}, nil
}

func (c *LlamaCppClient) SendRequest(ctx context.Context, req models.Request) (*models.Response, error) {
	start := time.Now()

	// Try chat completions endpoint first, fallback to completion
	if c.supportsChatCompletions() {
		return c.sendChatCompletionRequest(ctx, req, start)
	}
	return c.sendCompletionRequest(ctx, req, start)
}

func (c *LlamaCppClient) supportsChatCompletions() bool {
	// Check if chat format is specified or if we should use chat completions
	return c.config.Model.Parameters.ChatFormat != nil
}

func (c *LlamaCppClient) sendChatCompletionRequest(ctx context.Context, req models.Request, start time.Time) (*models.Response, error) {
	llamaReq := map[string]interface{}{
		"messages": c.formatMessages(req.Messages),
		"stream":   false,
	}

	c.addParameters(llamaReq, req.Options)
	c.addChatCompletionFeatures(llamaReq, req)

	jsonData, err := json.Marshal(llamaReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/v1/chat/completions", c.baseURL)
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	var llamaResp struct {
		Choices []struct {
			Message struct {
				Content   *string           `json:"content"`
				ToolCalls []models.ToolCall `json:"tool_calls,omitempty"`
			} `json:"message"`
			FinishReason *string `json:"finish_reason"`
		} `json:"choices"`
		Usage *models.Usage `json:"usage,omitempty"`
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&llamaResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	if llamaResp.Error.Message != "" {
		return &models.Response{
			Success:      false,
			Error:        llamaResp.Error.Message,
			ResponseTime: time.Since(start),
		}, nil
	}

	if len(llamaResp.Choices) == 0 {
		return &models.Response{
			Success:      false,
			Error:        "no response choices returned",
			ResponseTime: time.Since(start),
		}, nil
	}

	choice := llamaResp.Choices[0]
	var content string
	if choice.Message.Content != nil {
		content = *choice.Message.Content
	}

	return &models.Response{
		Content:      content,
		Success:      true,
		ResponseTime: time.Since(start),
		ToolCalls:    choice.Message.ToolCalls,
		Usage:        llamaResp.Usage,
		FinishReason: choice.FinishReason,
	}, nil
}

func (c *LlamaCppClient) sendCompletionRequest(ctx context.Context, req models.Request, start time.Time) (*models.Response, error) {
	prompt := c.formatMessagesAsPrompt(req.Messages)

	llamaReq := map[string]interface{}{
		"prompt": prompt,
		"stream": false,
	}

	c.addParameters(llamaReq, req.Options)

	jsonData, err := json.Marshal(llamaReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/completion", c.baseURL)
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	var llamaResp struct {
		Content string `json:"content"`
		Error   struct {
			Message string `json:"message"`
		} `json:"error"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&llamaResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	if llamaResp.Error.Message != "" {
		return &models.Response{
			Success:      false,
			Error:        llamaResp.Error.Message,
			ResponseTime: time.Since(start),
		}, nil
	}

	return &models.Response{
		Content:      llamaResp.Content,
		Success:      true,
		ResponseTime: time.Since(start),
	}, nil
}

func (c *LlamaCppClient) formatMessages(messages []models.Message) []interface{} {
	formatted := make([]interface{}, len(messages))
	for i, msg := range messages {
		msgMap := map[string]interface{}{
			"role": msg.Role,
		}

		// Handle different content types
		if msg.IsTextOnly() {
			msgMap["content"] = msg.GetTextContent()
		} else {
			msgMap["content"] = msg.Content
		}

		// Add optional fields
		if msg.Name != nil {
			msgMap["name"] = *msg.Name
		}
		if len(msg.ToolCalls) > 0 {
			msgMap["tool_calls"] = msg.ToolCalls
		}
		if msg.ToolCallId != nil {
			msgMap["tool_call_id"] = *msg.ToolCallId
		}
		if msg.FunctionCall != nil {
			msgMap["function_call"] = msg.FunctionCall
		}

		formatted[i] = msgMap
	}
	return formatted
}

func (c *LlamaCppClient) formatMessagesAsPrompt(messages []models.Message) string {
	var result string
	for _, msg := range messages {
		content := msg.GetTextContent()
		result += fmt.Sprintf("%s: %s\n", msg.Role, content)
	}
	return result + "assistant: "
}

func (c *LlamaCppClient) addParameters(req map[string]interface{}, params models.ModelParameters) {
	// Basic sampling parameters
	if params.Temperature != nil {
		req["temperature"] = *params.Temperature
	}
	if params.MaxTokens != nil {
		req["n_predict"] = *params.MaxTokens
	}
	if params.MinTokens != nil {
		req["min_tokens"] = *params.MinTokens
	}
	if params.TopP != nil {
		req["top_p"] = *params.TopP
	}
	if params.TopK != nil {
		req["top_k"] = *params.TopK
	}
	if params.MinP != nil {
		req["min_p"] = *params.MinP
	}
	
	// Penalty parameters
	if params.RepetitionPenalty != nil {
		req["repeat_penalty"] = *params.RepetitionPenalty
	}
	if params.PresencePenalty != nil {
		req["presence_penalty"] = *params.PresencePenalty
	}
	if params.FrequencyPenalty != nil {
		req["frequency_penalty"] = *params.FrequencyPenalty
	}
	
	// Seed and stop conditions
	if params.Seed != nil {
		req["seed"] = *params.Seed
	}
	if len(params.Stop) > 0 {
		req["stop"] = params.Stop
	}
	
	// Advanced sampling parameters supported by modern llama.cpp
	if params.Mirostat != nil {
		req["mirostat"] = *params.Mirostat
	}
	if params.MirostatTau != nil {
		req["mirostat_tau"] = *params.MirostatTau
	}
	if params.MirostatEta != nil {
		req["mirostat_eta"] = *params.MirostatEta
	}
	if params.TfsZ != nil {
		req["tfs_z"] = *params.TfsZ
	}
	if params.TypicalP != nil {
		req["typical_p"] = *params.TypicalP
	}
	
	// Request-level processing parameters
	if params.NKeep != nil {
		req["n_keep"] = *params.NKeep
	}
	if params.PenalizeNl != nil {
		req["penalize_nl"] = *params.PenalizeNl
	}
	
	// Additional parameters supported by modern llama.cpp
	if params.N != nil {
		req["n_choices"] = *params.N
	}
	if params.IgnoreEos != nil {
		req["ignore_eos"] = *params.IgnoreEos
	}
	if params.Logprobs != nil {
		req["logprobs"] = *params.Logprobs
	}
	
	// Chat format specification
	if params.ChatFormat != nil {
		req["chat_format"] = *params.ChatFormat
	}
}

func (c *LlamaCppClient) addChatCompletionFeatures(req map[string]interface{}, chatReq models.Request) {
	if len(chatReq.Tools) > 0 {
		req["tools"] = chatReq.Tools
	}
	if chatReq.ToolChoice != nil {
		req["tool_choice"] = chatReq.ToolChoice
	}
	if chatReq.ResponseFormat != nil {
		req["response_format"] = chatReq.ResponseFormat
	}
	if chatReq.Stream != nil {
		req["stream"] = *chatReq.Stream
	}
	if chatReq.StreamOptions != nil {
		req["stream_options"] = chatReq.StreamOptions
	}
}

func (c *LlamaCppClient) HealthCheck(ctx context.Context) error {
	url := fmt.Sprintf("%s/health", c.baseURL)
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return err
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("health check failed with status: %d", resp.StatusCode)
	}
	return nil
}

func (c *LlamaCppClient) GetServerInfo(ctx context.Context) (*models.ServerInfo, error) {
	serverInfo := &models.ServerInfo{
		ServerURL:  c.baseURL,
		ServerType: "llamacpp",
		Timestamp:  float64(time.Now().Unix()),
		Available:  false,
		Config:     make(map[string]interface{}),
		Models:     make(map[string]interface{}),
		Features:   make(map[string]interface{}),
	}

	// Check if server is available
	if err := c.HealthCheck(ctx); err != nil {
		return serverInfo, nil // Return with Available=false
	}
	serverInfo.Available = true

	// Try to fetch props (LlamaCpp server info)
	propsURL := fmt.Sprintf("%s/props", c.baseURL)
	req, err := http.NewRequestWithContext(ctx, "GET", propsURL, nil)
	if err == nil {
		resp, err := c.httpClient.Do(req)
		if err == nil && resp.StatusCode == http.StatusOK {
			defer resp.Body.Close()
			var propsResp map[string]interface{}
			if json.NewDecoder(resp.Body).Decode(&propsResp) == nil {
				serverInfo.Models = propsResp
			}
		}
	}

	// Try to fetch slots info
	slotsURL := fmt.Sprintf("%s/slots", c.baseURL)
	req, err = http.NewRequestWithContext(ctx, "GET", slotsURL, nil)
	if err == nil {
		resp, err := c.httpClient.Do(req)
		if err == nil && resp.StatusCode == http.StatusOK {
			defer resp.Body.Close()
			var slotsResp []map[string]interface{}
			if json.NewDecoder(resp.Body).Decode(&slotsResp) == nil {
				serverInfo.Features["slots"] = slotsResp
			}
		}
	}

	// Add configuration information from our client
	serverInfo.Config = map[string]interface{}{
		"base_url":   c.baseURL,
		"timeout":    c.timeout.String(),
		"model_name": c.config.Model.Name,
		"provider":   c.config.Provider.Name,
		"workers":    c.config.Processing.Workers,
		"batch_size": c.config.Processing.BatchSize,
		"rate_limit": c.config.Processing.RateLimit,
	}

	// Add model parameters
	if c.config.Model.Parameters.Temperature != nil {
		serverInfo.Config["temperature"] = *c.config.Model.Parameters.Temperature
	}
	if c.config.Model.Parameters.MaxTokens != nil {
		serverInfo.Config["max_tokens"] = *c.config.Model.Parameters.MaxTokens
	}
	if c.config.Model.Parameters.TopP != nil {
		serverInfo.Config["top_p"] = *c.config.Model.Parameters.TopP
	}
	if c.config.Model.Parameters.TopK != nil {
		serverInfo.Config["top_k"] = *c.config.Model.Parameters.TopK
	}

	// Add features information
	serverInfo.Features["supports_streaming"] = true
	serverInfo.Features["supports_chat"] = false // LlamaCpp typically uses completion format
	serverInfo.Features["supports_completions"] = true
	serverInfo.Features["supports_embeddings"] = false
	serverInfo.Features["supports_reasoning"] = false
	serverInfo.Features["supports_tool_calling"] = false
	serverInfo.Features["supports_vision"] = false
	serverInfo.Features["api_version"] = "llamacpp"

	return serverInfo, nil
}

func (c *LlamaCppClient) Close() error {
	return nil
}
