package client

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"llm-client/internal/models"
)

type OpenAIClient struct {
	apiKey     string
	httpClient *http.Client
	timeout    time.Duration
	config     *models.Config
}

func NewOpenAIClient(cfg *models.Config) (*OpenAIClient, error) {
	timeout, err := time.ParseDuration(cfg.Provider.Timeout)
	if err != nil {
		return nil, fmt.Errorf("invalid timeout: %w", err)
	}

	return &OpenAIClient{
		apiKey:     cfg.Provider.APIKey,
		httpClient: &http.Client{Timeout: timeout},
		timeout:    timeout,
		config:     cfg,
	}, nil
}

func (c *OpenAIClient) SendRequest(ctx context.Context, req models.Request) (*models.Response, error) {
	start := time.Now()

	openaiReq := map[string]interface{}{
		"model":    c.config.Model.Name,
		"messages": c.formatMessages(req.Messages),
	}

	c.addParameters(openaiReq, req.Options)
	c.addChatCompletionFeatures(openaiReq, req)

	jsonData, err := json.Marshal(openaiReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", "https://api.openai.com/v1/chat/completions", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.apiKey))

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	var openaiResp struct {
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

	if err := json.NewDecoder(resp.Body).Decode(&openaiResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	if openaiResp.Error.Message != "" {
		return &models.Response{
			Success:      false,
			Error:        openaiResp.Error.Message,
			ResponseTime: time.Since(start),
		}, nil
	}

	if len(openaiResp.Choices) == 0 {
		return &models.Response{
			Success:      false,
			Error:        "no response choices returned",
			ResponseTime: time.Since(start),
		}, nil
	}

	choice := openaiResp.Choices[0]
	var content string
	if choice.Message.Content != nil {
		content = *choice.Message.Content
	}

	return &models.Response{
		Content:      content,
		Success:      true,
		ResponseTime: time.Since(start),
		ToolCalls:    choice.Message.ToolCalls,
		Usage:        openaiResp.Usage,
		FinishReason: choice.FinishReason,
	}, nil
}

func (c *OpenAIClient) formatMessages(messages []models.Message) []interface{} {
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

func (c *OpenAIClient) addParameters(req map[string]interface{}, params models.ModelParameters) {
	if params.Temperature != nil {
		req["temperature"] = *params.Temperature
	}
	if params.MaxTokens != nil {
		req["max_tokens"] = *params.MaxTokens
	}
	if params.TopP != nil {
		req["top_p"] = *params.TopP
	}
	if params.Seed != nil {
		req["seed"] = *params.Seed
	}
	if len(params.Stop) > 0 {
		req["stop"] = params.Stop
	}
	if params.PresencePenalty != nil {
		req["presence_penalty"] = *params.PresencePenalty
	}
	if params.FrequencyPenalty != nil {
		req["frequency_penalty"] = *params.FrequencyPenalty
	}
	if params.N != nil {
		req["n"] = *params.N
	}
	if params.Logprobs != nil {
		req["logprobs"] = *params.Logprobs
	}
}

func (c *OpenAIClient) addChatCompletionFeatures(req map[string]interface{}, chatReq models.Request) {
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

func (c *OpenAIClient) HealthCheck(ctx context.Context) error {
	testReq := map[string]interface{}{
		"model": "gpt-3.5-turbo",
		"messages": []map[string]string{
			{"role": "user", "content": "test"},
		},
		"max_tokens": 1,
	}

	jsonData, err := json.Marshal(testReq)
	if err != nil {
		return err
	}

	req, err := http.NewRequestWithContext(context.Background(), "POST", "https://api.openai.com/v1/chat/completions", bytes.NewBuffer(jsonData))
	if err != nil {
		return err
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.apiKey))

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	return nil
}

func (c *OpenAIClient) GetServerInfo(ctx context.Context) (*models.ServerInfo, error) {
	serverInfo := &models.ServerInfo{
		ServerURL:  "https://api.openai.com",
		ServerType: "OpenAI",
		Timestamp:  float64(time.Now().Unix()),
		Available:  true,
		Features: map[string]interface{}{
			"chat_completions": true,
			"tool_calling":     true,
			"function_calling": true,
			"multimodal":       true,
			"streaming":        true,
			"json_mode":        true,
		},
	}
	return serverInfo, nil
}

func (c *OpenAIClient) Close() error {
	return nil
}
