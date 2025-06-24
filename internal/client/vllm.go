package client

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"

	"github.com/Vitruves/llm-client/internal/models"
)

var (
	// Pool for JSON encoding buffers to reduce allocations
	bufferPool = sync.Pool{
		New: func() interface{} {
			return &bytes.Buffer{}
		},
	}
)

type VLLMClient struct {
	baseURL    string
	httpClient *http.Client
	timeout    time.Duration
	config     *models.Config
	requestCh  chan requestItem
}

type requestItem struct {
	req    *http.Request
	respCh chan *http.Response
	errCh  chan error
	ctx    context.Context
}

func NewVLLMClient(cfg *models.Config) (*VLLMClient, error) {
	timeout, err := time.ParseDuration(cfg.Provider.Timeout)
	if err != nil {
		return nil, fmt.Errorf("invalid timeout: %w", err)
	}

	// Optimize transport settings based on worker count from config
	maxConns := cfg.Processing.Workers * 2

	transport := &http.Transport{
		MaxIdleConns:          maxConns,
		MaxIdleConnsPerHost:   maxConns,
		IdleConnTimeout:       30 * time.Second,
		DisableKeepAlives:     false,
		MaxConnsPerHost:       maxConns,
		ResponseHeaderTimeout: timeout,
		TLSHandshakeTimeout:   10 * time.Second,
		ExpectContinueTimeout: 1 * time.Second,
		ForceAttemptHTTP2:     false,
	}

	// Optimize request channel buffer based on worker count from config
	bufferSize := cfg.Processing.Workers * 4

	client := &VLLMClient{
		baseURL: cfg.Provider.BaseURL,
		httpClient: &http.Client{
			Transport: transport,
			Timeout:   timeout,
		},
		timeout:   timeout,
		config:    cfg,
		requestCh: make(chan requestItem, bufferSize),
	}

	if cfg.Processing.RateLimit {
		go client.rateLimitedRequestProcessor()
	}

	return client, nil
}

func (c *VLLMClient) rateLimitedRequestProcessor() {
	ticker := time.NewTicker(10 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case item := <-c.requestCh:
			select {
			case <-ticker.C:
			case <-item.ctx.Done():
				item.errCh <- item.ctx.Err()
				continue
			}

			resp, err := c.httpClient.Do(item.req)
			if err != nil {
				item.errCh <- err
			} else {
				item.respCh <- resp
			}

		case <-time.After(100 * time.Millisecond):
			continue
		}
	}
}

func (c *VLLMClient) SendRequest(ctx context.Context, req models.Request) (*models.Response, error) {
	start := time.Now()

	vllmReq := map[string]interface{}{
		"messages": c.formatMessages(req.Messages),
		"stream":   false,
	}

	c.addParameters(vllmReq, req.Options)
	c.addChatCompletionFeatures(vllmReq, req)

	// Use buffer pool for better performance
	buf := bufferPool.Get().(*bytes.Buffer)
	buf.Reset()
	defer bufferPool.Put(buf)

	if err := json.NewEncoder(buf).Encode(vllmReq); err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/v1/chat/completions", c.baseURL)
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(buf.Bytes()))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")

	var resp *http.Response
	if c.config.Processing.RateLimit {
		resp, err = c.sendRateLimited(ctx, httpReq)
	} else {
		resp, err = c.httpClient.Do(httpReq)
	}

	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	var vllmResp struct {
		Choices []struct {
			Message struct {
				Content          *string           `json:"content"`
				ReasoningContent *string           `json:"reasoning_content"`
				ToolCalls        []models.ToolCall `json:"tool_calls,omitempty"`
			} `json:"message"`
			FinishReason *string `json:"finish_reason"`
		} `json:"choices"`
		Usage *models.Usage `json:"usage,omitempty"`
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&vllmResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	if vllmResp.Error.Message != "" {
		return &models.Response{
			Success:      false,
			Error:        vllmResp.Error.Message,
			ResponseTime: time.Since(start),
		}, nil
	}

	if len(vllmResp.Choices) == 0 {
		return &models.Response{
			Success:      false,
			Error:        "no response choices returned",
			ResponseTime: time.Since(start),
		}, nil
	}

	choice := vllmResp.Choices[0]
	var finalContent string

	if choice.Message.Content != nil {
		finalContent = *choice.Message.Content
	}

	if choice.Message.ReasoningContent != nil && *choice.Message.ReasoningContent != "" {
		reasoningContent := *choice.Message.ReasoningContent

		// Format reasoning content for parser to handle
		if finalContent == "" {
			// When content is null, put reasoning content in thinking tags
			// The parser will extract the final answer using configured patterns
			finalContent = fmt.Sprintf("<think>\n%s\n</think>", reasoningContent)
		} else {
			finalContent = fmt.Sprintf("<think>\n%s\n</think>\n\n%s", reasoningContent, finalContent)
		}
	}

	return &models.Response{
		Content:      finalContent,
		Success:      true,
		ResponseTime: time.Since(start),
		ToolCalls:    choice.Message.ToolCalls,
		Usage:        vllmResp.Usage,
		FinishReason: choice.FinishReason,
	}, nil
}

func (c *VLLMClient) sendRateLimited(ctx context.Context, req *http.Request) (*http.Response, error) {
	respCh := make(chan *http.Response, 1)
	errCh := make(chan error, 1)

	item := requestItem{
		req:    req,
		respCh: respCh,
		errCh:  errCh,
		ctx:    ctx,
	}

	select {
	case c.requestCh <- item:
	case <-ctx.Done():
		return nil, ctx.Err()
	}

	select {
	case resp := <-respCh:
		return resp, nil
	case err := <-errCh:
		return nil, err
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func (c *VLLMClient) addParameters(req map[string]interface{}, params models.ModelParameters) {
	flashInferSafe := c.config.Processing.FlashInferSafe != nil && *c.config.Processing.FlashInferSafe

	// Basic sampling parameters
	if params.Temperature != nil {
		req["temperature"] = *params.Temperature
	}
	if params.MaxTokens != nil {
		req["max_tokens"] = *params.MaxTokens
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
	if params.N != nil {
		req["n"] = *params.N
	}

	// Penalty parameters
	if params.RepetitionPenalty != nil {
		req["repetition_penalty"] = *params.RepetitionPenalty
	}
	if params.PresencePenalty != nil {
		req["presence_penalty"] = *params.PresencePenalty
	}
	if params.FrequencyPenalty != nil {
		req["frequency_penalty"] = *params.FrequencyPenalty
	}

	// Seed (only if FlashInfer safe mode is disabled)
	if !flashInferSafe && params.Seed != nil {
		req["seed"] = *params.Seed
	}

	// Stop conditions
	if len(params.Stop) > 0 {
		req["stop"] = params.Stop
	}
	if len(params.StopTokenIds) > 0 {
		req["stop_token_ids"] = params.StopTokenIds
	}
	if len(params.BadWords) > 0 {
		req["bad_words"] = params.BadWords
	}

	// Output control
	if params.IncludeStopStrInOutput != nil {
		req["include_stop_str_in_output"] = *params.IncludeStopStrInOutput
	}
	if params.IgnoreEos != nil {
		req["ignore_eos"] = *params.IgnoreEos
	}

	// Logprobs
	if params.Logprobs != nil {
		req["logprobs"] = *params.Logprobs
	}
	if params.PromptLogprobs != nil {
		req["prompt_logprobs"] = *params.PromptLogprobs
	}

	// Token handling
	if params.TruncatePromptTokens != nil {
		req["truncate_prompt_tokens"] = *params.TruncatePromptTokens
	}
	if params.SkipSpecialTokens != nil {
		req["skip_special_tokens"] = *params.SkipSpecialTokens
	}
	if params.SpacesBetweenSpecialTokens != nil {
		req["spaces_between_special_tokens"] = *params.SpacesBetweenSpecialTokens
	}
	
	// Qwen3 thinking mode (vLLM specific)
	// According to vLLM docs, enable_thinking must be in extra_body.chat_template_kwargs
	if params.EnableThinking != nil {
		extraBody := make(map[string]interface{})
		if existing, ok := req["extra_body"].(map[string]interface{}); ok {
			extraBody = existing
		}
		
		chatTemplateKwargs := make(map[string]interface{})
		if existing, ok := extraBody["chat_template_kwargs"].(map[string]interface{}); ok {
			chatTemplateKwargs = existing
		}
		
		chatTemplateKwargs["enable_thinking"] = *params.EnableThinking
		extraBody["chat_template_kwargs"] = chatTemplateKwargs
		req["extra_body"] = extraBody
	}

	// vLLM Guided Generation Parameters
	if len(params.GuidedChoice) > 0 {
		req["guided_choice"] = params.GuidedChoice
	}
	if params.GuidedRegex != nil {
		req["guided_regex"] = *params.GuidedRegex
	}
	if params.GuidedJSON != nil {
		req["guided_json"] = params.GuidedJSON
	}
	if params.GuidedGrammar != nil {
		req["guided_grammar"] = *params.GuidedGrammar
	}
	if params.GuidedWhitespacePattern != nil {
		req["guided_whitespace_pattern"] = *params.GuidedWhitespacePattern
	}
	if params.GuidedDecodingBackend != nil {
		req["guided_decoding_backend"] = *params.GuidedDecodingBackend
	}

	// Additional vLLM Parameters
	if params.MaxLogprobs != nil {
		req["max_logprobs"] = *params.MaxLogprobs
	}
	if params.Echo != nil {
		req["echo"] = *params.Echo
	}
	if params.BestOf != nil {
		req["best_of"] = *params.BestOf
	}
	if params.UseBeamSearch != nil {
		req["use_beam_search"] = *params.UseBeamSearch
	}
	if params.LengthPenalty != nil {
		req["length_penalty"] = *params.LengthPenalty
	}
	if params.EarlyStopping != nil {
		req["early_stopping"] = *params.EarlyStopping
	}
}

func (c *VLLMClient) formatMessages(messages []models.Message) []interface{} {
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

func (c *VLLMClient) addChatCompletionFeatures(req map[string]interface{}, chatReq models.Request) {
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

func (c *VLLMClient) HealthCheck(ctx context.Context) error {
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

func (c *VLLMClient) GetServerInfo(ctx context.Context) (*models.ServerInfo, error) {
	serverInfo := &models.ServerInfo{
		ServerURL:  c.baseURL,
		ServerType: "vllm",
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

	// Fetch models information
	modelsURL := fmt.Sprintf("%s/v1/models", c.baseURL)
	req, err := http.NewRequestWithContext(ctx, "GET", modelsURL, nil)
	if err == nil {
		resp, err := c.httpClient.Do(req)
		if err == nil && resp.StatusCode == http.StatusOK {
			defer resp.Body.Close()
			var modelsResp struct {
				Data []struct {
					ID          string        `json:"id"`
					Object      string        `json:"object"`
					Created     int64         `json:"created"`
					OwnedBy     string        `json:"owned_by"`
					Root        string        `json:"root"`
					Parent      string        `json:"parent"`
					MaxModelLen int           `json:"max_model_len"`
					Permission  []interface{} `json:"permission"`
				} `json:"data"`
			}
			if json.NewDecoder(resp.Body).Decode(&modelsResp) == nil && len(modelsResp.Data) > 0 {
				model := modelsResp.Data[0]
				serverInfo.Models = map[string]interface{}{
					"model_name":    model.ID,
					"max_model_len": model.MaxModelLen,
					"created":       model.Created,
					"owned_by":      model.OwnedBy,
					"root":          model.Root,
					"parent":        model.Parent,
				}
			}
		}
	}

	// Try to fetch version info (some vLLM servers expose this)
	versionURL := fmt.Sprintf("%s/version", c.baseURL)
	req, err = http.NewRequestWithContext(ctx, "GET", versionURL, nil)
	if err == nil {
		resp, err := c.httpClient.Do(req)
		if err == nil && resp.StatusCode == http.StatusOK {
			defer resp.Body.Close()
			var versionResp map[string]interface{}
			if json.NewDecoder(resp.Body).Decode(&versionResp) == nil {
				serverInfo.Features["version_info"] = versionResp
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
	serverInfo.Features = map[string]interface{}{
		"supports_streaming":    true,
		"supports_chat":         true,
		"supports_completions":  true,
		"supports_embeddings":   false, // vLLM can support embeddings but depends on model
		"supports_reasoning":    true,  // vLLM supports reasoning models
		"supports_tool_calling": true,
		"supports_vision":       false, // Depends on model
		"api_version":           "openai_compatible",
	}

	return serverInfo, nil
}

func (c *VLLMClient) Close() error {
	return nil
}
