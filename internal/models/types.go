package models

import (
	"context"
	"time"
)

type Config struct {
	Provider       ProviderConfig       `yaml:"provider"`
	Model          ModelConfig          `yaml:"model"`
	Classification ClassificationConfig `yaml:"classification"`
	Processing     ProcessingConfig     `yaml:"processing"`
	Output         OutputConfig         `yaml:"output"`
	Reference      ReferenceConfig      `yaml:"reference,omitempty"`
}

type ProviderConfig struct {
	Name    string `yaml:"name"`
	BaseURL string `yaml:"base_url"`
	Timeout string `yaml:"timeout"`
	APIKey  string `yaml:"api_key"`
}

type ModelConfig struct {
	Name       string          `yaml:"name"`
	Parameters ModelParameters `yaml:"parameters"`
}

type ModelParameters struct {
	// Basic sampling parameters
	Temperature                *float64 `yaml:"temperature,omitempty"`
	MaxTokens                  *int     `yaml:"max_tokens,omitempty"`
	MinTokens                  *int     `yaml:"min_tokens,omitempty"`
	TopP                       *float64 `yaml:"top_p,omitempty"`
	TopK                       *int     `yaml:"top_k,omitempty"`
	MinP                       *float64 `yaml:"min_p,omitempty"`
	RepetitionPenalty          *float64 `yaml:"repetition_penalty,omitempty"`
	PresencePenalty            *float64 `yaml:"presence_penalty,omitempty"`
	FrequencyPenalty           *float64 `yaml:"frequency_penalty,omitempty"`
	Seed                       *int64   `yaml:"seed,omitempty"`
	N                          *int     `yaml:"n,omitempty"`
	Stop                       []string `yaml:"stop,omitempty"`
	StopTokenIds               []int    `yaml:"stop_token_ids,omitempty"`
	BadWords                   []string `yaml:"bad_words,omitempty"`
	IncludeStopStrInOutput     *bool    `yaml:"include_stop_str_in_output,omitempty"`
	IgnoreEos                  *bool    `yaml:"ignore_eos,omitempty"`
	Logprobs                   *int     `yaml:"logprobs,omitempty"`
	PromptLogprobs             *int     `yaml:"prompt_logprobs,omitempty"`
	TruncatePromptTokens       *int     `yaml:"truncate_prompt_tokens,omitempty"`
	ChatFormat                 *string  `yaml:"chat_format,omitempty"`
	SkipSpecialTokens          *bool    `yaml:"skip_special_tokens,omitempty"`
	SpacesBetweenSpecialTokens *bool    `yaml:"spaces_between_special_tokens,omitempty"`
	EnableThinking             *bool    `yaml:"enable_thinking,omitempty"`

	// vLLM Guided Generation Parameters
	GuidedChoice              []string               `yaml:"guided_choice,omitempty"`
	GuidedRegex               *string                `yaml:"guided_regex,omitempty"`
	GuidedJSON                map[string]interface{} `yaml:"guided_json,omitempty"`
	GuidedGrammar             *string                `yaml:"guided_grammar,omitempty"`
	GuidedWhitespacePattern   *string                `yaml:"guided_whitespace_pattern,omitempty"`
	GuidedDecodingBackend     *string                `yaml:"guided_decoding_backend,omitempty"`

	// Additional vLLM Parameters
	MaxLogprobs               *int     `yaml:"max_logprobs,omitempty"`
	Echo                      *bool    `yaml:"echo,omitempty"`
	BestOf                    *int     `yaml:"best_of,omitempty"`
	UseBeamSearch             *bool    `yaml:"use_beam_search,omitempty"`
	LengthPenalty             *float64 `yaml:"length_penalty,omitempty"`
	EarlyStopping             *bool    `yaml:"early_stopping,omitempty"`

	// llama.cpp Sampling Parameters
	Mirostat                  *int     `yaml:"mirostat,omitempty"`
	MirostatTau               *float64 `yaml:"mirostat_tau,omitempty"`
	MirostatEta               *float64 `yaml:"mirostat_eta,omitempty"`
	TfsZ                      *float64 `yaml:"tfs_z,omitempty"`
	TypicalP                  *float64 `yaml:"typical_p,omitempty"`

	// llama.cpp Request-level Parameters
	NKeep                     *int  `yaml:"n_keep,omitempty"`
	PenalizeNl                *bool `yaml:"penalize_nl,omitempty"`
}

type ClassificationConfig struct {
	Template     TemplateConfig      `yaml:"template"`
	Parsing      ParsingConfig       `yaml:"parsing"`
	FieldMapping *FieldMappingConfig `yaml:"field_mapping,omitempty"`
}

type FieldMappingConfig struct {
	InputTextField string            `yaml:"input_text_field,omitempty"`
	PlaceholderMap map[string]string `yaml:"placeholder_map,omitempty"`
}

type TemplateConfig struct {
	System string `yaml:"system"`
	User   string `yaml:"user"`
}

type ParsingConfig struct {
	Find             []string          `yaml:"find"`
	Default          string            `yaml:"default"`
	Fallback         string            `yaml:"fallback"`
	Map              map[string]string `yaml:"map"`
	ThinkingTags     string            `yaml:"thinking_tags"`
	PreserveThinking *bool             `yaml:"preserve_thinking,omitempty"`
	AnswerPatterns   []string          `yaml:"answer_patterns"`
	CaseSensitive    *bool             `yaml:"case_sensitive,omitempty"`
	ExactMatch       *bool             `yaml:"exact_match,omitempty"`
}

type ProcessingConfig struct {
	Workers        int          `yaml:"workers"`
	BatchSize      int          `yaml:"batch_size"`
	Repeat         int          `yaml:"repeat"`
	RateLimit      bool         `yaml:"rate_limit"`
	FlashInferSafe *bool        `yaml:"flashinfer_safe,omitempty"`
	LiveMetrics    *LiveMetrics `yaml:"live_metrics,omitempty"`
	MinimalMode    bool         `yaml:"minimal_mode,omitempty"`
}

type LiveMetrics struct {
	Enabled     bool     `yaml:"enabled"`
	Metric      string   `yaml:"metric"`
	GroundTruth string   `yaml:"ground_truth"`
	Average     string   `yaml:"average,omitempty"`
	Classes     []string `yaml:"classes,omitempty"`
}

type OutputConfig struct {
	Directory          string `yaml:"directory"`
	Format             string `yaml:"format"`
	InputTextField     string `yaml:"input_text_field,omitempty"`
	IncludeRawResponse bool   `yaml:"include_raw_response,omitempty"`
	IncludeThinking    bool   `yaml:"include_thinking,omitempty"`
	StreamOutput       bool   `yaml:"stream_output,omitempty"`
	StreamSaveEvery    int    `yaml:"stream_save_every,omitempty"`
}

type ReferenceConfig struct {
	File        string `yaml:"file,omitempty"`        // Path to reference/ground truth file
	Column      string `yaml:"column,omitempty"`      // Column name containing reference values
	Format      string `yaml:"format,omitempty"`      // File format (csv, json, xlsx, parquet)
	IndexColumn string `yaml:"index_column,omitempty"` // Column to match with input data index
}

// Enhanced Message struct with support for different content types and tool calls
type Message struct {
	Role         string        `json:"role"`
	Content      interface{}   `json:"content"`
	Name         *string       `json:"name,omitempty"`
	ToolCalls    []ToolCall    `json:"tool_calls,omitempty"`
	ToolCallId   *string       `json:"tool_call_id,omitempty"`
	FunctionCall *FunctionCall `json:"function_call,omitempty"`
}

// Content types for multimodal support
type TextContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type ImageContent struct {
	Type     string    `json:"type"`
	ImageUrl *ImageUrl `json:"image_url,omitempty"`
}

type ImageUrl struct {
	Url    string  `json:"url"`
	Detail *string `json:"detail,omitempty"`
}

// Tool calling support
type ToolCall struct {
	Id       string   `json:"id"`
	Type     string   `json:"type"`
	Function Function `json:"function"`
}

type Function struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// Tool definition for function calling
type Tool struct {
	Type     string             `json:"type"`
	Function FunctionDefinition `json:"function"`
}

type FunctionDefinition struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Parameters  map[string]interface{} `json:"parameters"`
}

// Helper methods for Message
func NewTextMessage(role, content string) Message {
	return Message{
		Role:    role,
		Content: content,
	}
}

func NewMultiModalMessage(role string, contents []interface{}) Message {
	return Message{
		Role:    role,
		Content: contents,
	}
}

func NewToolMessage(toolCallId, content string) Message {
	return Message{
		Role:       "tool",
		Content:    content,
		ToolCallId: &toolCallId,
	}
}

func (m *Message) GetTextContent() string {
	switch content := m.Content.(type) {
	case string:
		return content
	case []interface{}:
		for _, item := range content {
			if textContent, ok := item.(map[string]interface{}); ok {
				if textContent["type"] == "text" {
					if text, exists := textContent["text"]; exists {
						if textStr, ok := text.(string); ok {
							return textStr
						}
					}
				}
			}
		}
	}
	return ""
}

func (m *Message) IsTextOnly() bool {
	_, ok := m.Content.(string)
	return ok
}

func (m *Message) HasToolCalls() bool {
	return len(m.ToolCalls) > 0
}

type Request struct {
	Messages       []Message       `json:"messages"`
	Options        ModelParameters `json:",inline"`
	Tools          []Tool          `json:"tools,omitempty"`
	ToolChoice     interface{}     `json:"tool_choice,omitempty"`
	ResponseFormat *ResponseFormat `json:"response_format,omitempty"`
	Stream         *bool           `json:"stream,omitempty"`
	StreamOptions  *StreamOptions  `json:"stream_options,omitempty"`
}

type ResponseFormat struct {
	Type   string                 `json:"type"`
	Schema map[string]interface{} `json:"schema,omitempty"`
}

type StreamOptions struct {
	IncludeUsage *bool `json:"include_usage,omitempty"`
}

type Response struct {
	Content      string        `json:"content"`
	Success      bool          `json:"success"`
	Error        string        `json:"error,omitempty"`
	ResponseTime time.Duration `json:"response_time"`
	ToolCalls    []ToolCall    `json:"tool_calls,omitempty"`
	Usage        *Usage        `json:"usage,omitempty"`
	FinishReason *string       `json:"finish_reason,omitempty"`
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type DataRow struct {
	Index int                    `json:"index"`
	Text  string                 `json:"text"`
	Data  map[string]interface{} `json:"data"`
}

type Result struct {
	Index           int                    `json:"index"`
	InputText       string                 `json:"input_text"`
	OriginalData    map[string]interface{} `json:"original_data,omitempty"`
	GroundTruth     string                 `json:"ground_truth"`
	FinalAnswer     string                 `json:"final_answer"`
	RawResponse     string                 `json:"raw_response"`
	ThinkingContent string                 `json:"thinking_content,omitempty"`
	Success         bool                   `json:"success"`
	Error           string                 `json:"error,omitempty"`
	ResponseTime    time.Duration          `json:"response_time"`
	Attempts        []Attempt              `json:"attempts,omitempty"`
	Consensus       *Consensus             `json:"consensus,omitempty"`
	ToolCalls       []ToolCall             `json:"tool_calls,omitempty"`
	Usage           *Usage                 `json:"usage,omitempty"`
}

type Attempt struct {
	Response     string        `json:"response"`
	Answer       string        `json:"answer"`
	ResponseTime time.Duration `json:"response_time"`
	Success      bool          `json:"success"`
	Error        string        `json:"error,omitempty"`
	ToolCalls    []ToolCall    `json:"tool_calls,omitempty"`
}

type Consensus struct {
	FinalAnswer  string         `json:"final_answer"`
	Count        int            `json:"count"`
	Total        int            `json:"total"`
	Ratio        float64        `json:"ratio"`
	Distribution map[string]int `json:"distribution"`
}

type Client interface {
	SendRequest(ctx context.Context, req Request) (*Response, error)
	HealthCheck(ctx context.Context) error
	GetServerInfo(ctx context.Context) (*ServerInfo, error)
	Close() error
}

type ServerInfo struct {
	ServerURL  string                 `json:"server_url"`
	ServerType string                 `json:"server_type"`
	Timestamp  float64                `json:"timestamp"`
	Config     map[string]interface{} `json:"config,omitempty"`
	Models     map[string]interface{} `json:"models,omitempty"`
	Features   map[string]interface{} `json:"features,omitempty"`
	Available  bool                   `json:"available"`
}

// Resume state structures for restart functionality
type ResumeState struct {
	ConfigFile      string        `json:"config_file"`
	InputFile       string        `json:"input_file"`
	OutputDirectory string        `json:"output_directory"`
	ProcessedItems  []int         `json:"processed_items"`
	CompletedCount  int           `json:"completed_count"`
	TotalCount      int           `json:"total_count"`
	Results         []Result      `json:"results"`
	Timestamp       time.Time     `json:"timestamp"`
	Options         ResumeOptions `json:"options"`
}

type ResumeOptions struct {
	Workers      int  `json:"workers"`
	Repeat       int  `json:"repeat"`
	Limit        int  `json:"limit"`
	ShowProgress bool `json:"show_progress"`
	Verbose      bool `json:"verbose"`
}
