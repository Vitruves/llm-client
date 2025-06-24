package config

import (
	"os"
	"path/filepath"
	"testing"

	"llm-client/internal/models"
)

func TestLoad(t *testing.T) {
	tests := []struct {
		name        string
		configYAML  string
		expectError bool
		validate    func(*models.Config) error
	}{
		{
			name: "valid vLLM config",
			configYAML: `
provider:
  name: vllm
  base_url: http://localhost:8000/v1
  timeout: 60s

model:
  name: test-model
  parameters:
    temperature: 0.7
    max_tokens: 100

classification:
  template:
    system: "You are a helpful assistant"
    user: "Classify: {text}"
  parsing:
    find: ["positive", "negative", "neutral"]
    default: "unknown"
    fallback: "error"

processing:
  workers: 4
  batch_size: 1
  repeat: 1
  rate_limit: false

output:
  directory: "./output"
  format: json
`,
			expectError: false,
			validate: func(cfg *models.Config) error {
				if cfg.Provider.Name != "vllm" {
					t.Errorf("Expected provider name 'vllm', got '%s'", cfg.Provider.Name)
				}
				if cfg.Model.Parameters.Temperature == nil || *cfg.Model.Parameters.Temperature != 0.7 {
					t.Errorf("Expected temperature 0.7, got %v", cfg.Model.Parameters.Temperature)
				}
				return nil
			},
		},
		{
			name: "valid llama.cpp config with advanced parameters",
			configYAML: `
provider:
  name: llamacpp
  base_url: http://localhost:8080
  timeout: 120s

model:
  name: llama-model
  parameters:
    temperature: 0.5
    max_tokens: 200
    mirostat: 2
    mirostat_tau: 5.0
    mirostat_eta: 0.1
    tfs_z: 1.0
    typical_p: 1.0

classification:
  template:
    system: "You are a classifier"
    user: "Text: {text}"
  parsing:
    find: ["class1", "class2"]
    default: "unknown"
    fallback: "error"

processing:
  workers: 2
  batch_size: 1
  repeat: 1

output:
  directory: "./results"
  format: csv
`,
			expectError: false,
			validate: func(cfg *models.Config) error {
				if cfg.Provider.Name != "llamacpp" {
					t.Errorf("Expected provider name 'llamacpp', got '%s'", cfg.Provider.Name)
				}
				if cfg.Model.Parameters.Mirostat == nil || *cfg.Model.Parameters.Mirostat != 2 {
					t.Errorf("Expected mirostat 2, got %v", cfg.Model.Parameters.Mirostat)
				}
				return nil
			},
		},
		{
			name: "valid OpenAI config",
			configYAML: `
provider:
  name: openai
  api_key: sk-test123

model:
  name: gpt-3.5-turbo
  parameters:
    temperature: 0.8
    max_tokens: 150
    presence_penalty: 0.1
    frequency_penalty: 0.2

classification:
  template:
    system: "Classify the following"
    user: "{text}"
  parsing:
    find: ["positive", "negative"]
    default: "neutral"
    fallback: "error"

processing:
  workers: 1
  batch_size: 1
  repeat: 1

output:
  format: json
`,
			expectError: false,
			validate: func(cfg *models.Config) error {
				if cfg.Provider.Name != "openai" {
					t.Errorf("Expected provider name 'openai', got '%s'", cfg.Provider.Name)
				}
				if cfg.Provider.APIKey != "sk-test123" {
					t.Errorf("Expected API key 'sk-test123', got '%s'", cfg.Provider.APIKey)
				}
				return nil
			},
		},
		{
			name: "vLLM with guided generation",
			configYAML: `
provider:
  name: vllm
  base_url: http://localhost:8000/v1

model:
  name: test-model
  parameters:
    temperature: 0.0
    guided_choice: ["positive", "negative", "neutral"]
    guided_regex: "^(positive|negative|neutral)$"
    use_beam_search: true
    best_of: 5

classification:
  template:
    system: "Classify sentiment"
    user: "{text}"
  parsing:
    find: ["positive", "negative", "neutral"]
    default: "unknown"
    fallback: "error"

processing:
  workers: 1
  batch_size: 1
  repeat: 1

output:
  format: json
`,
			expectError: false,
			validate: func(cfg *models.Config) error {
				if len(cfg.Model.Parameters.GuidedChoice) != 3 {
					t.Errorf("Expected 3 guided choices, got %d", len(cfg.Model.Parameters.GuidedChoice))
				}
				if cfg.Model.Parameters.GuidedRegex == nil {
					t.Error("Expected guided regex to be set")
				}
				return nil
			},
		},
		{
			name: "missing provider name",
			configYAML: `
provider:
  base_url: http://localhost:8000
model:
  name: test
classification:
  template:
    system: "test"
    user: "test"
  parsing:
    find: ["test"]
    default: "test"
    fallback: "test"
processing:
  workers: 1
  batch_size: 1
  repeat: 1
output:
  format: json
`,
			expectError: true,
		},
		{
			name: "invalid provider",
			configYAML: `
provider:
  name: invalid-provider
  base_url: http://localhost:8000
model:
  name: test
classification:
  template:
    system: "test"
    user: "test"
  parsing:
    find: ["test"]
    default: "test"
    fallback: "test"
processing:
  workers: 1
  batch_size: 1
  repeat: 1
output:
  format: json
`,
			expectError: true,
		},
		{
			name: "missing base_url for vLLM",
			configYAML: `
provider:
  name: vllm
model:
  name: test
classification:
  template:
    system: "test"
    user: "test"
  parsing:
    find: ["test"]
    default: "test"
    fallback: "test"
processing:
  workers: 1
  batch_size: 1
  repeat: 1
output:
  format: json
`,
			expectError: true,
		},
		{
			name: "missing api_key for OpenAI",
			configYAML: `
provider:
  name: openai
model:
  name: gpt-3.5-turbo
classification:
  template:
    system: "test"
    user: "test"
  parsing:
    find: ["test"]
    default: "test"
    fallback: "test"
processing:
  workers: 1
  batch_size: 1
  repeat: 1
output:
  format: json
`,
			expectError: true,
		},
		{
			name: "invalid output format",
			configYAML: `
provider:
  name: vllm
  base_url: http://localhost:8000
model:
  name: test
classification:
  template:
    system: "test"
    user: "test"
  parsing:
    find: ["test"]
    default: "test"
    fallback: "test"
processing:
  workers: 1
  batch_size: 1
  repeat: 1
output:
  format: invalid
`,
			expectError: true,
		},
		{
			name: "invalid repeat count",
			configYAML: `
provider:
  name: vllm
  base_url: http://localhost:8000
model:
  name: test
classification:
  template:
    system: "test"
    user: "test"
  parsing:
    find: ["test"]
    default: "test"
    fallback: "test"
processing:
  workers: 1
  batch_size: 1
  repeat: 15
output:
  format: json
`,
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create temporary config file
			tmpDir := t.TempDir()
			configFile := filepath.Join(tmpDir, "config.yaml")
			
			err := os.WriteFile(configFile, []byte(tt.configYAML), 0644)
			if err != nil {
				t.Fatalf("Failed to write test config file: %v", err)
			}

			// Load config
			cfg, err := Load(configFile)
			
			if tt.expectError {
				if err == nil {
					t.Errorf("Expected error but got none")
				}
				return
			}

			if err != nil {
				t.Errorf("Unexpected error: %v", err)
				return
			}

			// Run custom validation if provided
			if tt.validate != nil {
				if err := tt.validate(cfg); err != nil {
					t.Errorf("Custom validation failed: %v", err)
				}
			}
		})
	}
}

func TestSetDefaults(t *testing.T) {
	cfg := &models.Config{}
	
	setDefaults(cfg)
	
	if cfg.Provider.Timeout != "60s" {
		t.Errorf("Expected default timeout '60s', got '%s'", cfg.Provider.Timeout)
	}
	
	if cfg.Processing.Workers != 4 {
		t.Errorf("Expected default workers 4, got %d", cfg.Processing.Workers)
	}
	
	if cfg.Processing.BatchSize != 1 {
		t.Errorf("Expected default batch size 1, got %d", cfg.Processing.BatchSize)
	}
	
	if cfg.Processing.Repeat != 1 {
		t.Errorf("Expected default repeat 1, got %d", cfg.Processing.Repeat)
	}
	
	if cfg.Output.Directory != "./output" {
		t.Errorf("Expected default output directory './output', got '%s'", cfg.Output.Directory)
	}
	
	if cfg.Output.Format != "json" {
		t.Errorf("Expected default output format 'json', got '%s'", cfg.Output.Format)
	}
}

func TestEnvironmentVariableExpansion(t *testing.T) {
	// Set environment variable for test
	os.Setenv("TEST_API_KEY", "test-key-123")
	defer os.Unsetenv("TEST_API_KEY")
	
	configYAML := `
provider:
  name: openai
  api_key: ${TEST_API_KEY}
model:
  name: gpt-3.5-turbo
classification:
  template:
    system: "test"
    user: "test"
  parsing:
    find: ["test"]
    default: "test"
    fallback: "test"
processing:
  workers: 1
  batch_size: 1
  repeat: 1
output:
  format: json
`

	tmpDir := t.TempDir()
	configFile := filepath.Join(tmpDir, "config.yaml")
	
	err := os.WriteFile(configFile, []byte(configYAML), 0644)
	if err != nil {
		t.Fatalf("Failed to write test config file: %v", err)
	}

	cfg, err := Load(configFile)
	if err != nil {
		t.Fatalf("Failed to load config: %v", err)
	}
	
	if cfg.Provider.APIKey != "test-key-123" {
		t.Errorf("Expected API key 'test-key-123', got '%s'", cfg.Provider.APIKey)
	}
}

func TestValidateEdgeCases(t *testing.T) {
	tests := []struct {
		name   string
		config *models.Config
		valid  bool
	}{
		{
			name: "repeat count boundary - valid 1",
			config: &models.Config{
				Provider: models.ProviderConfig{Name: "vllm", BaseURL: "http://test"},
				Processing: models.ProcessingConfig{Repeat: 1},
				Output: models.OutputConfig{Format: "json"},
			},
			valid: true,
		},
		{
			name: "repeat count boundary - valid 10",
			config: &models.Config{
				Provider: models.ProviderConfig{Name: "vllm", BaseURL: "http://test"},
				Processing: models.ProcessingConfig{Repeat: 10},
				Output: models.OutputConfig{Format: "json"},
			},
			valid: true,
		},
		{
			name: "repeat count boundary - invalid 0",
			config: &models.Config{
				Provider: models.ProviderConfig{Name: "vllm", BaseURL: "http://test"},
				Processing: models.ProcessingConfig{Repeat: 0},
				Output: models.OutputConfig{Format: "json"},
			},
			valid: false,
		},
		{
			name: "repeat count boundary - invalid 11",
			config: &models.Config{
				Provider: models.ProviderConfig{Name: "vllm", BaseURL: "http://test"},
				Processing: models.ProcessingConfig{Repeat: 11},
				Output: models.OutputConfig{Format: "json"},
			},
			valid: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validate(tt.config)
			if tt.valid && err != nil {
				t.Errorf("Expected valid config but got error: %v", err)
			}
			if !tt.valid && err == nil {
				t.Errorf("Expected invalid config but got no error")
			}
		})
	}
}