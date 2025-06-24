package config

import (
	"fmt"
	"os"

	"llm-client/internal/models"

	"gopkg.in/yaml.v3"
)

func Load(filename string) (*models.Config, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	content := os.ExpandEnv(string(data))

	var cfg models.Config
	if err := yaml.Unmarshal([]byte(content), &cfg); err != nil {
		return nil, fmt.Errorf("failed to parse config: %w", err)
	}

	setDefaults(&cfg)
	if err := validate(&cfg); err != nil {
		return nil, fmt.Errorf("invalid config: %w", err)
	}

	return &cfg, nil
}

func setDefaults(cfg *models.Config) {
	if cfg.Provider.Timeout == "" {
		cfg.Provider.Timeout = "60s"
	}
	if cfg.Processing.Workers == 0 {
		cfg.Processing.Workers = 4
	}
	if cfg.Processing.BatchSize == 0 {
		cfg.Processing.BatchSize = 1
	}
	if cfg.Processing.Repeat == 0 {
		cfg.Processing.Repeat = 1
	}
	if cfg.Output.Directory == "" {
		cfg.Output.Directory = "./output"
	}
	if cfg.Output.Format == "" {
		cfg.Output.Format = "json"
	}
	// Don't set a hardcoded default - let the config file specify it
}

func validate(cfg *models.Config) error {
	if cfg.Provider.Name == "" {
		return fmt.Errorf("provider name is required")
	}

	validProviders := map[string]bool{
		"llamacpp": true,
		"vllm":     true,
		"openai":   true,
	}

	if !validProviders[cfg.Provider.Name] {
		return fmt.Errorf("unsupported provider: %s", cfg.Provider.Name)
	}

	if cfg.Provider.BaseURL == "" && cfg.Provider.Name != "openai" {
		return fmt.Errorf("base_url is required for provider: %s", cfg.Provider.Name)
	}

	if cfg.Provider.Name == "openai" && cfg.Provider.APIKey == "" {
		return fmt.Errorf("api_key is required for OpenAI provider")
	}

	validFormats := map[string]bool{"json": true, "csv": true, "parquet": true, "xlsx": true}
	if !validFormats[cfg.Output.Format] {
		return fmt.Errorf("unsupported output format: %s", cfg.Output.Format)
	}

	if cfg.Processing.Repeat < 1 || cfg.Processing.Repeat > 10 {
		return fmt.Errorf("repeat count must be between 1 and 10")
	}

	return nil
}
