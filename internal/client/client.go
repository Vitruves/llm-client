package client

import (
	"fmt"
	"llm-client/internal/models"
)

func NewClient(cfg *models.Config) (models.Client, error) {
	switch cfg.Provider.Name {
	case "llamacpp":
		return NewLlamaCppClient(cfg)
	case "vllm":
		return NewVLLMClient(cfg)
	case "openai":
		return NewOpenAIClient(cfg)
	default:
		return nil, fmt.Errorf("unsupported provider: %s", cfg.Provider.Name)
	}
}
