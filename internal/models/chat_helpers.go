package models

import (
	"encoding/json"
	"fmt"
	"strings"
)

// ConversationBuilder helps build chat conversations
type ConversationBuilder struct {
	messages []Message
}

// NewConversation creates a new conversation builder
func NewConversation() *ConversationBuilder {
	return &ConversationBuilder{
		messages: make([]Message, 0),
	}
}

// AddSystemMessage adds a system message to the conversation
func (cb *ConversationBuilder) AddSystemMessage(content string) *ConversationBuilder {
	cb.messages = append(cb.messages, NewTextMessage("system", content))
	return cb
}

// AddUserMessage adds a user message to the conversation
func (cb *ConversationBuilder) AddUserMessage(content string) *ConversationBuilder {
	cb.messages = append(cb.messages, NewTextMessage("user", content))
	return cb
}

// AddAssistantMessage adds an assistant message to the conversation
func (cb *ConversationBuilder) AddAssistantMessage(content string) *ConversationBuilder {
	cb.messages = append(cb.messages, NewTextMessage("assistant", content))
	return cb
}

// AddUserMessageWithImages adds a user message with images (multimodal)
func (cb *ConversationBuilder) AddUserMessageWithImages(text string, imageUrls []string) *ConversationBuilder {
	contents := make([]interface{}, 0, len(imageUrls)+1)

	// Add text content
	if text != "" {
		contents = append(contents, map[string]interface{}{
			"type": "text",
			"text": text,
		})
	}

	// Add image contents
	for _, url := range imageUrls {
		contents = append(contents, map[string]interface{}{
			"type": "image_url",
			"image_url": map[string]interface{}{
				"url": url,
			},
		})
	}

	cb.messages = append(cb.messages, NewMultiModalMessage("user", contents))
	return cb
}

// AddToolMessage adds a tool response message
func (cb *ConversationBuilder) AddToolMessage(toolCallId, content string) *ConversationBuilder {
	cb.messages = append(cb.messages, NewToolMessage(toolCallId, content))
	return cb
}

// AddAssistantMessageWithToolCalls adds an assistant message with tool calls
func (cb *ConversationBuilder) AddAssistantMessageWithToolCalls(content string, toolCalls []ToolCall) *ConversationBuilder {
	msg := NewTextMessage("assistant", content)
	msg.ToolCalls = toolCalls
	cb.messages = append(cb.messages, msg)
	return cb
}

// Build returns the built conversation
func (cb *ConversationBuilder) Build() []Message {
	return cb.messages
}

// Clear clears the conversation
func (cb *ConversationBuilder) Clear() *ConversationBuilder {
	cb.messages = cb.messages[:0]
	return cb
}

// GetLastMessage returns the last message in the conversation
func (cb *ConversationBuilder) GetLastMessage() *Message {
	if len(cb.messages) == 0 {
		return nil
	}
	return &cb.messages[len(cb.messages)-1]
}

// GetMessageCount returns the number of messages in the conversation
func (cb *ConversationBuilder) GetMessageCount() int {
	return len(cb.messages)
}

// ChatTemplate represents a chat template for formatting conversations
type ChatTemplate struct {
	Name            string
	SystemPrefix    string
	SystemSuffix    string
	UserPrefix      string
	UserSuffix      string
	AssistantPrefix string
	AssistantSuffix string
	BosToken        string
	EosToken        string
	StopTokens      []string
}

// Common chat templates
var (
	ChatMLTemplate = ChatTemplate{
		Name:            "chatml",
		SystemPrefix:    "<|im_start|>system\n",
		SystemSuffix:    "<|im_end|>\n",
		UserPrefix:      "<|im_start|>user\n",
		UserSuffix:      "<|im_end|>\n",
		AssistantPrefix: "<|im_start|>assistant\n",
		AssistantSuffix: "<|im_end|>\n",
		BosToken:        "",
		EosToken:        "<|im_end|>",
		StopTokens:      []string{"<|im_end|>", "<|endoftext|>"},
	}

	Llama2Template = ChatTemplate{
		Name:            "llama2",
		SystemPrefix:    "<s>[INST] <<SYS>>\n",
		SystemSuffix:    "\n<</SYS>>\n\n",
		UserPrefix:      "",
		UserSuffix:      " [/INST] ",
		AssistantPrefix: "",
		AssistantSuffix: " </s><s>[INST] ",
		BosToken:        "<s>",
		EosToken:        "</s>",
		StopTokens:      []string{"</s>"},
	}

	Llama3Template = ChatTemplate{
		Name:            "llama3",
		SystemPrefix:    "<|start_header_id|>system<|end_header_id|>\n\n",
		SystemSuffix:    "<|eot_id|>",
		UserPrefix:      "<|start_header_id|>user<|end_header_id|>\n\n",
		UserSuffix:      "<|eot_id|>",
		AssistantPrefix: "<|start_header_id|>assistant<|end_header_id|>\n\n",
		AssistantSuffix: "<|eot_id|>",
		BosToken:        "<|begin_of_text|>",
		EosToken:        "<|eot_id|>",
		StopTokens:      []string{"<|eot_id|>", "<|end_of_text|>"},
	}

	VicunaTemplate = ChatTemplate{
		Name:            "vicuna",
		SystemPrefix:    "",
		SystemSuffix:    "\n\n",
		UserPrefix:      "USER: ",
		UserSuffix:      "\n",
		AssistantPrefix: "ASSISTANT: ",
		AssistantSuffix: "\n",
		BosToken:        "",
		EosToken:        "</s>",
		StopTokens:      []string{"</s>"},
	}
)

// GetChatTemplate returns a chat template by name
func GetChatTemplate(name string) *ChatTemplate {
	switch strings.ToLower(name) {
	case "chatml":
		return &ChatMLTemplate
	case "llama2":
		return &Llama2Template
	case "llama3":
		return &Llama3Template
	case "vicuna":
		return &VicunaTemplate
	default:
		return &ChatMLTemplate // Default fallback
	}
}

// FormatConversation formats a conversation using the specified template
func (ct *ChatTemplate) FormatConversation(messages []Message) string {
	var result strings.Builder

	if ct.BosToken != "" {
		result.WriteString(ct.BosToken)
	}

	for i, msg := range messages {
		content := msg.GetTextContent()

		switch msg.Role {
		case "system":
			result.WriteString(ct.SystemPrefix)
			result.WriteString(content)
			result.WriteString(ct.SystemSuffix)
		case "user":
			result.WriteString(ct.UserPrefix)
			result.WriteString(content)
			result.WriteString(ct.UserSuffix)
		case "assistant":
			result.WriteString(ct.AssistantPrefix)
			result.WriteString(content)
			if i < len(messages)-1 { // Don't add suffix for last assistant message
				result.WriteString(ct.AssistantSuffix)
			}
		}
	}

	return result.String()
}

// ValidateMessage validates a message for common issues
func ValidateMessage(msg Message) error {
	if msg.Role == "" {
		return fmt.Errorf("message role cannot be empty")
	}

	validRoles := []string{"system", "user", "assistant", "tool", "function"}
	isValidRole := false
	for _, role := range validRoles {
		if msg.Role == role {
			isValidRole = true
			break
		}
	}
	if !isValidRole {
		return fmt.Errorf("invalid message role: %s", msg.Role)
	}

	if msg.Content == nil || (msg.IsTextOnly() && msg.GetTextContent() == "") {
		if len(msg.ToolCalls) == 0 {
			return fmt.Errorf("message content cannot be empty unless it has tool calls")
		}
	}

	if msg.Role == "tool" && msg.ToolCallId == nil {
		return fmt.Errorf("tool messages must have a tool_call_id")
	}

	return nil
}

// ValidateConversation validates an entire conversation
func ValidateConversation(messages []Message) error {
	if len(messages) == 0 {
		return fmt.Errorf("conversation cannot be empty")
	}

	for i, msg := range messages {
		if err := ValidateMessage(msg); err != nil {
			return fmt.Errorf("message %d: %w", i, err)
		}
	}

	return nil
}

// EstimateTokenCount provides a rough estimate of token count for a conversation
func EstimateTokenCount(messages []Message) int {
	totalTokens := 0

	for _, msg := range messages {
		content := msg.GetTextContent()
		// Rough estimation: ~4 characters per token
		contentTokens := len(content) / 4
		// Add overhead for role and formatting
		totalTokens += contentTokens + 10

		// Add tokens for tool calls
		if len(msg.ToolCalls) > 0 {
			for _, toolCall := range msg.ToolCalls {
				totalTokens += len(toolCall.Function.Name)/4 + len(toolCall.Function.Arguments)/4 + 20
			}
		}
	}

	return totalTokens
}

// ConversationToJSON converts a conversation to JSON string
func ConversationToJSON(messages []Message) (string, error) {
	data, err := json.MarshalIndent(messages, "", "  ")
	if err != nil {
		return "", fmt.Errorf("failed to marshal conversation: %w", err)
	}
	return string(data), nil
}

// ConversationFromJSON creates a conversation from JSON string
func ConversationFromJSON(jsonStr string) ([]Message, error) {
	var messages []Message
	if err := json.Unmarshal([]byte(jsonStr), &messages); err != nil {
		return nil, fmt.Errorf("failed to unmarshal conversation: %w", err)
	}
	return messages, nil
}

// TruncateConversation truncates a conversation to fit within token limits
func TruncateConversation(messages []Message, maxTokens int) []Message {
	if len(messages) == 0 {
		return messages
	}

	// Always keep the system message if present
	var systemMsg *Message
	startIdx := 0
	if messages[0].Role == "system" {
		systemMsg = &messages[0]
		startIdx = 1
	}

	// Estimate tokens and truncate from the beginning (keeping recent messages)
	currentTokens := 0
	if systemMsg != nil {
		currentTokens = EstimateTokenCount([]Message{*systemMsg})
	}

	var result []Message
	if systemMsg != nil {
		result = append(result, *systemMsg)
	}

	// Add messages from the end, working backwards
	for i := len(messages) - 1; i >= startIdx; i-- {
		msgTokens := EstimateTokenCount([]Message{messages[i]})
		if currentTokens+msgTokens > maxTokens {
			break
		}
		currentTokens += msgTokens
		result = append([]Message{messages[i]}, result...)
	}

	return result
}
