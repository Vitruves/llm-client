package parser

import (
	"testing"

	"github.com/Vitruves/llm-client/internal/models"
)

func TestParse(t *testing.T) {
	tests := []struct {
		name     string
		config   models.ParsingConfig
		response string
		expected string
	}{
		{
			name: "simple find match",
			config: models.ParsingConfig{
				Find:     []string{"positive", "negative", "neutral"},
				Default:  "unknown",
				Fallback: "error",
			},
			response: "The sentiment is positive based on the analysis.",
			expected: "positive",
		},
		{
			name: "case insensitive find match",
			config: models.ParsingConfig{
				Find:          []string{"positive", "negative", "neutral"},
				Default:       "unknown",
				Fallback:      "error",
				CaseSensitive: boolPtr(false),
			},
			response: "The sentiment is POSITIVE based on the analysis.",
			expected: "positive",
		},
		{
			name: "case sensitive find no match",
			config: models.ParsingConfig{
				Find:          []string{"positive", "negative", "neutral"},
				Default:       "unknown",
				Fallback:      "error",
				CaseSensitive: boolPtr(true),
			},
			response: "The sentiment is POSITIVE based on the analysis.",
			expected: "unknown",
		},
		{
			name: "exact match required - success",
			config: models.ParsingConfig{
				Find:       []string{"positive", "negative", "neutral"},
				Default:    "unknown",
				Fallback:   "error",
				ExactMatch: boolPtr(true),
			},
			response: "positive",
			expected: "positive",
		},
		{
			name: "exact match required - failure",
			config: models.ParsingConfig{
				Find:       []string{"positive", "negative", "neutral"},
				Default:    "unknown",
				Fallback:   "error",
				ExactMatch: boolPtr(true),
			},
			response: "The sentiment is positive",
			expected: "unknown",
		},
		{
			name: "regex pattern match",
			config: models.ParsingConfig{
				AnswerPatterns: []string{`Classification: (\w+)`},
				Find:           []string{},
				Default:        "unknown",
				Fallback:       "error",
			},
			response: "After analysis, Classification: positive",
			expected: "positive",
		},
		{
			name: "multiple regex patterns - first match wins",
			config: models.ParsingConfig{
				AnswerPatterns: []string{
					`Result: (\w+)`,
					`Classification: (\w+)`,
				},
				Find:     []string{},
				Default:  "unknown",
				Fallback: "error",
			},
			response: "Classification: negative, Result: positive",
			expected: "positive", // First pattern matches "Result: positive"
		},
		{
			name: "regex pattern with find fallback",
			config: models.ParsingConfig{
				AnswerPatterns: []string{`Result: (\w+)`},
				Find:           []string{"positive", "negative", "neutral"},
				Default:        "unknown",
				Fallback:       "error",
			},
			response: "The sentiment is positive without pattern.",
			expected: "positive",
		},
		{
			name: "answer mapping",
			config: models.ParsingConfig{
				Find:     []string{"pos", "neg", "neu"},
				Default:  "unknown",
				Fallback: "error",
				Map: map[string]string{
					"pos": "positive",
					"neg": "negative",
					"neu": "neutral",
				},
			},
			response: "The result is pos",
			expected: "positive",
		},
		{
			name: "regex with mapping",
			config: models.ParsingConfig{
				AnswerPatterns: []string{`Answer: (\w+)`},
				Default:        "unknown",
				Fallback:       "error",
				Map: map[string]string{
					"good": "positive",
					"bad":  "negative",
				},
			},
			response: "Answer: good",
			expected: "positive",
		},
		{
			name: "wildcard match",
			config: models.ParsingConfig{
				Find:     []string{"*"},
				Default:  "unknown",
				Fallback: "error",
			},
			response: "any non-empty response",
			expected: "any non-empty response",
		},
		{
			name: "no match returns default",
			config: models.ParsingConfig{
				Find:     []string{"positive", "negative"},
				Default:  "unknown",
				Fallback: "error",
			},
			response: "This contains neither target word.",
			expected: "unknown",
		},
		{
			name: "empty response returns default",
			config: models.ParsingConfig{
				Find:     []string{"positive", "negative"},
				Default:  "unknown",
				Fallback: "error",
			},
			response: "",
			expected: "unknown",
		},
		{
			name: "complex multiline response with pattern",
			config: models.ParsingConfig{
				AnswerPatterns: []string{`(?s)Final Answer: (.*?)(?:\n|$)`},
				Default:        "unknown",
				Fallback:       "error",
			},
			response: `Let me think about this step by step.

First, I need to analyze the sentiment.
This appears to be a positive statement.

Final Answer: positive

That's my conclusion.`,
			expected: "positive",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parser := New(tt.config)
			result := parser.Parse(tt.response)

			if result != tt.expected {
				t.Errorf("Expected '%s', got '%s'", tt.expected, result)
			}
		})
	}
}

func TestParseWithThinking(t *testing.T) {
	tests := []struct {
		name             string
		config           models.ParsingConfig
		response         string
		expectedAnswer   string
		expectedThinking string
	}{
		{
			name: "extract thinking content",
			config: models.ParsingConfig{
				Find:             []string{"positive", "negative", "neutral"},
				Default:          "unknown",
				Fallback:         "error",
				ThinkingTags:     "<thinking></thinking>",
				PreserveThinking: boolPtr(true),
			},
			response: `<thinking>
This is a positive statement because it expresses happiness.
Let me analyze the words used.
</thinking>

The sentiment is positive.`,
			expectedAnswer:   "positive",
			expectedThinking: "This is a positive statement because it expresses happiness.\nLet me analyze the words used.",
		},
		{
			name: "thinking removed from response when preserve is false",
			config: models.ParsingConfig{
				Find:             []string{"positive", "negative", "neutral"},
				Default:          "unknown",
				Fallback:         "error",
				ThinkingTags:     "<thinking></thinking>",
				PreserveThinking: boolPtr(false),
			},
			response: `<thinking>
This should be removed from the response.
</thinking>

The sentiment is positive.`,
			expectedAnswer:   "positive",
			expectedThinking: "This should be removed from the response.",
		},
		{
			name: "multiple thinking blocks",
			config: models.ParsingConfig{
				Find:         []string{"positive", "negative", "neutral"},
				Default:      "unknown",
				Fallback:     "error",
				ThinkingTags: "<think></think>",
			},
			response: `<think>First thought process</think>
Analysis shows positive sentiment.
<think>Second thought process</think>`,
			expectedAnswer:   "positive",
			expectedThinking: "First thought process",
		},
		{
			name: "custom thinking tags",
			config: models.ParsingConfig{
				Find:         []string{"positive", "negative", "neutral"},
				Default:      "unknown",
				Fallback:     "error",
				ThinkingTags: "[REASONING][/REASONING]",
			},
			response: `[REASONING]
Let me analyze this carefully.
The tone seems optimistic.
[/REASONING]

Classification: positive`,
			expectedAnswer:   "positive",
			expectedThinking: "Let me analyze this carefully.\nThe tone seems optimistic.",
		},
		{
			name: "no thinking tags found",
			config: models.ParsingConfig{
				Find:         []string{"positive", "negative", "neutral"},
				Default:      "unknown",
				Fallback:     "error",
				ThinkingTags: "<thinking></thinking>",
			},
			response:         "Simple response with positive sentiment.",
			expectedAnswer:   "positive",
			expectedThinking: "",
		},
		{
			name: "malformed thinking tags",
			config: models.ParsingConfig{
				Find:         []string{"positive", "negative", "neutral"},
				Default:      "unknown",
				Fallback:     "error",
				ThinkingTags: "<thinking></thinking>",
			},
			response:         "<thinking>Unclosed thinking tag with positive sentiment.",
			expectedAnswer:   "positive",
			expectedThinking: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parser := New(tt.config)
			answer, thinking := parser.ParseWithThinking(tt.response)

			if answer != tt.expectedAnswer {
				t.Errorf("Expected answer '%s', got '%s'", tt.expectedAnswer, answer)
			}

			if thinking != tt.expectedThinking {
				t.Errorf("Expected thinking '%s', got '%s'", tt.expectedThinking, thinking)
			}
		})
	}
}

func TestRegexPatternValidation(t *testing.T) {
	tests := []struct {
		name    string
		pattern string
		valid   bool
	}{
		{
			name:    "valid pattern with capture group",
			pattern: `Answer: (\w+)`,
			valid:   true,
		},
		{
			name:    "invalid pattern - no capture group",
			pattern: `Answer: \w+`,
			valid:   false, // Should be skipped, not crash
		},
		{
			name:    "invalid regex syntax",
			pattern: `Answer: [`,
			valid:   false, // Should be skipped, not crash
		},
		{
			name:    "multiple capture groups - only first used",
			pattern: `Answer: (\w+) with confidence (\d+)`,
			valid:   true, // Should use first capture group
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := models.ParsingConfig{
				AnswerPatterns: []string{tt.pattern},
				Find:           []string{"fallback"},
				Default:        "unknown",
				Fallback:       "error",
			}

			parser := New(config)

			// Should not panic regardless of pattern validity
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("Parser panicked with pattern '%s': %v", tt.pattern, r)
				}
			}()

			testResponse := "Answer: positive with confidence 95"
			result := parser.Parse(testResponse)

			if tt.valid {
				// Valid patterns should extract the answer
				if result != "positive" && result != "fallback" {
					t.Errorf("Expected 'positive' or 'fallback', got '%s'", result)
				}
			} else {
				// Invalid patterns should fall back to find array
				if result != "fallback" && result != "unknown" {
					t.Errorf("Expected 'fallback' or 'unknown' for invalid pattern, got '%s'", result)
				}
			}
		})
	}
}

func TestEdgeCases(t *testing.T) {
	tests := []struct {
		name     string
		config   models.ParsingConfig
		response string
		expected string
	}{
		{
			name: "unicode characters",
			config: models.ParsingConfig{
				Find:     []string{"положительный", "отрицательный"},
				Default:  "неизвестно",
				Fallback: "ошибка",
			},
			response: "Результат: положительный",
			expected: "положительный",
		},
		{
			name: "very long response",
			config: models.ParsingConfig{
				Find:     []string{"positive"},
				Default:  "unknown",
				Fallback: "error",
			},
			response: "This is a very long response that goes on and on and contains the word positive somewhere in the middle of a lot of text that might slow down processing but should still work correctly.",
			expected: "positive",
		},
		{
			name: "special regex characters in find",
			config: models.ParsingConfig{
				Find:     []string{"[positive]", "(negative)", "neutral*"},
				Default:  "unknown",
				Fallback: "error",
			},
			response: "The result is [positive] classification.",
			expected: "[positive]",
		},
		{
			name: "numeric values",
			config: models.ParsingConfig{
				Find:     []string{"1", "2", "3", "4", "5"},
				Default:  "0",
				Fallback: "error",
			},
			response: "Rating: 3 out of 5",
			expected: "3",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parser := New(tt.config)
			result := parser.Parse(tt.response)

			if result != tt.expected {
				t.Errorf("Expected '%s', got '%s'", tt.expected, result)
			}
		})
	}
}

func TestParseThinkingTags(t *testing.T) {
	tests := []struct {
		name      string
		input     string
		startTag  string
		endTag    string
		expectErr bool
	}{
		{
			name:      "standard thinking tags",
			input:     "<thinking></thinking>",
			startTag:  "<thinking>",
			endTag:    "</thinking>",
			expectErr: false,
		},
		{
			name:      "custom tags",
			input:     "[REASON][/REASON]",
			startTag:  "[REASON]",
			endTag:    "[/REASON]",
			expectErr: false,
		},
		{
			name:      "malformed input - no separator",
			input:     "<thinking>",
			startTag:  "",
			endTag:    "",
			expectErr: true,
		},
		{
			name:      "empty input",
			input:     "",
			startTag:  "",
			endTag:    "",
			expectErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parser := &Parser{}
			startTag, endTag := parser.parseThinkingTags(tt.input)

			hasErr := (startTag == "" && endTag == "")
			if hasErr != tt.expectErr {
				t.Errorf("Expected error: %v, got error: %v", tt.expectErr, hasErr)
			}

			if !tt.expectErr {
				if startTag != tt.startTag {
					t.Errorf("Expected start tag '%s', got '%s'", tt.startTag, startTag)
				}
				if endTag != tt.endTag {
					t.Errorf("Expected end tag '%s', got '%s'", tt.endTag, endTag)
				}
			}
		})
	}
}

// Helper function to create bool pointers
func boolPtr(b bool) *bool {
	return &b
}
