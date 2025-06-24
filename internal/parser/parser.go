package parser

import (
	"github.com/Vitruves/llm-client/internal/models"
	"regexp"
	"strings"
)

type Parser struct {
	config models.ParsingConfig
}

func New(config models.ParsingConfig) *Parser {
	return &Parser{config: config}
}

func (p *Parser) Parse(response string) string {
	finalAnswer, _ := p.ParseWithThinking(response)
	return finalAnswer
}

func (p *Parser) ParseWithThinking(response string) (string, string) {
	thinkingContent := p.extractThinkingContent(response)
	processedResponse := p.processResponse(response)
	finalAnswer := p.extractFinalAnswer(processedResponse)

	return finalAnswer, thinkingContent
}

func (p *Parser) extractThinkingContent(response string) string {
	if p.config.ThinkingTags == "" {
		return ""
	}

	startTag, endTag := p.parseThinkingTags(p.config.ThinkingTags)
	if startTag == "" || endTag == "" {
		return ""
	}

	startIdx := strings.Index(response, startTag)
	if startIdx == -1 {
		return ""
	}

	startIdx += len(startTag)
	endIdx := strings.Index(response[startIdx:], endTag)
	if endIdx == -1 {
		return ""
	}

	return strings.TrimSpace(response[startIdx : startIdx+endIdx])
}

func (p *Parser) processResponse(response string) string {
	if p.config.PreserveThinking != nil && *p.config.PreserveThinking {
		return response
	}

	if p.config.ThinkingTags == "" {
		return response
	}

	startTag, endTag := p.parseThinkingTags(p.config.ThinkingTags)
	if startTag == "" || endTag == "" {
		return response
	}

	pattern := regexp.QuoteMeta(startTag) + ".*?" + regexp.QuoteMeta(endTag)
	re, err := regexp.Compile(pattern)
	if err != nil {
		return response
	}

	return re.ReplaceAllString(response, "")
}

func (p *Parser) extractFinalAnswer(response string) string {
	response = strings.TrimSpace(response)

	var answer string

	if len(p.config.AnswerPatterns) > 0 {
		if result := p.parseWithPatterns(response); result != "" {
			answer = result
		}
	}

	if answer == "" && len(p.config.Find) > 0 {
		if result := p.parseWithFind(response); result != "" {
			answer = result
		}
	}

	// Apply mapping if answer was found
	if answer != "" {
		if mapped, exists := p.config.Map[answer]; exists {
			return mapped
		}
		return answer
	}

	return p.config.Default
}

func (p *Parser) parseWithPatterns(response string) string {
	for _, pattern := range p.config.AnswerPatterns {
		flags := ""
		if p.config.CaseSensitive == nil || !*p.config.CaseSensitive {
			flags = "(?i)"
		}

		fullPattern := flags + pattern
		re, err := regexp.Compile(fullPattern)
		if err != nil {
			continue
		}

		matches := re.FindStringSubmatch(response)
		if len(matches) > 1 {
			return strings.TrimSpace(matches[1])
		}
	}
	return ""
}

func (p *Parser) parseWithFind(response string) string {
	responseToCheck := response
	if p.config.CaseSensitive == nil || !*p.config.CaseSensitive {
		responseToCheck = strings.ToLower(response)
	}

	for _, target := range p.config.Find {
		// Handle wildcard - matches any non-empty response
		if target == "*" {
			if strings.TrimSpace(response) != "" {
				return strings.TrimSpace(response)
			}
			continue
		}

		targetToCheck := target
		if p.config.CaseSensitive == nil || !*p.config.CaseSensitive {
			targetToCheck = strings.ToLower(target)
		}

		if p.config.ExactMatch != nil && *p.config.ExactMatch {
			if responseToCheck == targetToCheck {
				return target
			}
		} else {
			if strings.Contains(responseToCheck, targetToCheck) {
				return target
			}
		}
	}

	return ""
}

func (p *Parser) parseThinkingTags(tags string) (string, string) {
	// Handle empty tags
	if tags == "" {
		return "", ""
	}

	// Handle combined format like "<think></think>" or "[REASON][/REASON]"
	if strings.Contains(tags, "><") {
		parts := strings.SplitN(tags, "><", 2)
		if len(parts) == 2 {
			return parts[0] + ">", "<" + parts[1]
		}
	} else if strings.Contains(tags, "][") {
		parts := strings.SplitN(tags, "][", 2)
		if len(parts) == 2 {
			return parts[0] + "]", "[" + parts[1]
		}
	}

	// Handle space-separated format like "<think> </think>"
	if strings.Contains(tags, " ") {
		parts := strings.Fields(tags)
		if len(parts) == 2 {
			return parts[0], parts[1]
		}
	}

	// If no separator found, assume it's a single tag (shouldn't happen in normal usage)
	return "", ""
}
