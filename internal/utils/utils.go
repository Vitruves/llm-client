package utils

import (
	"fmt"
	"strings"
	"time"
)

func FormatDuration(d time.Duration) string {
	if d < time.Millisecond {
		return fmt.Sprintf("%.2fμs", float64(d.Nanoseconds())/1000)
	}
	if d < time.Second {
		return fmt.Sprintf("%.2fms", float64(d.Nanoseconds())/1000000)
	}
	return d.Round(time.Millisecond).String()
}

func TruncateString(s string, maxLength int) string {
	if len(s) <= maxLength {
		return s
	}
	if maxLength <= 3 {
		return "..."
	}
	return s[:maxLength-3] + "..."
}

func NormalizeLabel(label string) string {
	return strings.TrimSpace(strings.ToLower(label))
}

func CalculatePercentage(part, total int) float64 {
	if total == 0 {
		return 0
	}
	return float64(part) / float64(total) * 100
}

func ParseTemplate(template string, data map[string]interface{}) string {
	result := template
	for key, value := range data {
		placeholder := fmt.Sprintf("{%s}", key)
		result = strings.ReplaceAll(result, placeholder, fmt.Sprintf("%v", value))
	}
	return result
}

func Contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}

func RemoveDuplicates(slice []string) []string {
	seen := make(map[string]bool)
	var result []string

	for _, item := range slice {
		if !seen[item] {
			seen[item] = true
			result = append(result, item)
		}
	}

	return result
}

func MostFrequent(items []string) (string, int) {
	if len(items) == 0 {
		return "", 0
	}

	counts := make(map[string]int)
	for _, item := range items {
		counts[item]++
	}

	var maxItem string
	var maxCount int
	for item, count := range counts {
		if count > maxCount {
			maxCount = count
			maxItem = item
		}
	}

	return maxItem, maxCount
}
