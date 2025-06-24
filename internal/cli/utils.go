package cli

import (
	"llm-client/internal/logger"
	"os"
	"runtime"
)

// These color constants are now in the logger package
/*
const (
	ColorReset  = "\033[0m"
	ColorRed    = "\033[31m"
	ColorGreen  = "\033[32m"
	ColorYellow = "\033[33m"
	ColorBlue   = "\033[34m"
	ColorPurple = "\033[35m"
	ColorCyan   = "\033[36m"
	ColorGray   = "\033[90m"
	ColorBold   = "\033[1m"
	ColorWhite  = "\033[37m"
)
*/

// var colorEnabled = true // No longer needed as colors are managed by logger

func init() {
	// Disable colors on Windows unless explicitly enabled
	if runtime.GOOS == "windows" {
		if os.Getenv("FORCE_COLOR") == "" {
			// colorEnabled = false // No longer needed
		}
	}

	// Disable colors if NO_COLOR environment variable is set
	if os.Getenv("NO_COLOR") != "" {
		// colorEnabled = false // No longer needed
	}
}

// SetColorEnabled allows manual control of color output
// This function is no longer needed as color control is handled by the logger package.
func SetColorEnabled(enabled bool) {
	// No-op, colors are now controlled by logger package
}

// Helper functions for consistent CLI output
// These are deprecated and should be replaced by direct logger calls
func Error(text string) string {
	return text
}

func Success(text string) string {
	return text
}

func Warning(text string) string {
	return text
}

func Info(text string) string {
	return text
}

func Highlight(text string) string {
	return text
}

func Header(text string) string {
	return text
}

func Label(text string) string {
	return text
}

func Value(text string) string {
	return text
}

// PrintError prints an error message
func PrintError(format string, args ...interface{}) {
	logger.Error(format, args...)
}

// PrintWarning prints a warning message
func PrintWarning(format string, args ...interface{}) {
	logger.Warning(format, args...)
}

// PrintSuccess prints a success message
func PrintSuccess(format string, args ...interface{}) {
	logger.Success(format, args...)
}

// PrintInfo prints an info message
func PrintInfo(format string, args ...interface{}) {
	logger.Info(format, args...)
}
