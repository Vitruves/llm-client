package logger

import (
	"fmt"
	"os"
	"runtime"
	"time"

	"github.com/fatih/color"
)

// Color constants for log output
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

type LogLevel int

const (
	DEBUG LogLevel = iota
	INFO
	WARNING
	ERROR
	PROGRESS
)

var (
	currentLevel = INFO
	verbose      = false
	// Enhanced color functions using fatih/color
	colorError   = color.New(color.FgRed, color.Bold)
	colorSuccess = color.New(color.FgGreen, color.Bold)
	colorWarning = color.New(color.FgYellow, color.Bold)
	colorInfo    = color.New(color.FgCyan, color.Bold)
	colorDebug   = color.New(color.FgHiBlack)
	colorHeader  = color.New(color.FgMagenta, color.Bold, color.Underline)
	colorValue   = color.New(color.FgHiGreen)
	colorKey     = color.New(color.FgBlue)
)

// SetLevel sets the global log level
func SetLevel(level LogLevel) {
	currentLevel = level
}

// SetVerbose enables verbose logging (DEBUG level)
func SetVerbose(enabled bool) {
	verbose = enabled
	if enabled {
		currentLevel = DEBUG
	}
}

// IsVerbose returns whether verbose logging is enabled
func IsVerbose() bool {
	return verbose
}

func (l LogLevel) String() string {
	switch l {
	case DEBUG:
		return "DEBUG"
	case INFO:
		return "INFO"
	case WARNING:
		return "WARNING"
	case ERROR:
		return "ERROR"
	case PROGRESS:
		return "PROGRESS"
	default:
		return "UNKNOWN"
	}
}

func (l LogLevel) Color() string {
	switch l {
	case DEBUG:
		return ColorGray
	case INFO:
		return ColorBlue
	case WARNING:
		return ColorYellow
	case ERROR:
		return ColorRed
	case PROGRESS:
		return ColorCyan // Cyan for progress bar
	default:
		return ColorReset
	}
}

// GetColorFunc returns the enhanced color function for the log level
func (l LogLevel) GetColorFunc() *color.Color {
	switch l {
	case DEBUG:
		return colorDebug
	case INFO:
		return colorInfo
	case WARNING:
		return colorWarning
	case ERROR:
		return colorError
	case PROGRESS:
		return colorInfo
	default:
		return color.New(color.Reset)
	}
}

// formatMessage creates a formatted log message with timestamp and level
func formatMessage(level LogLevel, message string, args ...interface{}) string {
	now := time.Now()
	timestamp := colorKey.Sprintf("%s", now.Format("15:04")) // HH:MM format with color

	levelColorFunc := level.GetColorFunc()
	levelStr := levelColorFunc.Sprintf("%s", level.String())

	var formattedMsg string
	if level == PROGRESS {
		formattedMsg = message // Message is already pre-formatted for progress bar
	} else {
		formattedMsg = fmt.Sprintf(message, args...)
	}

	return fmt.Sprintf("%s - %s : %s", timestamp, levelStr, formattedMsg)
}

// shouldLog determines if a message at the given level should be logged
func shouldLog(level LogLevel) bool {
	// Progress messages should always be visible regardless of verbose setting
	if level == PROGRESS {
		return true
	}
	return level >= currentLevel
}

// Debug logs a debug message (only visible with --verbose)
func Debug(message string, args ...interface{}) {
	if shouldLog(DEBUG) {
		fmt.Println(formatMessage(DEBUG, message, args...))
	}
}

// Info logs an info message
func Info(message string, args ...interface{}) {
	if shouldLog(INFO) {
		fmt.Println(formatMessage(INFO, message, args...))
	}
}

// Warning logs a warning message
func Warning(message string, args ...interface{}) {
	if shouldLog(WARNING) {
		fmt.Println(formatMessage(WARNING, message, args...))
	}
}

// Error logs an error message
func Error(message string, args ...interface{}) {
	if shouldLog(ERROR) {
		fmt.Fprintf(os.Stderr, formatMessage(ERROR, message, args...)+"\n")
	}
}

// DebugRequest logs detailed request information (only in verbose mode)
func DebugRequest(provider, url string, params map[string]interface{}) {
	if !shouldLog(DEBUG) {
		return
	}

	Debug("Sending request to %s provider", provider)
	Debug("Request URL: %s", url)

	if len(params) > 0 {
		Debug("Request parameters:")
		for key, value := range params {
			Debug("  %s: %v", key, value)
		}
	}
}

// DebugResponse logs detailed response information (only in verbose mode)
func DebugResponse(statusCode int, response string, duration time.Duration) {
	if !shouldLog(DEBUG) {
		return
	}

	Debug("Response received (HTTP %d) in %v", statusCode, duration)

	// Truncate very long responses for readability
	if len(response) > 500 {
		Debug("Response body (truncated): %s...", response[:500])
	} else {
		Debug("Response body: %s", response)
	}
}

// DebugSystem logs system information (only in verbose mode)
func DebugSystem() {
	if !shouldLog(DEBUG) {
		return
	}

	Debug("System information:")
	Debug("  OS: %s", runtime.GOOS)
	Debug("  Architecture: %s", runtime.GOARCH)
	Debug("  Go version: %s", runtime.Version())
	Debug("  CPU count: %d", runtime.NumCPU())
}

// DebugConfig logs configuration details (only in verbose mode)
func DebugConfig(config interface{}) {
	if !shouldLog(DEBUG) {
		return
	}

	Debug("Configuration loaded:")
	Debug("  %+v", config)
}

// Progress creates a progress-compatible message that won't interfere with progress bars
func Progress(message string, args ...interface{}) {
	// Use formatMessage directly for progress output to ensure it matches the log format
	// Use fmt.Print to keep it on the same line and make it "sticky"
	fmt.Print("\r\033[K" + formatMessage(PROGRESS, message, args...))
}

// Fatal logs an error and exits the program
func Fatal(message string, args ...interface{}) {
	Error(message, args...)
	os.Exit(1)
}

// Success logs a success message with green color
func Success(message string, args ...interface{}) {
	if shouldLog(INFO) {
		fmt.Println(formatMessage(INFO, message, args...))
	}
}

// Header logs a header message for sections
func Header(message string, args ...interface{}) {
	if shouldLog(INFO) {
		formattedMsg := fmt.Sprintf(message, args...)
		fmt.Println(formatMessage(INFO, formattedMsg))
	}
}
