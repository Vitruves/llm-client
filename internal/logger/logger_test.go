package logger

import (
	"bytes"
	"os"
	"strings"
	"testing"
	"time"
)

func TestLogLevels(t *testing.T) {
	tests := []struct {
		level    LogLevel
		expected string
	}{
		{DEBUG, "DEBUG"},
		{INFO, "INFO"},
		{WARNING, "WARNING"},
		{ERROR, "ERROR"},
	}

	for _, tt := range tests {
		result := tt.level.String()
		if result != tt.expected {
			t.Errorf("Expected level string '%s', got '%s'", tt.expected, result)
		}
	}
}

func TestSetLevel(t *testing.T) {
	originalLevel := currentLevel
	defer func() { currentLevel = originalLevel }()

	SetLevel(WARNING)
	if currentLevel != WARNING {
		t.Errorf("Expected current level WARNING, got %v", currentLevel)
	}

	SetLevel(DEBUG)
	if currentLevel != DEBUG {
		t.Errorf("Expected current level DEBUG, got %v", currentLevel)
	}
}

func TestSetVerbose(t *testing.T) {
	originalLevel := currentLevel
	originalVerbose := verbose
	defer func() {
		currentLevel = originalLevel
		verbose = originalVerbose
	}()

	SetVerbose(true)
	if !verbose {
		t.Error("Expected verbose to be true")
	}
	if currentLevel != DEBUG {
		t.Errorf("Expected current level DEBUG when verbose, got %v", currentLevel)
	}

	SetVerbose(false)
	if verbose {
		t.Error("Expected verbose to be false")
	}
}

func TestIsVerbose(t *testing.T) {
	originalVerbose := verbose
	defer func() { verbose = originalVerbose }()

	verbose = true
	if !IsVerbose() {
		t.Error("Expected IsVerbose() to return true")
	}

	verbose = false
	if IsVerbose() {
		t.Error("Expected IsVerbose() to return false")
	}
}

func TestShouldLog(t *testing.T) {
	originalLevel := currentLevel
	defer func() { currentLevel = originalLevel }()

	SetLevel(WARNING)

	if shouldLog(DEBUG) {
		t.Error("DEBUG should not log when level is WARNING")
	}

	if shouldLog(INFO) {
		t.Error("INFO should not log when level is WARNING")
	}

	if !shouldLog(WARNING) {
		t.Error("WARNING should log when level is WARNING")
	}

	if !shouldLog(ERROR) {
		t.Error("ERROR should log when level is WARNING")
	}
}

func TestFormatMessage(t *testing.T) {
	message := formatMessage(INFO, "Test message %d", 42)

	// Check that it contains timestamp (HH:MM format)
	if !strings.Contains(message, ":") {
		t.Error("Expected message to contain timestamp with colon")
	}

	// Check that it contains the level
	if !strings.Contains(message, "INFO") {
		t.Error("Expected message to contain INFO level")
	}

	// Check that it contains the formatted message
	if !strings.Contains(message, "Test message 42") {
		t.Error("Expected message to contain formatted text")
	}

	// Check the general format
	parts := strings.Split(message, " - ")
	if len(parts) < 2 {
		t.Error("Expected message to have timestamp - level : message format")
	}
}

func TestLogLevelColors(t *testing.T) {
	tests := []struct {
		level    LogLevel
		hasColor bool
	}{
		{DEBUG, true},
		{INFO, true},
		{WARNING, true},
		{ERROR, true},
	}

	for _, tt := range tests {
		color := tt.level.Color()
		if tt.hasColor && color == "" {
			t.Errorf("Expected color for level %s", tt.level.String())
		}
		if tt.hasColor && !strings.HasPrefix(color, "\033[") {
			t.Errorf("Expected ANSI color code for level %s", tt.level.String())
		}
	}
}

func TestDebugRequest(t *testing.T) {
	originalLevel := currentLevel
	defer func() { currentLevel = originalLevel }()

	// Capture output by redirecting stdout
	oldStdout := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w

	SetLevel(DEBUG)
	DebugRequest("vllm", "http://localhost:8000", map[string]interface{}{
		"temperature": 0.7,
		"max_tokens":  100,
	})

	w.Close()
	os.Stdout = oldStdout

	var buf bytes.Buffer
	buf.ReadFrom(r)
	output := buf.String()

	// Should contain debug information
	if !strings.Contains(output, "vllm") {
		t.Error("Expected output to contain provider name")
	}
	if !strings.Contains(output, "http://localhost:8000") {
		t.Error("Expected output to contain URL")
	}
	if !strings.Contains(output, "temperature") {
		t.Error("Expected output to contain parameters")
	}
}

func TestDebugResponse(t *testing.T) {
	originalLevel := currentLevel
	defer func() { currentLevel = originalLevel }()

	// Test with verbose disabled - should not log
	SetLevel(INFO)
	// This should not produce output, but we can't easily test that without more complex stdout capture

	// Test with verbose enabled
	SetLevel(DEBUG)
	// DebugResponse should not panic
	DebugResponse(200, "test response", 100*time.Millisecond)
}

func TestDebugSystem(t *testing.T) {
	originalLevel := currentLevel
	defer func() { currentLevel = originalLevel }()

	SetLevel(DEBUG)
	// DebugSystem should not panic
	DebugSystem()
}

func TestDebugConfig(t *testing.T) {
	originalLevel := currentLevel
	defer func() { currentLevel = originalLevel }()

	SetLevel(DEBUG)
	// DebugConfig should not panic
	testConfig := map[string]interface{}{
		"provider": "vllm",
		"model":    "test-model",
	}
	DebugConfig(testConfig)
}

func TestLogFunctions(t *testing.T) {
	originalLevel := currentLevel
	defer func() { currentLevel = originalLevel }()

	SetLevel(DEBUG)

	// These should not panic
	Debug("Debug message %s", "test")
	Info("Info message %s", "test")
	Warning("Warning message %s", "test")
	Success("Success message %s", "test")
	Header("Header message %s", "test")
	Progress("Progress message %s", "test")

	// Error writes to stderr, so it's harder to test output
	Error("Error message %s", "test")
}

func TestFatal(t *testing.T) {
	// We can't actually test Fatal since it calls os.Exit(1)
	// This test just ensures the function exists and can be called in theory
	t.Log("Fatal function exists and is available for use")
}

func BenchmarkFormatMessage(b *testing.B) {
	for i := 0; i < b.N; i++ {
		formatMessage(INFO, "Test message %d", i)
	}
}

func BenchmarkShouldLog(b *testing.B) {
	SetLevel(INFO)
	for i := 0; i < b.N; i++ {
		shouldLog(INFO)
		shouldLog(DEBUG)
		shouldLog(WARNING)
		shouldLog(ERROR)
	}
}