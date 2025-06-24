package reporter

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/Vitruves/llm-client/internal/models"
)

func TestNew(t *testing.T) {
	results := createTestResults()
	reporter := New(results)

	if reporter == nil {
		t.Fatal("Expected reporter to be created")
	}

	if len(reporter.results) != len(results) {
		t.Errorf("Expected %d results, got %d", len(results), len(reporter.results))
	}

	if reporter.stats == nil {
		t.Fatal("Expected stats to be calculated")
	}
}

func TestGenerateText(t *testing.T) {
	results := createTestResults()
	reporter := New(results)

	text := reporter.GenerateText()

	if text == "" {
		t.Fatal("Expected text report to be generated")
	}

	// Check for key sections
	expectedSections := []string{
		"LLM CLASSIFICATION ANALYSIS REPORT",
		"OVERVIEW",
		"PERFORMANCE METRICS",
		"CLASSIFICATION METRICS",
		"SUMMARY",
	}

	for _, section := range expectedSections {
		if !strings.Contains(text, section) {
			t.Errorf("Expected report to contain section: %s", section)
		}
	}
}

func TestGenerateJSON(t *testing.T) {
	results := createTestResults()
	reporter := New(results)

	jsonStr, err := reporter.GenerateJSON()
	if err != nil {
		t.Fatalf("Failed to generate JSON: %v", err)
	}

	if jsonStr == "" {
		t.Fatal("Expected JSON report to be generated")
	}

	// Basic validation that it's JSON-like
	if !strings.HasPrefix(jsonStr, "{") || !strings.HasSuffix(strings.TrimSpace(jsonStr), "}") {
		t.Error("Generated content does not appear to be valid JSON")
	}
}

func TestSaveToFileText(t *testing.T) {
	results := createTestResults()
	reporter := New(results)

	tmpDir := t.TempDir()
	filename := filepath.Join(tmpDir, "report.txt")

	err := reporter.SaveToFile(filename, "text")
	if err != nil {
		t.Fatalf("Failed to save text report: %v", err)
	}

	// Verify file was created and contains content
	content, err := os.ReadFile(filename)
	if err != nil {
		t.Fatalf("Failed to read saved file: %v", err)
	}

	if len(content) == 0 {
		t.Error("Saved file is empty")
	}

	if !strings.Contains(string(content), "LLM CLASSIFICATION ANALYSIS REPORT") {
		t.Error("Saved file does not contain expected content")
	}
}

func TestSaveToFileJSON(t *testing.T) {
	results := createTestResults()
	reporter := New(results)

	tmpDir := t.TempDir()
	filename := filepath.Join(tmpDir, "report.json")

	err := reporter.SaveToFile(filename, "json")
	if err != nil {
		t.Fatalf("Failed to save JSON report: %v", err)
	}

	// Verify file was created and contains content
	content, err := os.ReadFile(filename)
	if err != nil {
		t.Fatalf("Failed to read saved file: %v", err)
	}

	if len(content) == 0 {
		t.Error("Saved file is empty")
	}

	jsonStr := string(content)
	if !strings.HasPrefix(jsonStr, "{") || !strings.HasSuffix(strings.TrimSpace(jsonStr), "}") {
		t.Error("Saved JSON file does not appear to be valid JSON")
	}
}

func TestSaveToFileCSV(t *testing.T) {
	results := createTestResults()
	reporter := New(results)

	tmpDir := t.TempDir()
	filename := filepath.Join(tmpDir, "report.csv")

	err := reporter.SaveToFile(filename, "csv")
	if err != nil {
		t.Fatalf("Failed to save CSV report: %v", err)
	}

	// Verify file was created and contains content
	content, err := os.ReadFile(filename)
	if err != nil {
		t.Fatalf("Failed to read saved file: %v", err)
	}

	if len(content) == 0 {
		t.Error("Saved file is empty")
	}

	// Check for CSV headers
	csvStr := string(content)
	if !strings.Contains(csvStr, "Section,Metric,Value") {
		t.Error("CSV file does not contain expected headers")
	}
}

func TestSaveToFileUnsupportedFormat(t *testing.T) {
	results := createTestResults()
	reporter := New(results)

	tmpDir := t.TempDir()
	filename := filepath.Join(tmpDir, "report.xyz")

	err := reporter.SaveToFile(filename, "xyz")
	if err == nil {
		t.Error("Expected error for unsupported format")
	}

	expectedError := "unsupported format: xyz"
	if err.Error() != expectedError {
		t.Errorf("Expected error '%s', got '%s'", expectedError, err.Error())
	}
}

func TestCalculateStats(t *testing.T) {
	results := createTestResults()
	stats := calculateStats(results)

	// Test basic statistics
	if stats.Total != len(results) {
		t.Errorf("Expected total %d, got %d", len(results), stats.Total)
	}

	if stats.Success == 0 {
		t.Error("Expected some successful results")
	}

	if stats.SuccessRate <= 0 || stats.SuccessRate > 100 {
		t.Errorf("Expected success rate between 0-100, got %.2f", stats.SuccessRate)
	}

	// Test classification metrics
	if stats.Accuracy < 0 || stats.Accuracy > 100 {
		t.Errorf("Expected accuracy between 0-100, got %.2f", stats.Accuracy)
	}

	// Test time statistics
	if stats.AvgTime <= 0 {
		t.Error("Expected positive average time")
	}

	if stats.TotalTime <= 0 {
		t.Error("Expected positive total time")
	}

	// Test throughput
	if stats.ThroughputPerSecond <= 0 {
		t.Error("Expected positive throughput")
	}
}

func TestCalculateStatsEmptyResults(t *testing.T) {
	var results []models.Result
	stats := calculateStats(results)

	if stats.Total != 0 {
		t.Errorf("Expected total 0 for empty results, got %d", stats.Total)
	}

	if stats.Success != 0 {
		t.Errorf("Expected success 0 for empty results, got %d", stats.Success)
	}

	if stats.SuccessRate != 0 {
		t.Errorf("Expected success rate 0 for empty results, got %.2f", stats.SuccessRate)
	}
}

func TestCalculateAccuracy(t *testing.T) {
	tests := []struct {
		name        string
		predictions []string
		actuals     []string
		expected    float64
	}{
		{
			name:        "perfect accuracy",
			predictions: []string{"positive", "negative", "neutral"},
			actuals:     []string{"positive", "negative", "neutral"},
			expected:    100.0,
		},
		{
			name:        "zero accuracy",
			predictions: []string{"positive", "positive", "positive"},
			actuals:     []string{"negative", "negative", "negative"},
			expected:    0.0,
		},
		{
			name:        "partial accuracy",
			predictions: []string{"positive", "negative", "positive"},
			actuals:     []string{"positive", "negative", "neutral"},
			expected:    66.66666666666666,
		},
		{
			name:        "empty arrays",
			predictions: []string{},
			actuals:     []string{},
			expected:    0.0,
		},
		{
			name:        "mismatched lengths",
			predictions: []string{"positive", "negative"},
			actuals:     []string{"positive"},
			expected:    0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			accuracy := calculateAccuracy(tt.predictions, tt.actuals)
			tolerance := 0.000001
			if abs(accuracy-tt.expected) > tolerance {
				t.Errorf("Expected accuracy %.6f, got %.6f", tt.expected, accuracy)
			}
		})
	}
}

func TestCalculateKappa(t *testing.T) {
	tests := []struct {
		name        string
		predictions []string
		actuals     []string
		expected    float64
	}{
		{
			name:        "perfect agreement",
			predictions: []string{"positive", "negative", "neutral"},
			actuals:     []string{"positive", "negative", "neutral"},
			expected:    1.0,
		},
		{
			name:        "no agreement",
			predictions: []string{"positive", "negative", "positive"},
			actuals:     []string{"negative", "positive", "negative"},
			expected:    -0.8,
		},
		{
			name:        "empty arrays",
			predictions: []string{},
			actuals:     []string{},
			expected:    0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			kappa := calculateKappa(tt.predictions, tt.actuals)
			tolerance := 0.001
			if abs(kappa-tt.expected) > tolerance {
				t.Errorf("Expected kappa %.3f, got %.3f", tt.expected, kappa)
			}
		})
	}
}

func TestCalculatePerClassMetrics(t *testing.T) {
	predictions := []string{"positive", "negative", "positive", "neutral", "negative"}
	actuals := []string{"positive", "negative", "negative", "neutral", "positive"}

	precision, recall, f1Score := calculatePerClassMetrics(predictions, actuals)

	// Test that all classes are represented
	expectedClasses := []string{"positive", "negative", "neutral"}
	for _, class := range expectedClasses {
		if _, exists := precision[class]; !exists {
			t.Errorf("Expected precision for class %s", class)
		}
		if _, exists := recall[class]; !exists {
			t.Errorf("Expected recall for class %s", class)
		}
		if _, exists := f1Score[class]; !exists {
			t.Errorf("Expected F1 score for class %s", class)
		}
	}

	// Test that values are reasonable (between 0 and 1)
	for class, p := range precision {
		if p < 0 || p > 1 {
			t.Errorf("Precision for class %s out of range [0,1]: %f", class, p)
		}
	}

	for class, r := range recall {
		if r < 0 || r > 1 {
			t.Errorf("Recall for class %s out of range [0,1]: %f", class, r)
		}
	}

	for class, f1 := range f1Score {
		if f1 < 0 || f1 > 1 {
			t.Errorf("F1 score for class %s out of range [0,1]: %f", class, f1)
		}
	}
}

func TestCalculateConfusionMatrix(t *testing.T) {
	predictions := []string{"positive", "negative", "positive", "neutral"}
	actuals := []string{"positive", "negative", "negative", "neutral"}

	matrix := calculateConfusionMatrix(predictions, actuals)

	// Test that matrix is created for all classes
	expectedClasses := []string{"positive", "negative", "neutral"}
	for _, actual := range expectedClasses {
		if _, exists := matrix[actual]; !exists {
			t.Errorf("Expected row for actual class %s", actual)
		}
		for _, predicted := range expectedClasses {
			if _, exists := matrix[actual][predicted]; !exists {
				t.Errorf("Expected cell for actual=%s, predicted=%s", actual, predicted)
			}
		}
	}

	// Test specific values
	if matrix["positive"]["positive"] != 1 {
		t.Errorf("Expected 1 for positive->positive, got %d", matrix["positive"]["positive"])
	}
	if matrix["negative"]["negative"] != 1 {
		t.Errorf("Expected 1 for negative->negative, got %d", matrix["negative"]["negative"])
	}
	if matrix["negative"]["positive"] != 1 {
		t.Errorf("Expected 1 for negative->positive, got %d", matrix["negative"]["positive"])
	}
}

func TestTimeStatistics(t *testing.T) {
	results := createTestResults()
	stats := calculateStats(results)

	// Test that time statistics are calculated
	if stats.AvgTime <= 0 {
		t.Error("Expected positive average time")
	}

	if stats.MinTime <= 0 {
		t.Error("Expected positive minimum time")
	}

	if stats.MaxTime <= 0 {
		t.Error("Expected positive maximum time")
	}

	if stats.P95Time <= 0 {
		t.Error("Expected positive P95 time")
	}

	if stats.P99Time <= 0 {
		t.Error("Expected positive P99 time")
	}

	// Test logical relationships
	if stats.MinTime > stats.AvgTime {
		t.Error("Min time should be less than or equal to average time")
	}

	if stats.AvgTime > stats.MaxTime {
		t.Error("Average time should be less than or equal to max time")
	}
}

func TestConsensusStats(t *testing.T) {
	results := []models.Result{
		{
			Success: true,
			Consensus: &models.Consensus{
				Total:        3,
				Ratio:        0.8,
				Distribution: map[string]int{"positive": 2, "negative": 1},
			},
		},
		{
			Success: true,
			Consensus: &models.Consensus{
				Total:        3,
				Ratio:        1.0,
				Distribution: map[string]int{"negative": 3},
			},
		},
	}

	consensusStats := calculateConsensusStats(results)

	if consensusStats == nil {
		t.Fatal("Expected consensus stats to be calculated")
	}

	if consensusStats.ItemsWithConsensus != 2 {
		t.Errorf("Expected 2 items with consensus, got %d", consensusStats.ItemsWithConsensus)
	}

	if consensusStats.RepeatCount != 3 {
		t.Errorf("Expected repeat count 3, got %d", consensusStats.RepeatCount)
	}

	expectedAvg := 0.9
	if abs(consensusStats.AvgConsensusRatio-expectedAvg) > 0.001 {
		t.Errorf("Expected average consensus ratio %.3f, got %.3f", expectedAvg, consensusStats.AvgConsensusRatio)
	}
}

// Helper function to create test results
func createTestResults() []models.Result {
	return []models.Result{
		{
			Index:        0,
			FinalAnswer:  "positive",
			GroundTruth:  "positive",
			Success:      true,
			ResponseTime: 100 * time.Millisecond,
		},
		{
			Index:        1,
			FinalAnswer:  "negative",
			GroundTruth:  "negative",
			Success:      true,
			ResponseTime: 150 * time.Millisecond,
		},
		{
			Index:        2,
			FinalAnswer:  "positive",
			GroundTruth:  "negative",
			Success:      true,
			ResponseTime: 200 * time.Millisecond,
		},
		{
			Index:        3,
			FinalAnswer:  "neutral",
			GroundTruth:  "neutral",
			Success:      true,
			ResponseTime: 120 * time.Millisecond,
		},
		{
			Index:        4,
			FinalAnswer:  "",
			GroundTruth:  "positive",
			Success:      false,
			Error:        "timeout error",
			ResponseTime: 5 * time.Second,
		},
	}
}

// Helper function for absolute value
func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}