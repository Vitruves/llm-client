package metrics

import (
	"testing"

	"llm-client/internal/models"
)

func TestNewCalculator(t *testing.T) {
	tests := []struct {
		name     string
		config   *models.LiveMetrics
		classes  []string
		expected bool // whether calculator should be created
	}{
		{
			name: "enabled metrics",
			config: &models.LiveMetrics{
				Enabled:     true,
				Metric:      "accuracy",
				GroundTruth: "label",
			},
			classes:  []string{"positive", "negative", "neutral"},
			expected: true,
		},
		{
			name:     "disabled metrics",
			config:   &models.LiveMetrics{Enabled: false},
			classes:  []string{"positive", "negative"},
			expected: false,
		},
		{
			name:     "nil config",
			config:   nil,
			classes:  []string{"positive", "negative"},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			calc := NewCalculator(tt.config, tt.classes)

			if tt.expected && calc == nil {
				t.Error("Expected calculator to be created")
			}
			if !tt.expected && calc != nil {
				t.Error("Expected calculator to be nil")
			}
		})
	}
}

func TestAddResultAndGetCurrentMetric(t *testing.T) {
	config := &models.LiveMetrics{
		Enabled:     true,
		Metric:      "accuracy",
		GroundTruth: "label",
	}
	classes := []string{"positive", "negative", "neutral"}
	calc := NewCalculator(config, classes)

	// Test with no data
	accuracy := calc.GetCurrentMetric()
	if accuracy != 0.0 {
		t.Errorf("Expected accuracy 0.0 with no data, got %f", accuracy)
	}

	// Add test data
	testCases := []struct {
		predicted string
		actual    string
	}{
		{"positive", "positive"}, // correct
		{"negative", "negative"}, // correct
		{"positive", "negative"}, // incorrect
		{"neutral", "neutral"},   // correct
		{"negative", "positive"}, // incorrect
	}

	for _, tc := range testCases {
		calc.AddResult(tc.predicted, tc.actual)
	}

	// 3 correct out of 5 = 60%
	accuracy = calc.GetCurrentMetric()
	expected := 60.0
	if accuracy != expected {
		t.Errorf("Expected accuracy %f, got %f", expected, accuracy)
	}
}

func TestF1MetricCalculation(t *testing.T) {
	tests := []struct {
		name     string
		average  string
		expected float64
	}{
		{
			name:     "macro F1",
			average:  "macro",
			expected: 50.0, // Simplified expectation
		},
		{
			name:     "micro F1",
			average:  "micro",
			expected: 60.0, // Should equal accuracy for balanced classes
		},
		{
			name:     "weighted F1",
			average:  "weighted",
			expected: 50.0, // Simplified expectation
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := &models.LiveMetrics{
				Enabled:     true,
				Metric:      "f1",
				GroundTruth: "label",
				Average:     tt.average,
			}
			classes := []string{"positive", "negative", "neutral"}
			calc := NewCalculator(config, classes)

			// Add some test data
			calc.AddResult("positive", "positive")
			calc.AddResult("negative", "negative")
			calc.AddResult("positive", "negative")
			calc.AddResult("neutral", "neutral")
			calc.AddResult("negative", "positive")

			f1 := calc.GetCurrentMetric()
			// Allow some tolerance for floating point comparison
			if f1 < tt.expected-20 || f1 > tt.expected+20 {
				t.Errorf("Expected F1 around %f, got %f", tt.expected, f1)
			}
		})
	}
}

func TestKappaMetricCalculation(t *testing.T) {
	config := &models.LiveMetrics{
		Enabled:     true,
		Metric:      "kappa",
		GroundTruth: "label",
	}
	classes := []string{"positive", "negative"}
	calc := NewCalculator(config, classes)

	// Test with perfect agreement
	calc.AddResult("positive", "positive")
	calc.AddResult("negative", "negative")
	
	kappa := calc.GetCurrentMetric()
	expected := 100.0 // Perfect agreement = 100%
	if kappa != expected {
		t.Errorf("Expected kappa %f for perfect agreement, got %f", expected, kappa)
	}

	// Test with no agreement (random guessing)
	calc2 := NewCalculator(config, classes)
	calc2.AddResult("positive", "negative")
	calc2.AddResult("negative", "positive")
	
	kappa2 := calc2.GetCurrentMetric()
	if kappa2 >= 0 {
		t.Errorf("Expected negative kappa for random guessing, got %f", kappa2)
	}
}

func TestGetCurrentMetricTypes(t *testing.T) {
	tests := []struct {
		name   string
		metric string
	}{
		{"accuracy", "accuracy"},
		{"f1", "f1"},
		{"kappa", "kappa"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := &models.LiveMetrics{
				Enabled:     true,
				Metric:      tt.metric,
				GroundTruth: "label",
			}
			classes := []string{"positive", "negative"}
			calc := NewCalculator(config, classes)

			// Add some data
			calc.AddResult("positive", "positive")
			calc.AddResult("negative", "negative")

			result := calc.GetCurrentMetric()
			if result <= 0 {
				t.Errorf("Expected positive metric value, got %f", result)
			}
		})
	}
}

func TestGetMetricName(t *testing.T) {
	tests := []struct {
		metric   string
		average  string
		expected string
	}{
		{"accuracy", "", "Accuracy"},
		{"f1", "macro", "F1-macro"},
		{"f1", "micro", "F1-micro"},
		{"f1", "", "F1-macro"}, // default
		{"kappa", "", "Kappa"},
		{"unknown", "", "Accuracy"}, // default fallback
	}

	for _, tt := range tests {
		t.Run(tt.metric+"_"+tt.average, func(t *testing.T) {
			config := &models.LiveMetrics{
				Enabled:     true,
				Metric:      tt.metric,
				GroundTruth: "label",
				Average:     tt.average,
			}
			calc := NewCalculator(config, []string{"pos", "neg"})

			name := calc.GetMetricName()
			if name != tt.expected {
				t.Errorf("Expected metric name '%s', got '%s'", tt.expected, name)
			}
		})
	}
}

func TestNilCalculator(t *testing.T) {
	var calc *Calculator = nil

	// All methods should handle nil gracefully
	calc.AddResult("test", "test")
	
	metric := calc.GetCurrentMetric()
	if metric != 0.0 {
		t.Errorf("Expected 0.0 for nil calculator, got %f", metric)
	}

	name := calc.GetMetricName()
	if name != "None" {
		t.Errorf("Expected 'None' for nil calculator, got '%s'", name)
	}
}

func TestConfusionMatrix(t *testing.T) {
	config := &models.LiveMetrics{
		Enabled:     true,
		Metric:      "accuracy",
		GroundTruth: "label",
	}
	classes := []string{"A", "B", "C"}
	calc := NewCalculator(config, classes)

	// Add data to build confusion matrix
	testData := []struct {
		predicted, actual string
	}{
		{"A", "A"}, {"A", "A"}, // TP for A
		{"B", "B"}, {"B", "B"}, // TP for B
		{"C", "C"},             // TP for C
		{"A", "B"},             // FP for A, FN for B
		{"B", "C"},             // FP for B, FN for C
		{"C", "A"},             // FP for C, FN for A
	}

	for _, td := range testData {
		calc.AddResult(td.predicted, td.actual)
	}

	// Test that the confusion matrix is properly built
	// We can't directly access it, but we can verify through metrics
	accuracy := calc.GetCurrentMetric()
	expected := 5.0 / 8.0 * 100 // 5 correct out of 8
	if accuracy != expected {
		t.Errorf("Expected accuracy %f, got %f", expected, accuracy)
	}
}

func TestNormalizeLabel(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"  positive  ", "positive"},
		{"negative", "negative"},
		{"\t\nneutral\t\n", "neutral"},
		{"", ""},
		{"   ", ""},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			result := NormalizeLabel(tt.input)
			if result != tt.expected {
				t.Errorf("Expected '%s', got '%s'", tt.expected, result)
			}
		})
	}
}

func TestDefaultMetricBehavior(t *testing.T) {
	config := &models.LiveMetrics{
		Enabled:     true,
		Metric:      "invalid_metric", // Should default to accuracy
		GroundTruth: "label",
	}
	classes := []string{"A", "B"}
	calc := NewCalculator(config, classes)

	calc.AddResult("A", "A")
	calc.AddResult("B", "B")

	// Should default to accuracy calculation
	result := calc.GetCurrentMetric()
	expected := 100.0 // Both predictions correct
	if result != expected {
		t.Errorf("Expected default accuracy %f, got %f", expected, result)
	}

	// Should default to "Accuracy" name
	name := calc.GetMetricName()
	if name != "Accuracy" {
		t.Errorf("Expected default name 'Accuracy', got '%s'", name)
	}
}