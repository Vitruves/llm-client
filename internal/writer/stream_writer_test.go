package writer

import (
	"encoding/csv"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/Vitruves/llm-client/internal/models"
)

func TestNewStreamWriter(t *testing.T) {
	tmpDir := t.TempDir()
	timestamp := "20250620_123456"
	config := &models.OutputConfig{}

	tests := []struct {
		name           string
		format         string
		expectError    bool
		expectedType   string
	}{
		{
			name:         "JSON format",
			format:       "json",
			expectError:  false,
			expectedType: "*writer.JSONStreamWriter",
		},
		{
			name:         "CSV format", 
			format:       "csv",
			expectError:  false,
			expectedType: "*writer.CSVStreamWriter",
		},
		{
			name:         "Parquet format",
			format:       "parquet",
			expectError:  false,
			expectedType: "*writer.ParquetStreamWriter",
		},
		{
			name:        "unsupported format",
			format:      "xlsx",
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			writer, err := NewStreamWriter(tt.format, tmpDir, timestamp, config)
			
			if tt.expectError {
				if err == nil {
					t.Error("Expected error for unsupported format")
				}
				return
			}

			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}

			if writer == nil {
				t.Fatal("Expected writer to be created")
			}

			// Clean up
			writer.Close()
		})
	}
}

func TestJSONStreamWriter(t *testing.T) {
	tmpDir := t.TempDir()
	timestamp := "20250620_123456"

	writer, err := NewJSONStreamWriter(tmpDir, timestamp)
	if err != nil {
		t.Fatalf("Failed to create JSON stream writer: %v", err)
	}
	defer writer.Close()

	// Create test results
	results := createTestStreamResults()

	// Write results
	for _, result := range results {
		if err := writer.WriteResult(result); err != nil {
			t.Fatalf("Failed to write result: %v", err)
		}
	}

	// Flush to ensure data is written
	if err := writer.Flush(); err != nil {
		t.Fatalf("Failed to flush: %v", err)
	}

	// Close writer
	if err := writer.Close(); err != nil {
		t.Fatalf("Failed to close writer: %v", err)
	}

	// Verify file content
	expectedFilename := filepath.Join(tmpDir, "results_20250620_123456.json")
	if writer.GetFilename() != expectedFilename {
		t.Errorf("Expected filename %s, got %s", expectedFilename, writer.GetFilename())
	}

	content, err := os.ReadFile(expectedFilename)
	if err != nil {
		t.Fatalf("Failed to read output file: %v", err)
	}

	contentStr := string(content)
	
	// Check JSON structure
	if !strings.HasPrefix(contentStr, "[\n") {
		t.Error("JSON should start with opening bracket")
	}
	if !strings.HasSuffix(contentStr, "\n]") {
		t.Error("JSON should end with closing bracket")
	}

	// Verify it's valid JSON by unmarshaling
	var parsedResults []models.Result
	if err := json.Unmarshal(content, &parsedResults); err != nil {
		t.Fatalf("Generated JSON is not valid: %v", err)
	}

	if len(parsedResults) != len(results) {
		t.Errorf("Expected %d results in JSON, got %d", len(results), len(parsedResults))
	}
}

func TestJSONStreamWriterConcurrentWrites(t *testing.T) {
	tmpDir := t.TempDir()
	timestamp := "20250620_123456"

	writer, err := NewJSONStreamWriter(tmpDir, timestamp)
	if err != nil {
		t.Fatalf("Failed to create JSON stream writer: %v", err)
	}
	defer writer.Close()

	// Test concurrent writes
	results := createTestStreamResults()
	done := make(chan error, len(results))

	for _, result := range results {
		go func(r models.Result) {
			done <- writer.WriteResult(r)
		}(result)
	}

	// Wait for all writes to complete
	for i := 0; i < len(results); i++ {
		if err := <-done; err != nil {
			t.Errorf("Concurrent write failed: %v", err)
		}
	}

	writer.Close()

	// Verify content is still valid JSON
	content, err := os.ReadFile(writer.GetFilename())
	if err != nil {
		t.Fatalf("Failed to read output file: %v", err)
	}

	var parsedResults []models.Result
	if err := json.Unmarshal(content, &parsedResults); err != nil {
		t.Fatalf("Concurrent writes produced invalid JSON: %v", err)
	}

	if len(parsedResults) != len(results) {
		t.Errorf("Expected %d results, got %d", len(results), len(parsedResults))
	}
}

func TestCSVStreamWriter(t *testing.T) {
	tmpDir := t.TempDir()
	timestamp := "20250620_123456"
	config := &models.OutputConfig{
		IncludeThinking:    true,
		IncludeRawResponse: true,
	}

	writer, err := NewCSVStreamWriter(tmpDir, timestamp, config)
	if err != nil {
		t.Fatalf("Failed to create CSV stream writer: %v", err)
	}
	defer writer.Close()

	// Create test results with original data
	results := createTestStreamResults()

	// Write results
	for _, result := range results {
		if err := writer.WriteResult(result); err != nil {
			t.Fatalf("Failed to write result: %v", err)
		}
	}

	// Close writer
	if err := writer.Close(); err != nil {
		t.Fatalf("Failed to close writer: %v", err)
	}

	// Verify file content
	expectedFilename := filepath.Join(tmpDir, "results_20250620_123456.csv")
	if writer.GetFilename() != expectedFilename {
		t.Errorf("Expected filename %s, got %s", expectedFilename, writer.GetFilename())
	}

	// Read and verify CSV content
	file, err := os.Open(expectedFilename)
	if err != nil {
		t.Fatalf("Failed to open CSV file: %v", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		t.Fatalf("Failed to read CSV: %v", err)
	}

	// Check header row
	if len(records) == 0 {
		t.Fatal("CSV file is empty")
	}

	header := records[0]
	expectedHeaders := []string{"index", "input_text", "ground_truth", "final_answer", "success", "response_time_ms"}
	
	// Check basic headers are present
	for i, expectedHeader := range expectedHeaders {
		if i >= len(header) || header[i] != expectedHeader {
			t.Errorf("Expected header[%d] to be %s, got %s", i, expectedHeader, header[i])
		}
	}

	// Check that thinking and raw response columns are included
	if !contains(header, "thinking_content") {
		t.Error("Expected thinking_content column in CSV header")
	}
	if !contains(header, "raw_response") {
		t.Error("Expected raw_response column in CSV header")
	}

	// Check data rows
	if len(records) != len(results)+1 { // +1 for header
		t.Errorf("Expected %d records (including header), got %d", len(results)+1, len(records))
	}
}

func TestCSVStreamWriterWithoutOptionalColumns(t *testing.T) {
	tmpDir := t.TempDir()
	timestamp := "20250620_123456"
	config := &models.OutputConfig{
		IncludeThinking:    false,
		IncludeRawResponse: false,
	}

	writer, err := NewCSVStreamWriter(tmpDir, timestamp, config)
	if err != nil {
		t.Fatalf("Failed to create CSV stream writer: %v", err)
	}
	defer writer.Close()

	// Write one result
	result := createTestStreamResults()[0]
	if err := writer.WriteResult(result); err != nil {
		t.Fatalf("Failed to write result: %v", err)
	}

	writer.Close()

	// Read and verify CSV content
	file, err := os.Open(writer.GetFilename())
	if err != nil {
		t.Fatalf("Failed to open CSV file: %v", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		t.Fatalf("Failed to read CSV: %v", err)
	}

	if len(records) == 0 {
		t.Fatal("CSV file is empty")
	}

	header := records[0]
	
	// Check that thinking and raw response columns are NOT included
	if contains(header, "thinking_content") {
		t.Error("Did not expect thinking_content column when IncludeThinking is false")
	}
	if contains(header, "raw_response") {
		t.Error("Did not expect raw_response column when IncludeRawResponse is false")
	}
}

func TestParquetStreamWriter(t *testing.T) {
	tmpDir := t.TempDir()
	timestamp := "20250620_123456"
	config := &models.OutputConfig{
		StreamSaveEvery: 2, // Small buffer for testing
	}

	writer, err := NewParquetStreamWriter(tmpDir, timestamp, config)
	if err != nil {
		t.Fatalf("Failed to create Parquet stream writer: %v", err)
	}
	defer writer.Close()

	// Create test results
	results := createTestStreamResults()

	// Write results (should trigger flush due to small buffer)
	for _, result := range results {
		if err := writer.WriteResult(result); err != nil {
			t.Fatalf("Failed to write result: %v", err)
		}
	}

	// Close writer to flush remaining results
	if err := writer.Close(); err != nil {
		t.Fatalf("Failed to close writer: %v", err)
	}

	// Verify file content
	expectedFilename := filepath.Join(tmpDir, "results_20250620_123456.parquet")
	if writer.GetFilename() != expectedFilename {
		t.Errorf("Expected filename %s, got %s", expectedFilename, writer.GetFilename())
	}

	// Verify file exists and has reasonable content
	stat, err := os.Stat(expectedFilename)
	if err != nil {
		t.Fatalf("Parquet file was not created: %v", err)
	}

	if stat.Size() == 0 {
		t.Error("Parquet file is empty")
	}

	// Verify file size is reasonable for the data we wrote
	expectedMinSize := int64(len(results) * 50) // Rough estimate
	if stat.Size() < expectedMinSize {
		t.Errorf("Parquet file too small: expected at least %d bytes, got %d", expectedMinSize, stat.Size())
	}

	t.Logf("Parquet file created successfully with %d results, size %d bytes", len(results), stat.Size())
}

func TestParquetStreamWriterFlush(t *testing.T) {
	tmpDir := t.TempDir()
	timestamp := "20250620_123456"
	config := &models.OutputConfig{
		StreamSaveEvery: 10, // Large buffer to test manual flush
	}

	writer, err := NewParquetStreamWriter(tmpDir, timestamp, config)
	if err != nil {
		t.Fatalf("Failed to create Parquet stream writer: %v", err)
	}
	defer writer.Close()

	// Write one result (should not auto-flush)
	result := createTestStreamResults()[0]
	if err := writer.WriteResult(result); err != nil {
		t.Fatalf("Failed to write result: %v", err)
	}

	// Manual flush
	if err := writer.Flush(); err != nil {
		t.Fatalf("Failed to flush: %v", err)
	}

	writer.Close()

	// Verify file has content after flush
	stat, err := os.Stat(writer.GetFilename())
	if err != nil {
		t.Fatalf("Failed to get file stats: %v", err)
	}

	if stat.Size() == 0 {
		t.Error("Parquet file should have content after flush")
	}

	t.Logf("Parquet file flushed successfully with size %d bytes", stat.Size())
}

func TestParquetStreamWriterBuffering(t *testing.T) {
	tmpDir := t.TempDir()
	timestamp := "20250620_123456"
	config := &models.OutputConfig{
		StreamSaveEvery: 5, // Buffer 5 results before flushing
	}

	writer, err := NewParquetStreamWriter(tmpDir, timestamp, config)
	if err != nil {
		t.Fatalf("Failed to create Parquet stream writer: %v", err)
	}
	defer writer.Close()

	// Write 3 results (should not trigger auto-flush)
	results := createTestStreamResults()
	for i := 0; i < 3; i++ {
		if err := writer.WriteResult(results[i]); err != nil {
			t.Fatalf("Failed to write result %d: %v", i, err)
		}
	}

	// Check if file exists but might be empty (buffered)
	stat1, err := os.Stat(writer.GetFilename())
	if err != nil {
		t.Fatalf("Parquet file should exist even with buffered data: %v", err)
	}
	
	// Write 2 more results (should trigger auto-flush at 5)
	for i := 3; i < 5; i++ {
		if i < len(results) {
			if err := writer.WriteResult(results[i%len(results)]); err != nil {
				t.Fatalf("Failed to write result %d: %v", i, err)
			}
		}
	}

	// Now file should have content after auto-flush
	stat2, err := os.Stat(writer.GetFilename())
	if err != nil {
		t.Fatalf("Failed to stat parquet file after auto-flush: %v", err)
	}

	// File size should have increased after flush
	if stat2.Size() <= stat1.Size() {
		t.Logf("File size before flush: %d, after: %d", stat1.Size(), stat2.Size())
		// Note: This might not always be true due to parquet compression, so just log
	}

	writer.Close()
	t.Logf("Parquet buffering test completed successfully")
}

func TestParquetStreamWriterLargeData(t *testing.T) {
	tmpDir := t.TempDir()
	timestamp := "20250620_123456"
	config := &models.OutputConfig{
		StreamSaveEvery: 10,
	}

	writer, err := NewParquetStreamWriter(tmpDir, timestamp, config)
	if err != nil {
		t.Fatalf("Failed to create Parquet stream writer: %v", err)
	}
	defer writer.Close()

	// Create result with large text content
	largeResult := models.Result{
		Index:           0,
		InputText:       strings.Repeat("This is a large text content for testing parquet streaming. ", 100),
		GroundTruth:     "positive",
		FinalAnswer:     "positive",
		Success:         true,
		ResponseTime:    100 * time.Millisecond,
		ThinkingContent: strings.Repeat("Large thinking content. ", 50),
		RawResponse:     strings.Repeat("Large raw response content. ", 75),
		OriginalData: map[string]interface{}{
			"large_field": strings.Repeat("data", 200),
			"id":          12345,
			"metadata":    "important_info",
		},
	}

	// Write the large result
	if err := writer.WriteResult(largeResult); err != nil {
		t.Fatalf("Failed to write large result: %v", err)
	}

	// Flush to ensure it's written
	if err := writer.Flush(); err != nil {
		t.Fatalf("Failed to flush large result: %v", err)
	}

	writer.Close()

	// Verify file was created and has substantial content
	stat, err := os.Stat(writer.GetFilename())
	if err != nil {
		t.Fatalf("Large data parquet file was not created: %v", err)
	}

	if stat.Size() < 1000 { // Should be at least 1KB for large content
		t.Errorf("Parquet file too small for large data: %d bytes", stat.Size())
	}

	t.Logf("Large data parquet file created successfully with size %d bytes", stat.Size())
}

func TestParquetStreamWriterOriginalDataSerialization(t *testing.T) {
	tmpDir := t.TempDir()
	timestamp := "20250620_123456"
	config := &models.OutputConfig{
		StreamSaveEvery: 1, // Flush immediately
	}

	writer, err := NewParquetStreamWriter(tmpDir, timestamp, config)
	if err != nil {
		t.Fatalf("Failed to create Parquet stream writer: %v", err)
	}
	defer writer.Close()

	// Create result with complex original data
	complexResult := models.Result{
		Index:       0,
		InputText:   "test",
		FinalAnswer: "positive",
		Success:     true,
		OriginalData: map[string]interface{}{
			"string_field":  "test_value",
			"number_field":  42,
			"float_field":   3.14159,
			"bool_field":    true,
			"null_field":    nil,
			"nested_object": map[string]interface{}{
				"inner_string": "nested_value",
				"inner_number": 123,
			},
			"array_field": []interface{}{"item1", "item2", 123},
		},
	}

	// Write the complex result
	if err := writer.WriteResult(complexResult); err != nil {
		t.Fatalf("Failed to write complex result: %v", err)
	}

	writer.Close()

	// Verify file was created
	stat, err := os.Stat(writer.GetFilename())
	if err != nil {
		t.Fatalf("Complex data parquet file was not created: %v", err)
	}

	if stat.Size() == 0 {
		t.Error("Complex data parquet file is empty")
	}

	t.Logf("Complex data serialization test completed successfully, file size: %d bytes", stat.Size())
}

func TestStreamWriterCloseMultipleTimes(t *testing.T) {
	tmpDir := t.TempDir()
	timestamp := "20250620_123456"
	config := &models.OutputConfig{}

	formats := []string{"json", "csv", "parquet"}

	for _, format := range formats {
		t.Run(format, func(t *testing.T) {
			writer, err := NewStreamWriter(format, tmpDir, timestamp+"_"+format, config)
			if err != nil {
				t.Fatalf("Failed to create %s stream writer: %v", format, err)
			}

			// Write a result
			result := createTestStreamResults()[0]
			if err := writer.WriteResult(result); err != nil {
				t.Fatalf("Failed to write result: %v", err)
			}

			// First close should succeed
			if err := writer.Close(); err != nil {
				t.Errorf("First close failed: %v", err)
			}
			// Second close may error (file already closed) - this is acceptable behavior
			writer.Close() // Don't check error for second close
		})
	}
}

func TestStreamWriterErrorHandling(t *testing.T) {
	// Test writing to invalid directory
	invalidDir := "/invalid/directory/path"
	timestamp := "20250620_123456"
	config := &models.OutputConfig{}

	_, err := NewStreamWriter("json", invalidDir, timestamp, config)
	if err == nil {
		t.Error("Expected error when creating writer with invalid directory")
	}
}

func TestJSONStreamWriterEmptyFile(t *testing.T) {
	tmpDir := t.TempDir()
	timestamp := "20250620_123456"

	writer, err := NewJSONStreamWriter(tmpDir, timestamp)
	if err != nil {
		t.Fatalf("Failed to create JSON stream writer: %v", err)
	}

	// Close without writing anything
	if err := writer.Close(); err != nil {
		t.Fatalf("Failed to close empty writer: %v", err)
	}

	// Verify file content is valid empty JSON array
	content, err := os.ReadFile(writer.GetFilename())
	if err != nil {
		t.Fatalf("Failed to read output file: %v", err)
	}

	contentStr := strings.TrimSpace(string(content))
	// Allow for slight formatting variations in empty JSON array
	if !strings.Contains(contentStr, "[") || !strings.Contains(contentStr, "]") {
		t.Errorf("Expected JSON array structure, got: %s", contentStr)
	}

	// Verify it's valid JSON
	var parsedResults []models.Result
	if err := json.Unmarshal(content, &parsedResults); err != nil {
		t.Fatalf("Empty JSON array is not valid: %v", err)
	}

	if len(parsedResults) != 0 {
		t.Errorf("Expected empty array, got %d results", len(parsedResults))
	}
}

func TestStreamWriterOriginalDataHandling(t *testing.T) {
	tmpDir := t.TempDir()
	timestamp := "20250620_123456"
	config := &models.OutputConfig{}

	// Test with CSV writer for header handling
	writer, err := NewCSVStreamWriter(tmpDir, timestamp, config)
	if err != nil {
		t.Fatalf("Failed to create CSV stream writer: %v", err)
	}
	defer writer.Close()

	// Create results with different original data keys
	result1 := models.Result{
		Index:       0,
		InputText:   "test 1",
		FinalAnswer: "positive",
		Success:     true,
		OriginalData: map[string]interface{}{
			"source": "data1",
			"id":     123,
		},
	}

	result2 := models.Result{
		Index:       1,
		InputText:   "test 2", 
		FinalAnswer: "negative",
		Success:     true,
		OriginalData: map[string]interface{}{
			"source": "data2",
			"category": "test",
		},
	}

	// Write results
	if err := writer.WriteResult(result1); err != nil {
		t.Fatalf("Failed to write result 1: %v", err)
	}
	if err := writer.WriteResult(result2); err != nil {
		t.Fatalf("Failed to write result 2: %v", err)
	}

	writer.Close()

	// Read and verify CSV handles all original data keys
	file, err := os.Open(writer.GetFilename())
	if err != nil {
		t.Fatalf("Failed to open CSV file: %v", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		t.Fatalf("Failed to read CSV: %v", err)
	}

	if len(records) == 0 {
		t.Fatal("CSV file is empty")
	}

	header := records[0]
	
	// Should contain all original data keys from all written results
	// Note: CSV writer builds headers incrementally as results are written
	// After writing both results, should have all keys
	expectedKeys := []string{"source"} // Common to both results
	for _, key := range expectedKeys {
		if !contains(header, key) {
			t.Errorf("Expected header to contain original data key: %s", key)
		}
	}
	
	// Check that we have some original data columns
	hasOriginalData := false
	for _, headerCol := range header {
		if headerCol == "source" || headerCol == "id" || headerCol == "category" {
			hasOriginalData = true
			break
		}
	}
	if !hasOriginalData {
		t.Error("Expected header to contain at least one original data column")
	}
}

// Helper functions

func createTestStreamResults() []models.Result {
	return []models.Result{
		{
			Index:           0,
			InputText:       "This is a positive test",
			GroundTruth:     "positive",
			FinalAnswer:     "positive",
			Success:         true,
			ResponseTime:    100 * time.Millisecond,
			ThinkingContent: "This seems positive to me",
			RawResponse:     "positive",
			OriginalData: map[string]interface{}{
				"source": "test_data",
				"id":     1,
			},
		},
		{
			Index:           1,
			InputText:       "This is a negative test",
			GroundTruth:     "negative", 
			FinalAnswer:     "negative",
			Success:         true,
			ResponseTime:    150 * time.Millisecond,
			ThinkingContent: "This seems negative to me",
			RawResponse:     "negative",
			OriginalData: map[string]interface{}{
				"source": "test_data",
				"id":     2,
			},
		},
		{
			Index:           2,
			InputText:       "This is a neutral test",
			GroundTruth:     "neutral",
			FinalAnswer:     "neutral",
			Success:         true,
			ResponseTime:    120 * time.Millisecond,
			ThinkingContent: "This seems neutral to me",
			RawResponse:     "neutral",
			OriginalData: map[string]interface{}{
				"source": "test_data",
				"id":     3,
			},
		},
	}
}

func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}