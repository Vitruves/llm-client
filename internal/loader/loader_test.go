package loader

import (
	"os"
	"path/filepath"
	"testing"

	"llm-client/internal/models"
)

func TestLoadCSV(t *testing.T) {
	tests := []struct {
		name        string
		csvContent  string
		expectError bool
		expectRows  int
		validate    func([]models.DataRow) error
	}{
		{
			name: "basic CSV with headers",
			csvContent: `text,label
"This is positive",positive
"This is negative",negative
"This is neutral",neutral`,
			expectError: false,
			expectRows:  3,
			validate: func(rows []models.DataRow) error {
				if rows[0].Text != "This is positive" {
					t.Errorf("Expected first row text 'This is positive', got '%s'", rows[0].Text)
				}
				if rows[0].Data["label"] != "positive" {
					t.Errorf("Expected first row label 'positive', got '%v'", rows[0].Data["label"])
				}
				if rows[0].Index != 0 {
					t.Errorf("Expected first row index 0, got %d", rows[0].Index)
				}
				return nil
			},
		},
		{
			name: "CSV with content column",
			csvContent: `content,category,score
"Great product",positive,5
"Terrible service",negative,1`,
			expectError: false,
			expectRows:  2,
			validate: func(rows []models.DataRow) error {
				if rows[0].Text != "Great product" {
					t.Errorf("Expected text to be set from content column, got '%s'", rows[0].Text)
				}
				return nil
			},
		},
		{
			name: "CSV with no text/content column",
			csvContent: `review,rating
"Good product",5
"Bad product",1`,
			expectError: false,
			expectRows:  2,
			validate: func(rows []models.DataRow) error {
				// Should use first column as text
				if rows[0].Text != "Good product" {
					t.Errorf("Expected text from first column, got '%s'", rows[0].Text)
				}
				return nil
			},
		},
		{
			name: "CSV with special characters",
			csvContent: `text,label
"Text with ""quotes"" and commas, here",positive
"Text with 
newlines",negative`,
			expectError: false,
			expectRows:  2,
			validate: func(rows []models.DataRow) error {
				expectedFirst := `Text with "quotes" and commas, here`
				if rows[0].Text != expectedFirst {
					t.Errorf("Expected '%s', got '%s'", expectedFirst, rows[0].Text)
				}
				return nil
			},
		},
		{
			name:        "empty CSV",
			csvContent:  `text,label`,
			expectError: true, // Should error with no data rows
		},
		{
			name: "malformed CSV - inconsistent columns",
			csvContent: `text,label
"First row",positive
"Second row"`,
			expectError: false,
			expectRows:  1, // Should skip malformed rows
			validate: func(rows []models.DataRow) error {
				if len(rows) != 1 {
					t.Errorf("Expected 1 valid row, got %d", len(rows))
				}
				return nil
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create temporary CSV file
			tmpDir := t.TempDir()
			csvFile := filepath.Join(tmpDir, "test.csv")

			err := os.WriteFile(csvFile, []byte(tt.csvContent), 0644)
			if err != nil {
				t.Fatalf("Failed to write test CSV file: %v", err)
			}

			// Load CSV
			rows, err := LoadData(csvFile)

			if tt.expectError {
				if err == nil {
					t.Errorf("Expected error but got none")
				}
				return
			}

			if err != nil {
				t.Errorf("Unexpected error: %v", err)
				return
			}

			if len(rows) != tt.expectRows {
				t.Errorf("Expected %d rows, got %d", tt.expectRows, len(rows))
			}

			// Run custom validation if provided
			if tt.validate != nil && len(rows) > 0 {
				if err := tt.validate(rows); err != nil {
					t.Errorf("Custom validation failed: %v", err)
				}
			}
		})
	}
}

func TestLoadJSON(t *testing.T) {
	tests := []struct {
		name        string
		jsonContent string
		expectError bool
		expectRows  int
		validate    func([]models.DataRow) error
	}{
		{
			name: "basic JSON array",
			jsonContent: `[
				{"text": "This is positive", "label": "positive"},
				{"text": "This is negative", "label": "negative"},
				{"text": "This is neutral", "label": "neutral"}
			]`,
			expectError: false,
			expectRows:  3,
			validate: func(rows []models.DataRow) error {
				if rows[0].Text != "This is positive" {
					t.Errorf("Expected first row text 'This is positive', got '%s'", rows[0].Text)
				}
				if rows[0].Data["label"] != "positive" {
					t.Errorf("Expected first row label 'positive', got '%v'", rows[0].Data["label"])
				}
				return nil
			},
		},
		{
			name: "JSON with content field",
			jsonContent: `[
				{"content": "Great product", "category": "positive"},
				{"content": "Poor service", "category": "negative"}
			]`,
			expectError: false,
			expectRows:  2,
			validate: func(rows []models.DataRow) error {
				if rows[0].Text != "Great product" {
					t.Errorf("Expected text from content field, got '%s'", rows[0].Text)
				}
				return nil
			},
		},
		{
			name: "JSON with nested objects",
			jsonContent: `[
				{
					"text": "Review text",
					"metadata": {
						"author": "user1",
						"date": "2024-01-01"
					},
					"rating": 5
				}
			]`,
			expectError: false,
			expectRows:  1,
			validate: func(rows []models.DataRow) error {
				if rows[0].Text != "Review text" {
					t.Errorf("Expected text 'Review text', got '%s'", rows[0].Text)
				}
				// Check nested object is preserved
				metadata, ok := rows[0].Data["metadata"].(map[string]interface{})
				if !ok {
					t.Error("Expected metadata to be preserved as map")
				} else if metadata["author"] != "user1" {
					t.Errorf("Expected nested author 'user1', got '%v'", metadata["author"])
				}
				return nil
			},
		},
		{
			name: "JSON with mixed data types",
			jsonContent: `[
				{
					"text": "Test text",
					"score": 4.5,
					"count": 100,
					"active": true,
					"tags": ["positive", "review"]
				}
			]`,
			expectError: false,
			expectRows:  1,
			validate: func(rows []models.DataRow) error {
				if score, ok := rows[0].Data["score"].(float64); !ok || score != 4.5 {
					t.Errorf("Expected score 4.5, got %v", rows[0].Data["score"])
				}
				if active, ok := rows[0].Data["active"].(bool); !ok || !active {
					t.Errorf("Expected active true, got %v", rows[0].Data["active"])
				}
				return nil
			},
		},
		{
			name:        "empty JSON array",
			jsonContent: `[]`,
			expectError: false,
			expectRows:  0,
		},
		{
			name:        "invalid JSON",
			jsonContent: `[{"text": "incomplete"`,
			expectError: true,
		},
		{
			name:        "JSON not an array",
			jsonContent: `{"text": "single object"}`,
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create temporary JSON file
			tmpDir := t.TempDir()
			jsonFile := filepath.Join(tmpDir, "test.json")

			err := os.WriteFile(jsonFile, []byte(tt.jsonContent), 0644)
			if err != nil {
				t.Fatalf("Failed to write test JSON file: %v", err)
			}

			// Load JSON
			rows, err := LoadData(jsonFile)

			if tt.expectError {
				if err == nil {
					t.Errorf("Expected error but got none")
				}
				return
			}

			if err != nil {
				t.Errorf("Unexpected error: %v", err)
				return
			}

			if len(rows) != tt.expectRows {
				t.Errorf("Expected %d rows, got %d", tt.expectRows, len(rows))
			}

			// Run custom validation if provided
			if tt.validate != nil && len(rows) > 0 {
				if err := tt.validate(rows); err != nil {
					t.Errorf("Custom validation failed: %v", err)
				}
			}
		})
	}
}

func TestLoadDataFormatDetection(t *testing.T) {
	tests := []struct {
		name        string
		filename    string
		content     string
		expectError bool
	}{
		{
			name:     "CSV file",
			filename: "test.csv",
			content: `text,label
"Test",positive`,
			expectError: false,
		},
		{
			name:        "JSON file",
			filename:    "test.json",
			content:     `[{"text": "Test", "label": "positive"}]`,
			expectError: false,
		},
		{
			name:        "Excel file - .xlsx",
			filename:    "test.xlsx",
			content:     "",   // Would need actual Excel content, but testing format detection
			expectError: true, // Will fail on content but format is detected
		},
		{
			name:        "Excel file - .xls",
			filename:    "test.xls",
			content:     "",
			expectError: true,
		},
		{
			name:        "Parquet file",
			filename:    "test.parquet",
			content:     "",
			expectError: true, // Will fail on content but format is detected
		},
		{
			name:        "unsupported format",
			filename:    "test.txt",
			content:     "some text",
			expectError: true,
		},
		{
			name:        "no extension",
			filename:    "test",
			content:     "some content",
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tmpDir := t.TempDir()
			filePath := filepath.Join(tmpDir, tt.filename)

			err := os.WriteFile(filePath, []byte(tt.content), 0644)
			if err != nil {
				t.Fatalf("Failed to write test file: %v", err)
			}

			_, err = LoadData(filePath)

			if tt.expectError {
				if err == nil {
					t.Errorf("Expected error but got none for file %s", tt.filename)
				}
			} else {
				if err != nil {
					t.Errorf("Unexpected error for file %s: %v", tt.filename, err)
				}
			}
		})
	}
}

func TestLoadNonExistentFile(t *testing.T) {
	_, err := LoadData("/non/existent/file.csv")
	if err == nil {
		t.Error("Expected error for non-existent file")
	}
}

func TestDataRowIndexing(t *testing.T) {
	csvContent := `text,label
"First",positive
"Second",negative
"Third",neutral`

	tmpDir := t.TempDir()
	csvFile := filepath.Join(tmpDir, "test.csv")

	err := os.WriteFile(csvFile, []byte(csvContent), 0644)
	if err != nil {
		t.Fatalf("Failed to write test CSV file: %v", err)
	}

	rows, err := LoadData(csvFile)
	if err != nil {
		t.Fatalf("Failed to load CSV: %v", err)
	}

	// Check that indexes are assigned correctly
	for i, row := range rows {
		if row.Index != i {
			t.Errorf("Expected row %d to have index %d, got %d", i, i, row.Index)
		}
	}
}

func TestDataIntegrity(t *testing.T) {
	// Test that all original data is preserved in the Data map
	jsonContent := `[
		{
			"text": "Test review",
			"original_id": "12345",
			"timestamp": "2024-01-01T10:00:00Z",
			"metadata": {
				"source": "website",
				"verified": true
			},
			"tags": ["product", "review"],
			"score": 4.5
		}
	]`

	tmpDir := t.TempDir()
	jsonFile := filepath.Join(tmpDir, "test.json")

	err := os.WriteFile(jsonFile, []byte(jsonContent), 0644)
	if err != nil {
		t.Fatalf("Failed to write test JSON file: %v", err)
	}

	rows, err := LoadData(jsonFile)
	if err != nil {
		t.Fatalf("Failed to load JSON: %v", err)
	}

	if len(rows) != 1 {
		t.Fatalf("Expected 1 row, got %d", len(rows))
	}

	row := rows[0]

	// Check that all fields are preserved
	expectedFields := []string{"text", "original_id", "timestamp", "metadata", "tags", "score"}
	for _, field := range expectedFields {
		if _, exists := row.Data[field]; !exists {
			t.Errorf("Expected field '%s' to be preserved in Data map", field)
		}
	}

	// Check specific values and types
	if row.Data["original_id"] != "12345" {
		t.Errorf("Expected original_id '12345', got %v", row.Data["original_id"])
	}

	if score, ok := row.Data["score"].(float64); !ok || score != 4.5 {
		t.Errorf("Expected score 4.5 as float64, got %v (%T)", row.Data["score"], row.Data["score"])
	}

	// Check nested object
	metadata, ok := row.Data["metadata"].(map[string]interface{})
	if !ok {
		t.Error("Expected metadata to be map[string]interface{}")
	} else {
		if verified, ok := metadata["verified"].(bool); !ok || !verified {
			t.Errorf("Expected metadata.verified to be true, got %v", metadata["verified"])
		}
	}

	// Check array
	tags, ok := row.Data["tags"].([]interface{})
	if !ok {
		t.Error("Expected tags to be []interface{}")
	} else if len(tags) != 2 {
		t.Errorf("Expected 2 tags, got %d", len(tags))
	}
}
