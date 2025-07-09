package loader

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/Vitruves/llm-client/internal/models"

	"github.com/parquet-go/parquet-go"
	"github.com/xuri/excelize/v2"
)

func LoadData(filename string) ([]models.DataRow, error) {
	ext := strings.ToLower(filepath.Ext(filename))

	switch ext {
	case ".csv":
		return loadCSV(filename)
	case ".json":
		return loadJSON(filename)
	case ".xlsx", ".xls":
		return loadExcel(filename)
	case ".parquet":
		return loadParquet(filename)
	default:
		return nil, fmt.Errorf("unsupported file format: %s", ext)
	}
}

func loadCSV(filename string) ([]models.DataRow, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)

	// Read header first
	headers, err := reader.Read()
	if err != nil {
		return nil, fmt.Errorf("failed to read CSV header: %w", err)
	}

	var data []models.DataRow
	filePosition := 0  // Track actual file position (including skipped rows)

	// Read rows one by one to handle malformed rows gracefully
	for {
		record, err := reader.Read()
		if err != nil {
			// End of file is expected
			if err == io.EOF {
				break
			}
			// Skip malformed rows and continue, but increment file position
			filePosition++
			continue
		}

		// Skip rows with incorrect number of columns, but increment file position
		if len(record) != len(headers) {
			filePosition++
			continue
		}

		row := models.DataRow{
			Index: filePosition,  // Use file position for consistent indexing
			Data:  make(map[string]interface{}),
		}

		for j, value := range record {
			row.Data[headers[j]] = value
			if headers[j] == "text" || headers[j] == "content" {
				row.Text = value
			}
		}

		if row.Text == "" && len(record) > 0 {
			row.Text = record[0]
		}

		data = append(data, row)
		filePosition++
	}

	if len(data) == 0 {
		return nil, fmt.Errorf("CSV file must have at least one valid data row")
	}

	return data, nil
}

func loadJSON(filename string) ([]models.DataRow, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var rawData []map[string]interface{}
	if err := json.NewDecoder(file).Decode(&rawData); err != nil {
		return nil, err
	}

	var data []models.DataRow
	for i, item := range rawData {
		row := models.DataRow{
			Index: i,
			Data:  item,
		}

		if text, ok := item["text"].(string); ok {
			row.Text = text
		} else if content, ok := item["content"].(string); ok {
			row.Text = content
		} else {
			for _, value := range item {
				if str, ok := value.(string); ok && len(str) > 10 {
					row.Text = str
					break
				}
			}
		}

		data = append(data, row)
	}

	return data, nil
}

func loadExcel(filename string) ([]models.DataRow, error) {
	f, err := excelize.OpenFile(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	sheets := f.GetSheetList()
	if len(sheets) == 0 {
		return nil, fmt.Errorf("no sheets found")
	}

	rows, err := f.GetRows(sheets[0])
	if err != nil {
		return nil, err
	}

	if len(rows) < 2 {
		return nil, fmt.Errorf("Excel file must have at least a header and one data row")
	}

	headers := rows[0]
	var data []models.DataRow
	filePosition := 0  // Track actual file position (including skipped rows)

	for _, row := range rows[1:] {
		if len(row) == 0 {
			filePosition++
			continue
		}

		dataRow := models.DataRow{
			Index: filePosition,  // Use file position for consistent indexing
			Data:  make(map[string]interface{}),
		}

		for j, value := range row {
			if j < len(headers) && headers[j] != "" {
				dataRow.Data[headers[j]] = value
				if headers[j] == "text" || headers[j] == "content" {
					dataRow.Text = value
				}
			}
		}

		if dataRow.Text == "" && len(row) > 0 {
			dataRow.Text = row[0]
		}

		data = append(data, dataRow)
		filePosition++
	}

	return data, nil
}

func loadParquet(filename string) ([]models.DataRow, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	// Get file info for size
	stat, err := file.Stat()
	if err != nil {
		return nil, err
	}

	// Open parquet file
	pf, err := parquet.OpenFile(file, stat.Size())
	if err != nil {
		return nil, err
	}

	// Get the schema to understand the structure
	schema := pf.Schema()
	columns := schema.Columns()

	// Read all row groups
	var data []models.DataRow
	rowIndex := 0
	
	for _, rowGroup := range pf.RowGroups() {
		rows := rowGroup.Rows()
		defer rows.Close()

		// Read all rows in the group at once
		totalRows := rowGroup.NumRows()
		allRows := make([]parquet.Row, totalRows)
		
		// Read all rows at once to avoid EOF issues
		n, err := rows.ReadRows(allRows)
		if err != nil && err.Error() != "EOF" {
			return nil, err
		}
		
		// Process each row that was successfully read
		for i := 0; i < n; i++ {
			row := allRows[i]

			// Convert parquet row to DataRow
			dataRow := models.DataRow{
				Index: rowIndex,
				Data:  make(map[string]interface{}),
			}

			// Extract values from each column in the row
			row.Range(func(columnIndex int, columnValues []parquet.Value) bool {
				if columnIndex < len(columns) {
					columnPath := columns[columnIndex]
					columnName := columnPath[len(columnPath)-1] // Get the leaf column name

					// Use the first value if there are multiple (for repeated fields)
					if len(columnValues) > 0 {
						value := columnValues[0]

						// Convert parquet value to Go value
						var goValue interface{}
						if value.IsNull() {
							goValue = nil
						} else {
							switch value.Kind() {
							case parquet.Boolean:
								goValue = value.Boolean()
							case parquet.Int32:
								goValue = value.Int32()
							case parquet.Int64:
								goValue = value.Int64()
							case parquet.Float:
								goValue = value.Float()
							case parquet.Double:
								goValue = value.Double()
							case parquet.ByteArray:
								goValue = string(value.ByteArray())
							case parquet.FixedLenByteArray:
								goValue = string(value.ByteArray())
							default:
								goValue = value.String()
							}
						}

						dataRow.Data[columnName] = goValue

						// Set Text field if we find common text field names
						if (columnName == "text" || columnName == "content" || columnName == "REVIEW") && goValue != nil {
							if str, ok := goValue.(string); ok {
								dataRow.Text = str
							}
						}
					}
				}
				return true // Continue iteration
			})

			// If no text field found, use first string field as fallback
			if dataRow.Text == "" {
				for _, value := range dataRow.Data {
					if str, ok := value.(string); ok && len(str) > 10 {
						dataRow.Text = str
						break
					}
				}
			}

			data = append(data, dataRow)
			rowIndex++
		}
	}
	

	if len(data) == 0 {
		return nil, fmt.Errorf("parquet file is empty")
	}

	return data, nil
}
