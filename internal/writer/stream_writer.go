package writer

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strconv"
	"sync"

	"llm-client/internal/models"

	"github.com/parquet-go/parquet-go"
)

type StreamWriter interface {
	WriteResult(result models.Result) error
	Close() error
	GetFilename() string
	Flush() error
}

type JSONStreamWriter struct {
	filename   string
	file       *os.File
	encoder    *json.Encoder
	mutex      sync.Mutex
	firstWrite bool
}

type CSVStreamWriter struct {
	filename        string
	file            *os.File
	writer          *csv.Writer
	mutex           sync.Mutex
	headerWritten   bool
	config          *models.OutputConfig
	knownHeaders    []string
	originalHeaders map[string]struct{}
}

type ParquetStreamWriter struct {
	filename   string
	file       *os.File
	mutex      sync.Mutex
	config     *models.OutputConfig
	buffer     []StreamParquetResult
	bufferSize int
}

func NewStreamWriter(format, directory, timestamp string, config *models.OutputConfig) (StreamWriter, error) {
	if err := os.MkdirAll(directory, 0755); err != nil {
		return nil, err
	}

	switch format {
	case "json":
		return NewJSONStreamWriter(directory, timestamp)
	case "csv":
		return NewCSVStreamWriter(directory, timestamp, config)
	case "parquet":
		return NewParquetStreamWriter(directory, timestamp, config)
	default:
		return nil, fmt.Errorf("unsupported streaming format: %s", format)
	}
}

func NewJSONStreamWriter(directory, timestamp string) (*JSONStreamWriter, error) {
	filename := fmt.Sprintf("%s/results_%s.json", directory, timestamp)
	file, err := os.Create(filename)
	if err != nil {
		return nil, err
	}

	// Write opening bracket for JSON array
	if _, err := file.WriteString("[\n"); err != nil {
		file.Close()
		return nil, err
	}

	encoder := json.NewEncoder(file)
	encoder.SetIndent("  ", "  ")

	return &JSONStreamWriter{
		filename:   filename,
		file:       file,
		encoder:    encoder,
		firstWrite: true,
	}, nil
}

func (w *JSONStreamWriter) WriteResult(result models.Result) error {
	w.mutex.Lock()
	defer w.mutex.Unlock()

	if !w.firstWrite {
		if _, err := w.file.WriteString(",\n"); err != nil {
			return err
		}
	} else {
		w.firstWrite = false
	}

	// Write with indentation
	if _, err := w.file.WriteString("  "); err != nil {
		return err
	}

	return w.encoder.Encode(result)
}

func (w *JSONStreamWriter) Close() error {
	w.mutex.Lock()
	defer w.mutex.Unlock()

	if w.file != nil {
		// Write closing bracket
		if _, err := w.file.WriteString("\n]"); err != nil {
			w.file.Close()
			return err
		}
		return w.file.Close()
	}
	return nil
}

func (w *JSONStreamWriter) GetFilename() string {
	return w.filename
}

func (w *JSONStreamWriter) Flush() error {
	w.mutex.Lock()
	defer w.mutex.Unlock()

	if w.file != nil {
		return w.file.Sync()
	}
	return nil
}

func NewCSVStreamWriter(directory, timestamp string, config *models.OutputConfig) (*CSVStreamWriter, error) {
	filename := fmt.Sprintf("%s/results_%s.csv", directory, timestamp)
	file, err := os.Create(filename)
	if err != nil {
		return nil, err
	}

	writer := csv.NewWriter(file)

	return &CSVStreamWriter{
		filename:        filename,
		file:            file,
		writer:          writer,
		config:          config,
		originalHeaders: make(map[string]struct{}),
	}, nil
}

func (w *CSVStreamWriter) WriteResult(result models.Result) error {
	w.mutex.Lock()
	defer w.mutex.Unlock()

	// Collect headers from this result
	for key := range result.OriginalData {
		w.originalHeaders[key] = struct{}{}
	}

	// Write header if this is the first result
	if !w.headerWritten {
		w.writeHeader()
		w.headerWritten = true
	}

	// Write data row
	row := []string{
		strconv.Itoa(result.Index),
		result.InputText,
		result.GroundTruth,
		result.FinalAnswer,
		strconv.FormatBool(result.Success),
		strconv.FormatInt(result.ResponseTime.Nanoseconds()/1000000, 10),
	}

	// Add original data values in header order
	for _, h := range w.knownHeaders {
		val := result.OriginalData[h]
		row = append(row, fmt.Sprintf("%v", val))
	}

	if w.config.IncludeThinking {
		row = append(row, result.ThinkingContent)
	}
	if w.config.IncludeRawResponse {
		row = append(row, result.RawResponse)
	}

	if err := w.writer.Write(row); err != nil {
		return err
	}

	w.writer.Flush()
	return w.writer.Error()
}

func (w *CSVStreamWriter) writeHeader() {
	// Sort original headers for consistent output
	var sortedHeaders []string
	for key := range w.originalHeaders {
		sortedHeaders = append(sortedHeaders, key)
	}
	sort.Strings(sortedHeaders)
	w.knownHeaders = sortedHeaders

	header := []string{"index", "input_text", "ground_truth", "final_answer", "success", "response_time_ms"}
	header = append(header, w.knownHeaders...)

	if w.config.IncludeThinking {
		header = append(header, "thinking_content")
	}
	if w.config.IncludeRawResponse {
		header = append(header, "raw_response")
	}

	w.writer.Write(header)
	w.writer.Flush()
}

func (w *CSVStreamWriter) Close() error {
	w.mutex.Lock()
	defer w.mutex.Unlock()

	if w.writer != nil {
		w.writer.Flush()
	}
	if w.file != nil {
		return w.file.Close()
	}
	return nil
}

func (w *CSVStreamWriter) GetFilename() string {
	return w.filename
}

func (w *CSVStreamWriter) Flush() error {
	w.mutex.Lock()
	defer w.mutex.Unlock()

	if w.writer != nil {
		w.writer.Flush()
		return w.writer.Error()
	}
	return nil
}

// StreamParquetResult represents the structure for parquet streaming output
// We serialize OriginalData as JSON to avoid interface{} schema issues
type StreamParquetResult struct {
	Index            int32  `parquet:"index"`
	InputText        string `parquet:"input_text"`
	GroundTruth      string `parquet:"ground_truth"`
	FinalAnswer      string `parquet:"final_answer"`
	Success          bool   `parquet:"success"`
	ResponseTimeMs   int64  `parquet:"response_time_ms"`
	ThinkingContent  string `parquet:"thinking_content,optional"`
	RawResponse      string `parquet:"raw_response,optional"`
	OriginalDataJSON string `parquet:"original_data_json,optional"`
}

func NewParquetStreamWriter(directory, timestamp string, config *models.OutputConfig) (*ParquetStreamWriter, error) {
	filename := fmt.Sprintf("%s/results_%s.parquet", directory, timestamp)
	file, err := os.Create(filename)
	if err != nil {
		return nil, err
	}

	// Use a reasonable buffer size - flush every 100 results by default
	bufferSize := config.StreamSaveEvery
	if bufferSize <= 0 {
		bufferSize = 100
	}

	return &ParquetStreamWriter{
		filename:   filename,
		file:       file,
		config:     config,
		buffer:     make([]StreamParquetResult, 0, bufferSize),
		bufferSize: bufferSize,
	}, nil
}

func (w *ParquetStreamWriter) WriteResult(result models.Result) error {
	w.mutex.Lock()
	defer w.mutex.Unlock()

	// Convert result to parquet format
	// Serialize OriginalData to JSON to avoid interface{} schema issues
	var originalDataJSON string
	if result.OriginalData != nil {
		if jsonBytes, err := json.Marshal(result.OriginalData); err == nil {
			originalDataJSON = string(jsonBytes)
		}
	}

	parquetResult := StreamParquetResult{
		Index:            int32(result.Index),
		InputText:        result.InputText,
		GroundTruth:      result.GroundTruth,
		FinalAnswer:      result.FinalAnswer,
		Success:          result.Success,
		ResponseTimeMs:   result.ResponseTime.Nanoseconds() / 1000000,
		ThinkingContent:  result.ThinkingContent,
		RawResponse:      result.RawResponse,
		OriginalDataJSON: originalDataJSON,
	}

	// Add to buffer
	w.buffer = append(w.buffer, parquetResult)

	// Flush when buffer is full
	if len(w.buffer) >= w.bufferSize {
		return w.flushBuffer()
	}

	return nil
}

func (w *ParquetStreamWriter) flushBuffer() error {
	if len(w.buffer) == 0 {
		return nil
	}

	// Write the buffered results to parquet file
	err := parquet.Write(w.file, w.buffer)
	if err != nil {
		return err
	}

	// Clear the buffer after successful write
	w.buffer = w.buffer[:0]
	return nil
}

func (w *ParquetStreamWriter) Close() error {
	w.mutex.Lock()
	defer w.mutex.Unlock()

	// Flush any remaining buffered results
	err := w.flushBuffer()
	if err != nil {
		if w.file != nil {
			w.file.Close()
		}
		return err
	}

	// Close the file
	if w.file != nil {
		return w.file.Close()
	}
	return nil
}

func (w *ParquetStreamWriter) GetFilename() string {
	return w.filename
}

func (w *ParquetStreamWriter) Flush() error {
	w.mutex.Lock()
	defer w.mutex.Unlock()

	// Flush any buffered results to the file
	err := w.flushBuffer()
	if err != nil {
		return err
	}

	// Sync the underlying file
	if w.file != nil {
		return w.file.Sync()
	}
	return nil
}
