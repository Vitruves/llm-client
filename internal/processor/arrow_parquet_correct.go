package processor

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	"github.com/apache/arrow/go/v14/arrow"
	"github.com/apache/arrow/go/v14/arrow/array"
	"github.com/apache/arrow/go/v14/arrow/memory"
	"github.com/apache/arrow/go/v14/parquet"
	"github.com/apache/arrow/go/v14/parquet/file"
	"github.com/apache/arrow/go/v14/parquet/pqarrow"
	"github.com/Vitruves/llm-client/internal/models"
)

// saveAsArrowParquet saves results using Apache Arrow with proper API usage
func (p *Processor) saveAsArrowParquet(results []models.Result, timestamp string) error {
	if len(results) == 0 {
		return fmt.Errorf("no results to save")
	}

	filename := filepath.Join(p.config.Output.Directory, fmt.Sprintf("results_%s.parquet", timestamp))

	// Check if we have parquet input to preserve schema
	if p.inputFile != "" && filepath.Ext(p.inputFile) == ".parquet" {
		return p.combineWithOriginalParquetCorrect(results, filename)
	}
	
	// Create new parquet file
	return p.createNewParquetCorrect(results, filename)
}

// combineWithOriginalParquetCorrect reads original parquet and adds new columns
func (p *Processor) combineWithOriginalParquetCorrect(results []models.Result, outputFilename string) error {
	// Read original parquet file
	inputFile, err := os.Open(p.inputFile)
	if err != nil {
		return fmt.Errorf("failed to open input parquet: %w", err)
	}
	defer inputFile.Close()

	// Create parquet reader
	reader, err := file.NewParquetReader(inputFile)
	if err != nil {
		return fmt.Errorf("failed to create parquet reader: %w", err)
	}
	defer reader.Close()

	// Create Arrow reader
	arrowReader, err := pqarrow.NewFileReader(reader, pqarrow.ArrowReadProperties{}, memory.DefaultAllocator)
	if err != nil {
		return fmt.Errorf("failed to create arrow reader: %w", err)
	}

	// Read the entire table
	table, err := arrowReader.ReadTable(context.Background())
	if err != nil {
		return fmt.Errorf("failed to read table: %w", err)
	}
	defer table.Release()

	// Find the maximum result index to determine actual data range
	maxIndex := -1
	for _, result := range results {
		if result.Index > maxIndex {
			maxIndex = result.Index
		}
	}
	
	// Create a new table with only the rows that match the actual loaded data
	// This handles cases where parquet-go and arrow see different row counts
	actualRows := maxIndex + 1
	if actualRows < int(table.NumRows()) {
		
		// Create sliced columns
		slicedColumns := make([]arrow.Column, table.NumCols())
		for i := 0; i < int(table.NumCols()); i++ {
			col := table.Column(i)
			chunked := col.Data()
			
			// Create new chunked array with only the needed rows
			var newChunks []arrow.Array
			remainingRows := int64(actualRows)
			
			for j := 0; j < chunked.Len() && remainingRows > 0; j++ {
				chunk := chunked.Chunk(j)
				if int64(chunk.Len()) <= remainingRows {
					newChunks = append(newChunks, chunk)
					remainingRows -= int64(chunk.Len())
				} else {
					// Need to slice this chunk
					slicedChunk := array.NewSlice(chunk, 0, remainingRows)
					newChunks = append(newChunks, slicedChunk)
					remainingRows = 0
				}
			}
			
			newChunked := arrow.NewChunked(chunked.DataType(), newChunks)
			slicedColumns[i] = *arrow.NewColumn(col.Field(), newChunked)
		}
		
		// Create new table with sliced columns
		table.Release()
		table = array.NewTable(table.Schema(), slicedColumns, int64(actualRows))
	}

	// Get original schema
	originalSchema := table.Schema()
	
	// Create result index map
	resultMap := make(map[int]models.Result)
	for _, result := range results {
		resultMap[result.Index] = result
	}

	// Create new fields for processing results
	newFields := []arrow.Field{
		{Name: "raw_response", Type: arrow.BinaryTypes.String, Nullable: true},
		{Name: "response_time", Type: arrow.PrimitiveTypes.Int64, Nullable: false},
		{Name: "inference_success", Type: arrow.FixedWidthTypes.Boolean, Nullable: false},
		{Name: "final_answer", Type: arrow.BinaryTypes.String, Nullable: true},
	}

	// Add thinking content field if needed
	if p.config.Output.IncludeThinking {
		newFields = append(newFields, arrow.Field{Name: "thinking_content", Type: arrow.BinaryTypes.String, Nullable: true})
	}

	// Add ground truth field if needed
	if p.config.Processing.LiveMetrics != nil && p.config.Processing.LiveMetrics.GroundTruth != "" {
		newFields = append(newFields, arrow.Field{Name: "ground_truth", Type: arrow.BinaryTypes.String, Nullable: true})
	}

	// Combine original and new fields
	allFields := make([]arrow.Field, 0, len(originalSchema.Fields())+len(newFields))
	allFields = append(allFields, originalSchema.Fields()...)
	allFields = append(allFields, newFields...)

	// Create new schema
	newSchema := arrow.NewSchema(allFields, nil)

	// Create memory allocator
	mem := memory.DefaultAllocator

	// Create builders for new columns
	rawResponseBuilder := array.NewStringBuilder(mem)
	responseTimeBuilder := array.NewInt64Builder(mem)
	inferenceSuccessBuilder := array.NewBooleanBuilder(mem)
	finalAnswerBuilder := array.NewStringBuilder(mem)

	var thinkingContentBuilder *array.StringBuilder
	var groundTruthBuilder *array.StringBuilder

	if p.config.Output.IncludeThinking {
		thinkingContentBuilder = array.NewStringBuilder(mem)
	}

	if p.config.Processing.LiveMetrics != nil && p.config.Processing.LiveMetrics.GroundTruth != "" {
		groundTruthBuilder = array.NewStringBuilder(mem)
	}

	// Build new column data
	numRows := int(table.NumRows())
	
	// Debug logging removed - issue resolved
	
	// Now process all rows - table has already been sliced to match actual data
	for i := 0; i < numRows; i++ {
		if result, exists := resultMap[i]; exists {
			rawResponseBuilder.Append(result.RawResponse)
			responseTimeBuilder.Append(result.ResponseTime.Nanoseconds() / 1000000)
			inferenceSuccessBuilder.Append(result.Success)
			finalAnswerBuilder.Append(result.FinalAnswer)

			if thinkingContentBuilder != nil {
				thinkingContentBuilder.Append(result.ThinkingContent)
			}

			if groundTruthBuilder != nil {
				groundTruthBuilder.Append(result.GroundTruth)
			}
		} else {
			// No result for this row - should not happen with sliced table
			rawResponseBuilder.AppendNull()
			responseTimeBuilder.Append(0)
			inferenceSuccessBuilder.Append(false)
			finalAnswerBuilder.AppendNull()

			if thinkingContentBuilder != nil {
				thinkingContentBuilder.AppendNull()
			}

			if groundTruthBuilder != nil {
				groundTruthBuilder.AppendNull()
			}
		}
	}

	// Create new arrays
	rawResponseArray := rawResponseBuilder.NewArray()
	responseTimeArray := responseTimeBuilder.NewArray()
	inferenceSuccessArray := inferenceSuccessBuilder.NewArray()
	finalAnswerArray := finalAnswerBuilder.NewArray()

	defer rawResponseArray.Release()
	defer responseTimeArray.Release()
	defer inferenceSuccessArray.Release()
	defer finalAnswerArray.Release()

	// Create new chunked arrays
	newChunkedArrays := []*arrow.Chunked{
		arrow.NewChunked(rawResponseArray.DataType(), []arrow.Array{rawResponseArray}),
		arrow.NewChunked(responseTimeArray.DataType(), []arrow.Array{responseTimeArray}),
		arrow.NewChunked(inferenceSuccessArray.DataType(), []arrow.Array{inferenceSuccessArray}),
		arrow.NewChunked(finalAnswerArray.DataType(), []arrow.Array{finalAnswerArray}),
	}

	// Add thinking content if needed
	if thinkingContentBuilder != nil {
		thinkingContentArray := thinkingContentBuilder.NewArray()
		defer thinkingContentArray.Release()
		newChunkedArrays = append(newChunkedArrays, 
			arrow.NewChunked(thinkingContentArray.DataType(), []arrow.Array{thinkingContentArray}))
	}

	// Add ground truth if needed
	if groundTruthBuilder != nil {
		groundTruthArray := groundTruthBuilder.NewArray()
		defer groundTruthArray.Release()
		newChunkedArrays = append(newChunkedArrays, 
			arrow.NewChunked(groundTruthArray.DataType(), []arrow.Array{groundTruthArray}))
	}

	// Get original chunked arrays
	originalChunkedArrays := make([]*arrow.Chunked, table.NumCols())
	for i := 0; i < int(table.NumCols()); i++ {
		originalChunkedArrays[i] = table.Column(i).Data()
	}

	// Combine all chunked arrays
	allChunkedArrays := append(originalChunkedArrays, newChunkedArrays...)

	// Convert to columns
	columns := make([]arrow.Column, len(allChunkedArrays))
	for i, chunked := range allChunkedArrays {
		columns[i] = *arrow.NewColumn(newSchema.Field(i), chunked)
	}

	// Create new table
	newTable := array.NewTable(newSchema, columns, -1)
	defer newTable.Release()

	// Write to parquet file
	return p.writeTableToParquet(newTable, outputFilename)
}

// createNewParquetCorrect creates a new parquet file with results
func (p *Processor) createNewParquetCorrect(results []models.Result, filename string) error {
	// Create schema
	fields := []arrow.Field{
		{Name: "index", Type: arrow.PrimitiveTypes.Int32, Nullable: false},
		{Name: "raw_response", Type: arrow.BinaryTypes.String, Nullable: true},
		{Name: "response_time", Type: arrow.PrimitiveTypes.Int64, Nullable: false},
		{Name: "inference_success", Type: arrow.FixedWidthTypes.Boolean, Nullable: false},
		{Name: "final_answer", Type: arrow.BinaryTypes.String, Nullable: true},
	}

	// Add thinking content field if needed
	if p.config.Output.IncludeThinking {
		fields = append(fields, arrow.Field{Name: "thinking_content", Type: arrow.BinaryTypes.String, Nullable: true})
	}

	// Add ground truth field if needed
	if p.config.Processing.LiveMetrics != nil && p.config.Processing.LiveMetrics.GroundTruth != "" {
		fields = append(fields, arrow.Field{Name: "ground_truth", Type: arrow.BinaryTypes.String, Nullable: true})
	}

	// Add original data fields dynamically
	originalFieldsMap := make(map[string]bool)
	for _, result := range results {
		for k := range result.OriginalData {
			originalFieldsMap[k] = true
		}
	}

	// Add original fields to schema
	for fieldName := range originalFieldsMap {
		fields = append(fields, arrow.Field{Name: fieldName, Type: arrow.BinaryTypes.String, Nullable: true})
	}

	schema := arrow.NewSchema(fields, nil)

	// Create memory allocator
	mem := memory.DefaultAllocator

	// Create builders
	indexBuilder := array.NewInt32Builder(mem)
	rawResponseBuilder := array.NewStringBuilder(mem)
	responseTimeBuilder := array.NewInt64Builder(mem)
	inferenceSuccessBuilder := array.NewBooleanBuilder(mem)
	finalAnswerBuilder := array.NewStringBuilder(mem)

	var thinkingContentBuilder *array.StringBuilder
	var groundTruthBuilder *array.StringBuilder

	if p.config.Output.IncludeThinking {
		thinkingContentBuilder = array.NewStringBuilder(mem)
	}

	if p.config.Processing.LiveMetrics != nil && p.config.Processing.LiveMetrics.GroundTruth != "" {
		groundTruthBuilder = array.NewStringBuilder(mem)
	}

	// Create builders for original data fields
	originalBuilders := make(map[string]*array.StringBuilder)
	for fieldName := range originalFieldsMap {
		originalBuilders[fieldName] = array.NewStringBuilder(mem)
	}

	// Build arrays
	for _, result := range results {
		indexBuilder.Append(int32(result.Index))
		rawResponseBuilder.Append(result.RawResponse)
		responseTimeBuilder.Append(result.ResponseTime.Nanoseconds() / 1000000)
		inferenceSuccessBuilder.Append(result.Success)
		finalAnswerBuilder.Append(result.FinalAnswer)

		if thinkingContentBuilder != nil {
			thinkingContentBuilder.Append(result.ThinkingContent)
		}

		if groundTruthBuilder != nil {
			groundTruthBuilder.Append(result.GroundTruth)
		}

		// Add original data values
		for fieldName, builder := range originalBuilders {
			if val, exists := result.OriginalData[fieldName]; exists {
				builder.Append(fmt.Sprintf("%v", val))
			} else {
				builder.AppendNull()
			}
		}
	}

	// Create arrays
	indexArray := indexBuilder.NewArray()
	rawResponseArray := rawResponseBuilder.NewArray()
	responseTimeArray := responseTimeBuilder.NewArray()
	inferenceSuccessArray := inferenceSuccessBuilder.NewArray()
	finalAnswerArray := finalAnswerBuilder.NewArray()

	defer indexArray.Release()
	defer rawResponseArray.Release()
	defer responseTimeArray.Release()
	defer inferenceSuccessArray.Release()
	defer finalAnswerArray.Release()

	// Create chunked arrays
	chunkedArrays := []*arrow.Chunked{
		arrow.NewChunked(indexArray.DataType(), []arrow.Array{indexArray}),
		arrow.NewChunked(rawResponseArray.DataType(), []arrow.Array{rawResponseArray}),
		arrow.NewChunked(responseTimeArray.DataType(), []arrow.Array{responseTimeArray}),
		arrow.NewChunked(inferenceSuccessArray.DataType(), []arrow.Array{inferenceSuccessArray}),
		arrow.NewChunked(finalAnswerArray.DataType(), []arrow.Array{finalAnswerArray}),
	}

	// Add thinking content if needed
	if thinkingContentBuilder != nil {
		thinkingContentArray := thinkingContentBuilder.NewArray()
		defer thinkingContentArray.Release()
		chunkedArrays = append(chunkedArrays, 
			arrow.NewChunked(thinkingContentArray.DataType(), []arrow.Array{thinkingContentArray}))
	}

	// Add ground truth if needed
	if groundTruthBuilder != nil {
		groundTruthArray := groundTruthBuilder.NewArray()
		defer groundTruthArray.Release()
		chunkedArrays = append(chunkedArrays, 
			arrow.NewChunked(groundTruthArray.DataType(), []arrow.Array{groundTruthArray}))
	}

	// Add original data arrays
	for _, builder := range originalBuilders {
		arr := builder.NewArray()
		defer arr.Release()
		chunkedArrays = append(chunkedArrays, 
			arrow.NewChunked(arr.DataType(), []arrow.Array{arr}))
	}

	// Convert to columns
	columns := make([]arrow.Column, len(chunkedArrays))
	for i, chunked := range chunkedArrays {
		columns[i] = *arrow.NewColumn(schema.Field(i), chunked)
	}

	// Create table
	table := array.NewTable(schema, columns, -1)
	defer table.Release()

	// Write to parquet file
	return p.writeTableToParquet(table, filename)
}

// writeTableToParquet writes an Arrow table to a parquet file
func (p *Processor) writeTableToParquet(table arrow.Table, filename string) error {
	outputFile, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create output file: %w", err)
	}
	defer outputFile.Close()

	// Create parquet writer properties
	props := parquet.NewWriterProperties()
	arrowProps := pqarrow.ArrowWriterProperties{}

	// Create parquet writer
	writer, err := pqarrow.NewFileWriter(table.Schema(), outputFile, props, arrowProps)
	if err != nil {
		return fmt.Errorf("failed to create parquet writer: %w", err)
	}
	defer writer.Close()

	// Write table
	err = writer.WriteTable(table, table.NumRows())
	if err != nil {
		return fmt.Errorf("failed to write table: %w", err)
	}

	return nil
}