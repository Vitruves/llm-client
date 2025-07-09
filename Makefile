# Makefile for LLM Client

# Variables
BINARY_NAME=llm-client
BUILD_DIR=./bin
INSTALL_DIR=$(HOME)/.local/bin
MAIN_FILE=./main.go

# Default target
.PHONY: all
all: build

# Build the binary
.PHONY: build
build:
	@echo "Building $(BINARY_NAME)..."
	@mkdir -p $(BUILD_DIR)
	go build -o $(BUILD_DIR)/$(BINARY_NAME) $(MAIN_FILE)
	@echo "Build complete: $(BUILD_DIR)/$(BINARY_NAME)"

# Clean build artifacts
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(BUILD_DIR)
	@echo "Clean complete"

# Install binary to ~/.local/bin
.PHONY: install
install: build
	@echo "Installing $(BINARY_NAME) to $(INSTALL_DIR)..."
	@mkdir -p $(INSTALL_DIR)
	cp $(BUILD_DIR)/$(BINARY_NAME) $(INSTALL_DIR)/
	@echo "Installation complete: $(INSTALL_DIR)/$(BINARY_NAME)"

# Uninstall binary from ~/.local/bin
.PHONY: uninstall
uninstall:
	@echo "Uninstalling $(BINARY_NAME) from $(INSTALL_DIR)..."
	rm -f $(INSTALL_DIR)/$(BINARY_NAME)
	@echo "Uninstall complete"

# Run tests
.PHONY: test
test:
	go test ./...

# Show help
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  build     - Build the binary to $(BUILD_DIR)/"
	@echo "  clean     - Remove build artifacts"
	@echo "  install   - Build and install to $(INSTALL_DIR)"
	@echo "  uninstall - Remove from $(INSTALL_DIR)"
	@echo "  test      - Run tests"
	@echo "  help      - Show this help message"