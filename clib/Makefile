CC      ?= cc
CFLAGS  ?= -Wall -Wextra -O2

INCLUDE_DIR = include
TEST_DIR    = tests
EXAMPLE_DIR = examples
BENCH_DIR   = bench
BUILD_DIR   = build

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# --------------------------
# Test
# --------------------------
TEST_SRCS := $(wildcard $(TEST_DIR)/*.c)
TEST_BINS := $(patsubst $(TEST_DIR)/%.c,$(BUILD_DIR)/%,$(TEST_SRCS))

.PHONY: test
test: $(TEST_BINS)
	@for t in $(TEST_BINS); do echo "Running $$t..."; $$t || exit 1; done

$(BUILD_DIR)/%: $(TEST_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) $< -o $@

# --------------------------
# Example
# --------------------------
EXAMPLE_SRCS := $(wildcard $(EXAMPLE_DIR)/*.c)
EXAMPLE_BINS := $(patsubst $(EXAMPLE_DIR)/%.c,$(BUILD_DIR)/%,$(EXAMPLE_SRCS))

.PHONY: example
example: $(EXAMPLE_BINS)
	@for e in $(EXAMPLE_BINS); do echo "Running $$e..."; $$e || exit 1; done

$(BUILD_DIR)/%: $(EXAMPLE_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) $< -o $@

# --------------------------
# Bench (single bench.c)
# --------------------------
BENCH_BIN := $(BUILD_DIR)/bench

.PHONY: bench
bench: $(BENCH_BIN)
	@echo "Running $(BENCH_BIN)..."
	@$(BENCH_BIN)

$(BENCH_BIN): $(BENCH_DIR)/bench.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) $< -o $@

# --------------------------
# Clean
# --------------------------
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
