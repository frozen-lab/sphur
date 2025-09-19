CC      ?= cc
CFLAGS  ?= -Wall -Wextra -O2

INCLUDE_DIR = include
TEST_DIR    = tests
EXAMPLE_DIR = examples
BUILD_DIR   = build

TEST_SRCS     := $(wildcard $(TEST_DIR)/*.c)
TEST_BINS     := $(patsubst $(TEST_DIR)/%.c,$(BUILD_DIR)/%,$(TEST_SRCS))

EXAMPLE_SRCS  := $(wildcard $(EXAMPLE_DIR)/*.c)
EXAMPLE_BINS  := $(patsubst $(EXAMPLE_DIR)/%.c,$(BUILD_DIR)/%,$(EXAMPLE_SRCS))

.PHONY: all clean tests examples run-tests

all: tests examples

tests: $(TEST_BINS)

examples: $(EXAMPLE_BINS)

$(BUILD_DIR)/%: $(TEST_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) $< -o $@

$(BUILD_DIR)/%: $(EXAMPLE_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

test: tests
	@for t in $(TEST_BINS); do echo "Running $$t..."; $$t || exit 1; done
	$(MAKE) clean

example: examples
	@for e in $(EXAMPLE_BINS); do echo ""; echo "Running $$e..."; $$e || exit 1; done
	$(MAKE) clean

clean:
	rm -rf $(BUILD_DIR)
