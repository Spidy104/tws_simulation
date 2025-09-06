# Makefile — build pybind11 extensions and place .so into src/ for Python imports
PYBIND11_INCLUDES := $(shell python3 -m pybind11 --includes 2>/dev/null || echo "")
PY_LDFLAGS        := $(shell python3-config --ldflags 2>/dev/null || echo "")
EXT_SUFFIX        := $(shell python3 -c "import sysconfig as s; print(s.get_config_var('EXT_SUFFIX') or '.so')" 2>/dev/null)

CXX      ?= g++
CXXSTD   ?= -std=c++20
CXXFLAGS ?= -O3 -Wall -fPIC $(CXXSTD)
LDFLAGS  ?= -shared
LDLIBS   ?= $(PY_LDFLAGS)

SRC_DIR := src
BUILD_DIR := build
MODULES := wav_generator audio_processor battery_model

# object lists (one .o per module, placed in build/)
OBJS := $(patsubst %, $(BUILD_DIR)/%.o, $(MODULES))
SRCS := $(patsubst %, $(SRC_DIR)/%.cpp, $(MODULES))
HDRS := $(patsubst %, $(SRC_DIR)/%.hpp, $(MODULES))

.PHONY: all clean test dirs

all: dirs $(patsubst %, $(SRC_DIR)/%$(EXT_SUFFIX), $(MODULES))

dirs:
	@mkdir -p $(BUILD_DIR) outputs assets logs

# helpful message if pybind11 isn't available
check_pybind11:
ifndef PYBIND11_INCLUDES
	@echo "Warning: python3 -m pybind11 --includes failed. Ensure pybind11 is installed for headers to be found."
endif

# compile rules: compile .cpp to .o (depend on corresponding .hpp if present)
# If a header is missing we still compile, so use a pattern rule with optional header dependency
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp $(SRC_DIR)/%.hpp
	@echo "[CXX] $<"
	$(CXX) $(CXXFLAGS) $(PYBIND11_INCLUDES) -I$(SRC_DIR) -c $< -o $@

# fallback rule: if header missing, still compile
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo "[CXX] $< (no header dependency)"
	$(CXX) $(CXXFLAGS) $(PYBIND11_INCLUDES) -I$(SRC_DIR) -c $< -o $@

# link wav_generator with libsndfile (if needed)
$(SRC_DIR)/wav_generator$(EXT_SUFFIX): $(BUILD_DIR)/wav_generator.o
	@echo "[LINK] -> $@"
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS) -lsndfile

# link others (no extra libs by default)
$(SRC_DIR)/audio_processor$(EXT_SUFFIX): $(BUILD_DIR)/audio_processor.o
	@echo "[LINK] -> $@"
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(SRC_DIR)/battery_model$(EXT_SUFFIX): $(BUILD_DIR)/battery_model.o
	@echo "[LINK] -> $@"
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)

clean:
	@echo "Cleaning..."
	rm -rf $(BUILD_DIR)/*
	# remove generated Python extension files in src/ that match module names + suffix
	@for m in $(MODULES); do \
	  f="$(SRC_DIR)/$$m$(EXT_SUFFIX)"; \
	  if [ -f $$f ]; then echo "rm $$f"; rm -f $$f; fi; \
	done

# test: run python test (python/test_wav.py) with PYTHONPATH pointing at build and python/
test: all
	@PYTHONPATH=python:$(CURDIR) python3 python/test_wav.py

.PHONY: run-gui
run-gui: all
	@echo "Running GUI with PYTHONPATH=python:$(CURDIR)"
	@PYTHONPATH=python:$(CURDIR) python3 python/gui.py
