# Simple Makefile for building pybind11 C++ extensions into src/
# Usage:
#   make         # build all modules
#   make clean   # remove build artifacts
#   make install # copy .so into python/ for imports

CXX      = g++
CXXFLAGS = -O3 -Wall -std=c++20 -fPIC
LDFLAGS  = -shared

PYTHON   = python3
PY_INCLUDES = $(shell $(PYTHON) -m pybind11 --includes)
PY_LDFLAGS  = $(shell $(PYTHON)-config --ldflags)
EXT_SUFFIX  = $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

SRC_DIR  = src
BUILD_DIR= build
PY_DIR   = python

MODULES  = wav_generator audio_processor battery_model

.PHONY: all clean install

all: $(MODULES)

$(MODULES):
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(PY_INCLUDES) -I$(SRC_DIR) -c $(SRC_DIR)/$@.cpp -o $(BUILD_DIR)/$@.o
	$(CXX) $(LDFLAGS) -o $(SRC_DIR)/$@$(EXT_SUFFIX) $(BUILD_DIR)/$@.o $(PY_LDFLAGS) -lsndfile

clean:
	rm -rf $(BUILD_DIR)/*
	rm -f $(SRC_DIR)/*$(EXT_SUFFIX)

install: all
	@mkdir -p $(PY_DIR)
	cp $(SRC_DIR)/*$(EXT_SUFFIX) $(PY_DIR)/
