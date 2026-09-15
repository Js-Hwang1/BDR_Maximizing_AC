CXX ?= g++
CXXFLAGS ?= -std=c++20 -O2 -march=native -Wall -Wextra -Isrc
OMPFLAGS ?= -fopenmp
LDFLAGS ?=
LDLIBS ?=

BUILD := build
LIBSRC := $(wildcard src/*.cpp) \
          $(wildcard src/methods/*.cpp) \
          $(wildcard src/methods/detail/*.cpp)
HDRS := $(wildcard src/*.hpp) \
        $(wildcard src/methods/*.hpp) \
        $(wildcard src/methods/detail/*.hpp)
LIBOBJ := $(patsubst src/%.cpp,$(BUILD)/%.o,$(LIBSRC))
APPSRC := $(wildcard src/app/*.cpp)
APPS := $(patsubst src/app/%.cpp,$(BUILD)/%,$(APPSRC))

.DEFAULT_GOAL := all
.DELETE_ON_ERROR:
.PHONY: all apps test clean

all: apps $(BUILD)/tests

apps: $(APPS)

$(BUILD)/%.o: src/%.cpp $(HDRS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -c $< -o $@

$(BUILD)/tests: $(LIBOBJ) tests/test_all.cpp $(HDRS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) tests/test_all.cpp $(LIBOBJ) \
		$(LDFLAGS) $(OMPFLAGS) $(LDLIBS) -o $@

$(BUILD)/%: src/app/%.cpp $(LIBOBJ) $(HDRS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< $(LIBOBJ) \
		$(LDFLAGS) $(OMPFLAGS) $(LDLIBS) -o $@

test: $(BUILD)/tests
	OMP_NUM_THREADS=1 ./$(BUILD)/tests

clean:
	rm -rf $(BUILD)

