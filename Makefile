BUILD_DIR := build
TARGET := polyai
WASMTIME := wasmtime

ifndef WASI_SDK
$(error Please set the WASI_SDK environment variable, e.g. `export WASI_SDK=/path/to/wasi-sdk`)
endif

TOOLCHAIN_FILE := $(WASI_SDK)/share/cmake/wasi-sdk.cmake
CMAKE_FLAGS := -DCMAKE_TOOLCHAIN_FILE=$(TOOLCHAIN_FILE) -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5
CMAKE_BUILD_INTELLISENSE_FLAGS := -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_FLAGS="-DBT_NO_SIMD_OPERATOR_OVERLOADS=1" -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5

ifeq ($(OS),Windows_NT)
  SRC_FILES := $(shell dir /b /s src\*)
else
  SRC_FILES := $(shell find src -type f)
endif
EXECUTABLE := $(BUILD_DIR)/$(TARGET)

all: $(EXECUTABLE) run

$(BUILD_DIR)/build.ninja: CMakeLists.txt
	@echo "Configuring project using WASI SDK at: $(WASI_SDK)"
	cmake -S . -B $(BUILD_DIR) -G Ninja $(CMAKE_FLAGS)

$(EXECUTABLE): $(BUILD_DIR)/build.ninja $(SRC_FILES)
	@echo "Building $(TARGET)..."
	cmake --build $(BUILD_DIR)

run: $(EXECUTABLE)
	$(WASMTIME) $(EXECUTABLE)

clean:
	rm -rf $(BUILD_DIR)

build-intellisense:
	cmake -S . -B $(BUILD_DIR)/intellisense -G Ninja $(CMAKE_BUILD_INTELLISENSE_FLAGS)

check-intellisense: build-intellisense
	cmake --build $(BUILD_DIR)/intellisense

.PHONY: all run clean build-intellisense
