BUILD_DIR := build
CONFIG ?= Release
SRC_DIR := src

ifeq ($(OS),Windows_NT)
    NPROC := 1
    FIND := dir /b /s
    JFLAG := /m
else
    NPROC := $(shell nproc)
    FIND := find
    JFLAG := -j$(NPROC)
endif

CMAKE_FILES := CMakeLists.txt
SRC_FILES := $(shell $(FIND) $(SRC_DIR) -type f -name "*.cpp" -o -name "*.c" -o -name "*.h" -o -name "*.hpp" 2>/dev/null)

.PHONY: all configure build clean

all: build

$(BUILD_DIR)/CMakeCache.txt: $(CMAKE_FILES)
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(CONFIG)

configure: $(BUILD_DIR)/CMakeCache.txt

build: configure $(SRC_FILES)
ifeq ($(OS),Windows_NT)
	cmake --build $(BUILD_DIR) --config $(CONFIG) -- $(JFLAG)
else
	cmake --build $(BUILD_DIR) --config $(CONFIG) -- $(JFLAG)
endif

clean:
	cmake -E rm -rf $(BUILD_DIR)
