PROJECT := SalientDetect

CONFIG_FILE := Makefile.config
include $(CONFIG_FILE)

BUILD_DIR_LINK := $(BUILD_DIR)

SRC_DIRS := $(shell find * -type d -exec bash -c "find {} -maxdepth 1 -name '*.cpp' | grep -q ." \; -print)

BIN_NAME := $(PROJECT)
OBJ_DIR := $(BUILD_DIR)
DEP_DIR := $(BUILD_DIR)
RM := rm -rf

#LIBRARIES :=opencv_core opencv_highgui opencv_imgproc

CXX_SRCS := $(shell find $(SRC_DIRS) -name '*.cpp')
CXX_OBJS := $(addprefix $(OBJ_DIR)/, $(CXX_SRCS:.cpp=.o))
CXX_DEPS := $(addprefix $(DEP_DIR)/, $(CXX_SRCS:.cpp=.d))

ALL_BUILD_DIRS := $(sort $(BUILD_DIR) $(addprefix $(BUILD_DIR)/, $(SRC_DIRS)))
	
ifeq ($(OPENCV_VERSION), 3)
	LIBRARIES += opencv_imgcodecs opencv_videoio
endif
LIBRARIES += pthread

UNAME := $(shell uname -s)
ifeq ($(UNAME), Linux)
	LINUX := 1
endif

ifeq ($(LINUX), 1)
	CXX ?= /usr/bin/g++
endif

COMMON_FLAGS += $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir))
LDFLAGS += $(foreach librarydir,$(LIBRARY_DIRS),-L$(librarydir))
LIBS += $(foreach library,$(LIBRARIES),-l$(library))
CXXFLAGS += -Wno-strict-aliasing -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -std=c++11
LINKFLAGS += -fopenmp -std=c++11

all: $(PROJECT)

#$(PROJECT):
#	g++ -o "$@" $(COMMON_FLAGS) $(CXX_SRCS) $(LDFLAGS) $(LIBS) -std=c++11

$(PROJECT): $(CXX_OBJS)
	@echo 'Building target: $@'
	@echo 'Invoking: GCC C++ Linker'
	$(CXX) $(LDFLAGS) -o "$@" $(CXX_OBJS) $(LIBS) -std=c++11
	@echo 'Finished building target: $@'
 	
#$(CXX_OBJS): $(CXX_SRCS)    #Wrong!
$(BUILD_DIR)/%.o: %.cpp | $(ALL_BUILD_DIRS)
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	$(CXX) $(COMMON_FLAGS) -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:.o=.d)" -MT"$(@:.o=.d)" -o "$@" "$<" -std=c++11
	@echo 'Finished building: $<'
	@echo ' '

# $(shell mkdir -p $(ALL_BUILD_DIRS))
$(ALL_BUILD_DIRS): | $(BUILD_DIR_LINK)
	@ mkdir -p "$@"

clean:
	$(RM) $(CXX_DEPS) $(CXX_OBJS) $(PROJECT)
	@echo ' '
	
-include $(CXX_DEPS)
