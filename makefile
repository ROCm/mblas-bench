TARGET_EXEC ?= mblas-bench

BUILD_DIR ?= ./build
SRC_DIRS ?= src

#CUDA_ARCH_FLAGS ?= -arch=compute_70
#CC_FLAGS += -lcublas
# CC_FLAGS += -lcurand
# CC_FLAGS += -Xptxas
# CC_FLAGS += -v
# CC_FLAGS += -O3

SRCS := $(shell find $(SRC_DIRS) -name "*.cpp" -and ! -name "cutest.cpp" -or -name "*.cu")
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)
DEPS := $(OBJS:.o=.d)

INC_DIRS := $(shell find $(SRC_DIRS) -type d)
INC_FLAGS := $(addprefix -I,$(INC_DIRS))
LDFLAGS= -lcublas -lcublasLt


CPPFLAGS ?= $(INC_FLAGS) -MMD -MP -lcublas -arch=sm_80 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_90,code=sm_90 -O3
#CPPFLAGS ?= $(INC_FLAGS) -MMD -MP -lcublas -arch=sm_90 
CXXFLAGS += --std=c++14
CXX= nvcc
CC= nvcc
$(BUILD_DIR)/$(TARGET_EXEC): $(OBJS)
	$(CC) $(OBJS) -o $@ $(LDFLAGS)




## assembly
#$(BUILD_DIR)/%.s.o: %.s
#	$(MKDIR_P) $(dir $@)
#	$(AS) $(ASFLAGS) -c $< -o $@
#
## c source
#$(BUILD_DIR)/%.c.o: %.c
#	$(MKDIR_P) $(dir $@)
#	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@
#

# cu source
$(BUILD_DIR)/%.cu.o: %.cu
	$(MKDIR_P) $(dir $@)
	$(CC) $(CPPFLAGS) -c $< -o $@

# c++ source
$(BUILD_DIR)/%.cpp.o: %.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@


.PHONY: clean

clean:
	$(RM) -r $(BUILD_DIR)

-include $(DEPS)

MKDIR_P ?= mkdir -p
