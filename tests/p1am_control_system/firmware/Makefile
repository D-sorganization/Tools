CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2
FIRMWARE_SRC_DIR = ../../../src/p1am_control_system/firmware
CXXFLAGS += -I$(FIRMWARE_SRC_DIR)
vpath %.cpp $(FIRMWARE_SRC_DIR)
vpath %.h $(FIRMWARE_SRC_DIR)

SRCS = test_dcs.cpp SignalBroker.cpp PIDController.cpp SafetyInterlock.cpp StorageManager.cpp CommsWatchdog.cpp
OBJS = $(SRCS:.cpp=.o)
TARGET = test_dcs

.PHONY: all test clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

test: all
	./$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET) config.bin
