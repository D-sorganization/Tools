CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2

SRCS = test_dcs.cpp SignalBroker.cpp PIDController.cpp SafetyInterlock.cpp StorageManager.cpp
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
