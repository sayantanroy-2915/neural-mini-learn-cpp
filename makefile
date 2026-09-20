CC = g++
INCLUDE = -Iinclude
SOURCES = $(wildcard src/*.cpp)
TARGET = test.exe

$(TARGET) : $(SOURCES)
	$(CC) $(INCLUDE) $(SOURCES) -o $(TARGET)

clean:
	del $(TARGET)

run: $(TARGET)
	$(TARGET)

.PHONY: clean run