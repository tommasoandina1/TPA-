CC = g++
CFLAGS = -std=c++11
TARGET = ciao
SRC = lol.cpp

all: $(TARGET)

$(TARGET): $(SRC)
    $(CC) $(CFLAGS) $^ -o $@

clean:
    rm -f $(TARGET)
