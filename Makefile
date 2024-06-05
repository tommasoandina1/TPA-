CC = g++-14
CFLAGS = -std=c++11 -fopenmp
TARGET = ciao
SRC = matrix.cpp

all: $(TARGET)

$(TARGET): $(SRC)
    $(CC) $(CFLAGS) $^ -o $@

clean:
    rm -f $(TARGET)