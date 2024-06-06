all: ciao

ciao: matrix.cpp
	g++-14 -fopenmp -o ciao matrix.cpp

run: ciao
	./ciao

run_and_clean: run clean

clean:
	rm -f ciao

