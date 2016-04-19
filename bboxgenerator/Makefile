all: cut dt_box


cut:
	g++ -g -O3 gc.cpp -o cut `pkg-config --libs opencv` -lm

dt_box:
	g++ -g -O3 dt.c dt_box.cpp -o dt_box `pkg-config --libs opencv` -lm

.PHONY: clean
clean:
	rm cut dt_box
