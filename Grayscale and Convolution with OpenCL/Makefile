#Variables
CC = g++
CFLAGS = -Wall
LIBS = -L/usr/lib/x86_64-linux-gnu
INCLUDES = -I/usr/local/cuda/include/CL

#Recipes
all:
	$(CC) $(CFLAGS) $(LIBS) $(INCLUDES) image_filtering.cpp -o image lodepng.cpp -l OpenCL

compile:
	$(CC) $(CFLAGS) image_filtering.cpp -o image lodepng.cpp 

platform & device info:

clinfo:
	clinfo 

clean:
	rm -f *o image


