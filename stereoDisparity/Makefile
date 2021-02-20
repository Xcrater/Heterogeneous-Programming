#Variables
CC = g++
CFLAGS = -Wall
LIBS = -L/usr/lib/x86_64-linux-gnu
INCLUDES = -I/usr/local/includes

#Recipes
all:
	$(CC) $(CFLAGS) $(LIBS) $(INCLUDES) stereoDisparity.cpp -o stereo lodepng.cpp  

clean:
	rm -f *o image


