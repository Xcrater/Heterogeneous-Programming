#Variables
CC = g++
CFLAGS = -Wall
LIBS = -L/usr/lib/x86_64-linux-gnu
INCLUDES = -I/usr/local/includes

#Recipes
all:
	$(CC) $(CFLAGS) $(LIBS) $(INCLUDES) stereo_disparity.cpp -o stereo lodepng.cpp  

clean:
	rm -f *o image


