#Copyright (C) 2018 Andreas Mayr, Guenter Klambauer
#Licensed under GNU General Public License v3.0 (see https://github.com/ml-jku/PlattScaling/blob/master/LICENSE)

plattScaling.so: plattScaling.o
	gcc -shared -o $(target)/platt/libPlatt.so $(target)/platt/platt.o

plattScaling.o: plattScaling.c plattScaling.h
	gcc -c -Wall -Werror -fpic plattScaling.c -std=c99 -o $(target)/platt/platt.o

clean:
	rm $(target)/platt/platt.o
	rm $(target)/platt/libPlatt.so
