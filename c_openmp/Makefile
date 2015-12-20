All:main

CFLAGS=-std=c99

OBJ=loglikelihood.o EM.o utils.o timer.o 

main: main.c $(OBJ)
	gcc $(CFLAGS) -fopenmp $(OBJ) $< -o $@ -lm

loglikelihood.o:loglikelihood.c
	gcc $(CFLAGS) $< -c -o $@ 

EM.o:EM.c
	gcc $(CFLAGS) -fopenmp $< -c -o $@

utils.o:utils.c
	gcc $(CFLAGS) $< -c -o $@

timer.o:timer.c
	gcc $(CFLAGS) $< -c -o $@
