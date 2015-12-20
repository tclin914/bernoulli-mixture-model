All:main

CFLAGS=-std=gnu99

OBJ=loglikelihood.o EM.o utils.o timer.o main.o

CS =loglikelihood.cu EM.cu utils.cu timer.cu

main: main.cu $(CS)
	nvcc $(CS) $< -o $@ -lm

# main: main.c $(OBJ)
	# nvcc -arch=sm_20 $(OBJ) 

# main.o: main.c
	# gcc $(CFLAGS) $< -c -o $@

# loglikelihood.o:loglikelihood.c
	# gcc $(CFLAGS) $< -c -o $@ 

# EM.o:EM.cu
	# nvcc $< -c -o $@

# utils.o:utils.c
	# gcc $(CFLAGS) $< -c -o $@

# timer.o:timer.c
	# gcc $(CFLAGS) $< -c -o $@

clean:
	rm main
	rm $(OBJ)
