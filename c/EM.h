#ifndef EM_H
#define EM_H

#define K 40
#define N 50000
#define D 28 *28

void EM(int **train_images, int *train_labels, int **test_images, int *test_labels, double mu[K][D], double *pi, double z[N][K]);

#endif
