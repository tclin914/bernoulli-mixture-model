#ifndef EM_H
#define EM_H

#define K 50
#define N 50000
#define D 28 *28

void EM(int **train_images, int *train_labels, int **test_images, int *test_labels, double mu[K][D], double *pi, double z[N][K]);

void ExpectationStep(double z[N][K], double *pi, double mu[K][D], int **x);
double ExpectationSubStep(int n, int k, double *pi, double mu[K][D], int **x); 

void MaximizationStep(double z[N][K], double *pi, double mu[K][D], int **x);
double *Average(int m, int **x, double z[N][K]); 
double Nm(int m, double z[N][K]);

int GetCluster(double mu[K][D], int *image);
#endif
