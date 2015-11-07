#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <stdio.h>

#define K 40
#define N 50000
#define D 28 * 28

void ExpectationStep(double *pi, double mu[K][D], int **x);
double ExpectationSubStep(int n, int k, double *pi, double mu[K][D], int **x); 

void MaximizationStep(double *pi, double mu[K][D], int **x);
double *Average(int m, int **x); 
double Nm(int m);

double z[N][K];

void EM(int **train_images, int *train_labels, int **test_images, int *test_labels) {
    
    double pi[K] = {1 / K};
    double mu[K][D] = {{0}};

    srand(time(NULL));
    /* normalization */
    int normalizationFactor;
    for (int w = 0; w < K; w++) {
       normalizationFactor = 0;
       for (int g = 0; g < D; g++) {
           mu[w][g] = rand();
           normalizationFactor = normalizationFactor + mu[w][g];
       }

       for (int g = 0; g < D; g++) {
           mu[w][g] = mu[w][g] / normalizationFactor;
       }
    }
    
    memset(z, 0, sizeof(z));

    int start = time(NULL);
    for (int i = 0; i < 3; i++) {
        ExpectationStep(pi, mu, train_images);
        MaximizationStep(pi, mu, train_images); 
    }
    int end = time(NULL);
    printf("time = %d\n", end - start);
}

void ExpectationStep(double *pi, double mu[K][D], int **x) {
    int normalizationFactor;
    for (int n = 0; n < N; n++) {
        normalizationFactor = 0;

        for (int k = 0; k < K; k++) {
            z[n][k] = ExpectationSubStep(n, k, pi, mu, x);
            normalizationFactor = normalizationFactor + z[n][k];
        }

        for (int k = 0; k < K; k++) {
            if (normalizationFactor > 0) {
                z[n][k] = z[n][k] / normalizationFactor;
            } else {
                z[n][k] = 1 / K;
            }
        }
    }
}

double ExpectationSubStep(int n, int k, double *pi, double mu[K][D], int **x) {
    double z_nk = pi[k];
    for (int i = 0; i < D; i++) {
        z_nk = z_nk * pow(mu[k][i], x[n][i]) * pow(1 - mu[k][i], 1 - x[n][i]);      
    }
    return z_nk;
}

void MaximizationStep(double *pi, double mu[K][D], int **x) {
    for (int k = 0; k < K; k++) {
        pi[k] = Nm(k) / N;
    } 
    double *average;
    for (int k = 0; k < K; k++) {
        average = Average(k, x);
        
        for (int i = 0; i < D; i++) {
            mu[k][i] = average[i];
        }
    }
    free(average);
}

double *Average(int m, int **x) {
    double *result = (double*)malloc(sizeof(double) * D);
    memset(result, 0, sizeof(double) * D);
    for (int i = 0; i < D; i++) {
        for (int n = 0; n < N; n++) {
            result[i] = result[i] + z[n][m] * x[n][i];
        }
    }
    double currentNm = Nm(m);
    for (int i = 0; i < D; i++) {
        result[i] = result[i] / currentNm;
    }
    return result;
}

double Nm(int m) {
    int result = 0;
    for (int n = 0; n < N; n++) {
        result = result + z[n][m];
    }
    return result;
}








