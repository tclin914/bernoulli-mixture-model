#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <float.h>
#include "EM.h"

#include <omp.h>

void EM(int **train_images, int *train_labels, int **test_images, int *test_labels, double mu[K][D], double *pi, double z[N][K]) {

    srand(time(NULL));
    /* normalization */
    double normalizationFactor;
#pragma omp parallel for private(normalizationFactor)
    for (int w = 0; w < K; w++) {
       normalizationFactor = 0;

#pragma omp parallel for reduction(+:normalizationFactor)
       for (int g = 0; g < D; g++) {
           mu[w][g] = rand() / (double)RAND_MAX;
           normalizationFactor = normalizationFactor + mu[w][g];
       }

#pragma omp parallel for
       for (int g = 0; g < D; g++) {
           mu[w][g] = mu[w][g] / normalizationFactor;
       }
    }

    for (int i = 0; i < 3; i++) {
        ExpectationStep(z, pi, mu, train_images);
        MaximizationStep(z, pi, mu, train_images); 
    }
}

void ExpectationStep(double z[N][K], double *pi, double mu[K][D], int **x) {
    double normalizationFactor;
#pragma omp parallel for private(normalizationFactor)
    for (int n = 0; n < N; n++) {
        normalizationFactor = 0.0;

#pragma omp parallel for reduction(+:normalizationFactor)
        for (int k = 0; k < K; k++) {
            z[n][k] = ExpectationSubStep(n, k, pi, mu, x);
            normalizationFactor = normalizationFactor + z[n][k];
        }

#pragma omp parallel for 
        for (int k = 0; k < K; k++) {
            if (normalizationFactor > 0.0) {
                z[n][k] = z[n][k] / normalizationFactor;
            } else {
                z[n][k] = 1.0 / (float)K;
            }
        }
    }
}

double ExpectationSubStep(int n, int k, double *pi, double mu[K][D], int **x) {
    double z_nk = 1.0;
/* #pragma omp parallel for reduction(*:z_nk) */
    for (int i = 0; i < D; i++) {
        z_nk = z_nk * pow(mu[k][i], x[n][i]) * pow(1.0 - mu[k][i], 1.0 - x[n][i]);      
    }
    return z_nk * pi[k];
}

void MaximizationStep(double z[N][K],double *pi, double mu[K][D], int **x) {
#pragma omp parallel for
    for (int k = 0; k < K; k++) {
        pi[k] = Nm(k, z) / (double)N;
    } 
    double *average;
#pragma omp parallel for private(average)
    for (int k = 0; k < K; k++) {
        average = Average(k, x, z);
        
#pragma omp parallel for
        for (int i = 0; i < D; i++) {
            mu[k][i] = average[i];
        }
    }
    free(average);
}

double *Average(int m, int **x, double z[N][K]) {
    double *result = (double*)malloc(sizeof(double) * D);
    memset(result, 0, sizeof(double) * D);
#pragma omp parallel for collapse(2)
    for (int i = 0; i < D; i++) {
        for (int n = 0; n < N; n++) {
            result[i] = result[i] + z[n][m] * x[n][i];
        }
    }
    double currentNm = Nm(m, z);
#pragma omp parallel for
    for (int i = 0; i < D; i++) {
        result[i] = result[i] / currentNm;
    }
    return result;
}

double Nm(int m, double z[N][K]) {
    double result = 0.0;
#pragma omp parallel for reduction(+:result)
    for (int n = 0; n < N; n++) {
        result = result + z[n][m];
    }
    return result;
}

int GetCluster(double mu[K][D], int *image) {
    double maxClusterSum = -DBL_MAX;
    int maxCluster = -1;
    for (int k = 0; k < K; k++) {
        double currentClusterSum = 0.0;
/* #pragma omp parallel for reduction(+:currentClusterSum) */
        for (int i = 0; i < D; i++) {
            currentClusterSum += image[i] ? mu[k][i] : 1.0 - mu[k][i];
        }

        if (currentClusterSum > maxClusterSum) {
            maxClusterSum = currentClusterSum;
            maxCluster = k;
        }
    }
    return maxCluster;
}






