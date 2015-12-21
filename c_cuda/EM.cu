#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <float.h>
#include "EM.h"

int *imagesptr;
double *muptr;
double *piptr;
double *zptr;
int *kptr;
int *dptr;
int *nptr;
int *resultptr;
int result[K * D];

__device__ double Cuda_ExpectationSubStep(int *d_size, int n, int k, double *pi, double *mu, int *x) {
    double z_nk = pi[k];
    int i;
    for (i = 0; i < *d_size; i++) {
        z_nk = z_nk * 
            pow(mu[(*d_size) * k + i], x[(*d_size) * n + i]) * 
            pow(1.0 - mu[(*d_size) * k + i], 1.0 - x[(*d_size) * n + i]);
    }
    return z_nk;
}

__global__ void Cuda_ExpectationStep(int *k_size, int *d_size, double *pi, double *z, double *mu, int *x) {
    int n = (blockIdx.x * blockDim.x) + threadIdx.x;
    double normalizationFactor = 0.0;
    int k;

    for (k = 0; k < *k_size; k++) {
        z[(*k_size) * n + k] = Cuda_ExpectationSubStep(d_size, n, k, pi, mu, x);
        normalizationFactor = normalizationFactor + z[(*k_size) * n + k];
    }

    for (k = 0; k < *k_size; k++) {
        if (normalizationFactor > 0.0) {
            z[(*k_size) * n + k] = z[(*k_size) * n + k] / normalizationFactor;
        } else {
            z[(*k_size) * n + k] = 1.0 / (float)(*k_size);
        }
    }
}

__device__ double Nm(int m, int *k_size, int *n_size, double *z) {
    double result = 0.0;
    int n;
    for (n = 0; n < *n_size; n++) {
        result = result + z[(*k_size) * n + m];
    }
    return result;
}

__device__ void Average(int m, int *k_size, int *d_size, int *n_size, 
        int *x, double *z, double *mu) {
    double currentNm = Nm(m, k_size, n_size, z);
    int i, n;
    for (i = 0; i < *d_size; i++) {
        double result = 0.0;
        for (n = 0; n < *n_size; n++) {
            result = result + z[(*k_size) * n + m] * x[(*d_size) * n + i];
        }
        mu[(*d_size) * m + i] = result / currentNm; 
    }
}

__global__ void Cuda_MaximizationStep(int *k_size, int *d_size, int *n_size, 
        double *pi, double *z, double *mu, int *x) {     
    int k = threadIdx.x;
    pi[k] = Nm(k, k_size, n_size, z) / (double)(*n_size);
    
    Average(k, k_size, d_size, n_size, x, z, mu);
}

__global__ void test(double *pi) {
    int i;
    for (i = 0; i < 40; i++) {
        pi[i] = 2.5;
    }
}

void EM(int *train_images, int *train_labels, int *test_images, int *test_labels, double *mu, double *pi, double *z) {

    srand(time(NULL));
    /* normalization */
    double normalizationFactor;
    int w, g;
    for (w = 0; w < K; w++) {
        normalizationFactor = 0;
        for (g = 0; g < D; g++) {
            mu[D * w + g] = rand() / (double)RAND_MAX;
            normalizationFactor = normalizationFactor + mu[D * w + g];
        }

        for (g = 0; g < D; g++) {
            mu[D * w + g] = mu[D * w + g] / normalizationFactor;
        }
    }
    
    int count;
    cudaGetDeviceCount(&count);
    if (count == 0) {
        fprintf(stderr, "Here is no cuda device\n");
        exit(1);
    }

    int i;
    for (i = 0; i < count; i++) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            
        }
    }
    cudaSetDevice(i);

    cudaMalloc((void**)&imagesptr, sizeof(int) * N * D);
    cudaMalloc((void**)&muptr, sizeof(double) * K * D);
    cudaMalloc((void**)&piptr, sizeof(double) * K);
    cudaMalloc((void**)&zptr, sizeof(double) * N * K);
    cudaMalloc((void**)&kptr, sizeof(int));
    cudaMalloc((void**)&dptr, sizeof(int));
    cudaMalloc((void**)&nptr, sizeof(int));

    int k_size = K;
    int d_size = D;
    int n_size = N;

    cudaMemcpy(imagesptr, train_images, sizeof(int) * N * D, cudaMemcpyHostToDevice);
    cudaMemcpy(muptr, mu, sizeof(double) * K * D, cudaMemcpyHostToDevice);
    cudaMemcpy(piptr, pi, sizeof(double) * K, cudaMemcpyHostToDevice);
    cudaMemcpy(zptr, z, sizeof(double) * N * K, cudaMemcpyHostToDevice);
    cudaMemcpy(kptr, &k_size, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dptr, &d_size, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(nptr, &n_size, sizeof(int), cudaMemcpyHostToDevice);


    int threadPerBlock, numBlocks;

    for (i = 0; i < 3; i++) {
        threadPerBlock = 500;
        numBlocks = N / 500;

        Cuda_ExpectationStep<<<numBlocks, threadPerBlock>>>(kptr, dptr, piptr, zptr, muptr, imagesptr);

        threadPerBlock = K;
        numBlocks = 1;

        Cuda_MaximizationStep<<<numBlocks, threadPerBlock>>>(kptr, dptr, nptr, piptr, zptr, muptr, imagesptr);
    }

    cudaMemcpy(mu, muptr, sizeof(double) * K * D, cudaMemcpyDeviceToHost);

    cudaFree(imagesptr);
    cudaFree(muptr);
    cudaFree(piptr);
    cudaFree(zptr);
    cudaFree(kptr);
    cudaFree(dptr);
    cudaFree(nptr);
}

int GetCluster(double *mu, int *image) {
    double maxClusterSum = -DBL_MAX;
    int maxCluster = -1;
    int k, i;
    for (k = 0; k < K; k++) {
        double currentClusterSum = 0.0;
        for (i = 0; i < D; i++) {
            currentClusterSum += image[i] ? mu[D * k + i] : 1.0 - mu[D * k + i];
        }

        if (currentClusterSum > maxClusterSum) {
            maxClusterSum = currentClusterSum;
            maxCluster = k;
        }
    }
    return maxCluster;
}
