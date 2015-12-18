#include <math.h>
#include "EM.h"

double loglikelihood(int **x, double mu[K][D], double *pi, double z[N][K]) {
    double result = 0;
    for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k++) {
            double temp = 0;

            for (int i = 0; i < D; i++) {
                temp = temp + x[n][i] * log(mu[k][i]) + (1 - x[n][i]) * log(1 - mu[k][i]);
            }
            result = result + z[n][k] * (log(pi[k]) + temp);
        }
    }
    return result;
}
