#include <stdio.h>
#include <string.h>
#include <sys/sysinfo.h>
#include "utils.h"
#include "EM.h"
#include "loglikelihood.h"
#include "timer.h"

#include <omp.h>

#define TRAIN_IMAGE "../data/train-images-idx3-ubyte"
#define TRAIN_LABEL "../data/train-labels-idx1-ubyte"
#define TEST_IMAGE "../data/t10k-images-idx3-ubyte"
#define TEST_LABEL "../data/t10k-labels-idx1-ubyte"

#define T_READ 0
#define T_EM   1
#define T_TEST 2
#define T_LAST 3

double z[N][K];

int main(int argc, const char *argv[])
{
    /* openmp environment setting */
    omp_set_dynamic(0);
    omp_set_num_threads(8);
    /* printf("%d\n", get_nprocs()); */

    for (int i = 0; i < T_LAST; i++) {
        timer_clear(i);
    }

    timer_start(T_READ);

    /* read train data */
    int **train_images;
    int *train_labels;
    readMNIST(TRAIN_IMAGE, TRAIN_LABEL, 50000, &train_images, &train_labels);

    /* read test data */
    int **test_images;
    int *test_labels;
    readMNIST(TEST_IMAGE, TEST_LABEL, 10000, &test_images, &test_labels);

    timer_stop(T_READ);
    
    printf("Read MNIST data time = %f seconds\n", timer_read(T_READ));

    timer_start(T_EM);
    
    double pi[K];
    double mu[K][D] = {{0}};
    memset(z, 0, sizeof(z));
#pragma omp parallel for
    for (int i = 0; i < K; i++) {
        pi[i] = 1 / (double)K;
    }

    EM(train_images, train_labels, test_images, test_labels, mu, pi, z);

    timer_stop(T_EM);

    printf("EM Algorithm time = %f seconds\n", timer_read(T_EM));

    timer_start(T_TEST);

    int digitsOfClusters[K][10] = {{0}};
    for (int i = 0; i < N; i++) {
        int clusterNumber = GetCluster(mu, train_images[i]);
        digitsOfClusters[clusterNumber][train_labels[i]]++;
    }

    int maxLabelOfClusters[K] = {0};
    for (int i = 0; i < K; i++) {
        int maxDigits = -1;
        int maxLabel = -1;
        for (int j = 0; j < 10; j++) {
            if (digitsOfClusters[i][j] > maxDigits) {
                maxDigits = digitsOfClusters[i][j];
                maxLabel = j;
            }
        }
        maxLabelOfClusters[i] = maxLabel;
    }

    int errNum = 0;
    for (int i = 0; i < 10000; i++) {
        int clusterNumber = GetCluster(mu, test_images[i]);
        if (maxLabelOfClusters[clusterNumber] != test_labels[i]) {
            errNum++;
        }
    }
    
    timer_stop(T_TEST);

    printf("Test time  = %f seconds\n", timer_read(T_TEST));

    printf("ErrNum = %d\n", errNum);
    printf("ErrRate = %f\n", errNum / (double)10000);

    /* loglikelihood(train_images, mu, pi, z);     */

    for (int i = 0; i < N; i++) {
        free(train_images[i]);
    }
    free(train_labels);
    for (int i = 0; i < 10000; i++) {
        free(test_images[i]);
    }
    free(test_images[i]);

    return 0;
}
