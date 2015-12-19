#include <stdio.h>
#include <time.h>
#include <string.h>
#include "utils.h"
#include "EM.h"
#include "loglikelihood.h"

#define TRAIN_IMAGE "../data/data/train-images-idx3-ubyte"
#define TRAIN_LABEL "../data/train-labels-idx1-ubyte"
#define TEST_IMAGE "../data/t10k-images-idx3-ubyte"
#define TEST_LABEL "../data/t10k-labels-idx1-ubyte"

double z[N][K];

int main(int argc, const char *argv[])
{
    int start = time(NULL);
    /* read train data */
    int **train_images;
    int *train_labels;
    readMNIST(TRAIN_IMAGE, TRAIN_LABEL, 50000, &train_images, &train_labels);

    /* for (int i = 0; i < 10000; i++) { */
        /* for (int j = 0; j < 28 * 28; j++) { */
            /* printf("train_images[%d][%d] = %d\n", i, j, train_images[i][j]); */
        /* } */
    /* } */

    /* read test data */
    int **test_images;
    int *test_labels;
    readMNIST(TEST_IMAGE, TEST_LABEL, 10000, &test_images, &test_labels);

    double pi[K];
    double mu[K][D] = {{0}};
    memset(z, 0, sizeof(z));
    for (int i = 0; i < K; i++) {
        pi[i] = 1 / (double)K;
    }

    EM(train_images, train_labels, test_images, test_labels, mu, pi, z);

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
        /* printf("Label = %d test_label = %d\n", maxLabelOfClusters[clusterNumber], test_labels[i]); */
    }
    printf("errNum = %d\n", errNum);
    printf("errRate = %f\n", errNum / (double)10000);


    /* for (int k = 0; k < 30; k++) { */
        /* int clusterNumber = GetCluster(mu, test_images[k]); */
        /* printf("clusterNumber = %d\n", clusterNumber); */
        /* int digitsInTheSameCluster[10] = {0}; */
        /* for (int i = 0; i < N; i++) { */
            /* if (clusterNumber == GetCluster(mu, train_images[i])) { */
                /* digitsInTheSameCluster[train_labels[i]]++;         */
            /* } */
        /* } */

        /* int maxDigits = -1; */
        /* int maxLabel = -1; */
        /* for (int i = 0; i < 10; i++) { */
            /* if (digitsInTheSameCluster[i] > maxDigits) { */
                /* maxDigits = digitsInTheSameCluster[i]; */
                /* maxLabel = i; */
            /* } */
        /* } */
        /* printf("maxLabel = %d\n", maxLabel); */
        /* printf("test_label = %d\n", test_labels[k]); */
    /* } */
    /* for (int k = 0; k < K; k++) { */
        /* for (int i = 0; i < D; i++) { */
            /* printf("mu[%d][%d] = %f\n", k, i, mu[k][i]); */
        /* } */
    /* } */

    /* loglikelihood(train_images, mu, pi, z);     */

    /* for (int i = 0; i < K; i++) { */
        /* printf("pi[%d] = %f\n", i, pi[i]); */
    /* } */

    int end = time(NULL);
    printf("time = %d\n", end - start);

    return 0;
}
