#include <stdio.h>
#include <time.h>
#include <string.h>
#include "utils.h"
#include "EM.h"
#include "loglikelihood.h"

#define TRAIN_IMAGE "./data/train-images-idx3-ubyte"
#define TRAIN_LABEL "./data/train-labels-idx1-ubyte"
#define TEST_IMAGE "./data/t10k-images-idx3-ubyte"
#define TEST_LABEL "./data/t10k-labels-idx1-ubyte"

double z[N][K];

int main(int argc, const char *argv[])
{
    int start = time(NULL);
    /* read train data */
    int **train_images;
    int *train_labels;
    readMNIST(TRAIN_IMAGE, TRAIN_LABEL, 50000, &train_images, &train_labels);

    /* read test data */
    int **test_images;
    int *test_labels;
    readMNIST(TEST_IMAGE, TEST_LABEL, 10000, &test_images, &test_labels);

    double pi[K] = {1/ K};
    double mu[K][D] = {{0}};
    memset(z, 0, sizeof(z));

    EM(train_images, train_labels, test_images, test_labels, mu, pi, z);

    loglikelihood(train_images, mu, pi, z);    

    int end = time(NULL);
    printf("time = %d\n", end - start);

    return 0;
}
