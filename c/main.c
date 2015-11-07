#include <stdio.h>
#include "utils.h"
#include "EM.h"

#define TRAIN_IMAGE "./data/train-images-idx3-ubyte"
#define TRAIN_LABEL "./data/train-labels-idx1-ubyte"
#define TEST_IMAGE "./data/t10k-images-idx3-ubyte"
#define TEST_LABEL "./data/t10k-labels-idx1-ubyte"

int main(int argc, const char *argv[])
{
    /* read train data */
    int **train_images;
    int *train_labels;
    readMNIST(TRAIN_IMAGE, TRAIN_LABEL, 50000, &train_images, &train_labels);

    /* read test data */
    int **test_images;
    int *test_labels;
    readMNIST(TEST_IMAGE, TEST_LABEL, 10000, &test_images, &test_labels);

    EM(train_images, train_labels, test_images, test_labels);
    
    


    return 0;
}
