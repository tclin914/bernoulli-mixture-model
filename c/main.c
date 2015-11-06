#include "utils.h"

#define TRAIN_IMAGE "./data/train-images-idx3-ubyte"
#define TRAIN_LABEL "./data/train-labels-idx1-ubyte"
#define TEST_IMAGE "./data/t10k-images-idx3-ubyte"
#define TEST_LABEL "./data/t10k-labels-idx1-ubyte"

int main(int argc, const char *argv[])
{
    double **data;
    readMNIST(TEST_IMAGE, 50000, 400);    
    return 0;
}
