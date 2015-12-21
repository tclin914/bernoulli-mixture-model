#ifndef EM_H
#define EM_H

#define K 60
#define N 50000
#define D 28 *28

void EM(int *train_images, int *train_labels, int *test_images, int *test_labels, double *mu, double *pi, double *z);

int GetCluster(double *mu, int *image);
#endif
