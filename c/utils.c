#include <stdio.h>
#include <stdlib.h>

int reverseInt(int i) {
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

double **readMNIST(const char *filename, int num_images, int data_image) {
    /* read digits */
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "%s open failure\n", filename);
        exit(1);
    }
   
    int header = 0;
    int num_of_images = 0;
    int num_rows = 0;
    int num_cols = 0;

    fread(&header, sizeof(header), 1, file);
    header = reverseInt(header);
    if (header != 2051) {
        fprintf(stderr, "Invalid image file header");
        exit(1);
    }
    printf("header = %d\n", header);
    
    fread(&num_of_images, sizeof(num_of_images), 1, file);
    num_of_images = reverseInt(num_of_images);
    printf("num_of_images = %d\n", num_of_images);

    fread(&num_rows, sizeof(num_rows), 1, file);
    num_rows = reverseInt(num_rows);
    printf("num_rows = %d\n", num_rows);

    fread(&num_cols, sizeof(num_cols), 1, file);
    num_cols = reverseInt(num_cols);
    printf("num_cols = %d\n", num_cols);

    double **data = (double**)malloc(sizeof(double*) * num_images);
    for (int i = 0; i < num_images; i++) {
        data[i] = (double*)malloc(sizeof(double) * num_rows * num_cols);
    }

    for (int i = 0; i < num_images; i++) {
        for (int j = 0; j < num_rows; j++) {
            for (int k = 0; k < num_cols; k++) {
                unsigned char temp = 0;
                fread(&temp, sizeof(temp), 1, file);
                data[i][num_rows * j + k] = (double)temp;
                if (i < 3) {
                    printf("data = %f\n", (double)temp);
                    fflush(stdout);
                }

            }
        }
    }
}

