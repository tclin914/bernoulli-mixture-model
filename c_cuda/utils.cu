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

void readMNIST(const char *imagefile,const char *labelfile, int num_images, int **imagedata, int **labeldata) {
    /* read digits */
    FILE *file = fopen(imagefile, "r");
    if (file == NULL) {
        fprintf(stderr, "%s open failure\n", imagefile);
        exit(1);
    }
   
    int header = 0;
    int count = 0;
    int num_rows = 0;
    int num_cols = 0;

    fread(&header, sizeof(header), 1, file);
    header = reverseInt(header);
    if (header != 2051) {
        fprintf(stderr, "Invalid image file header\n");
        exit(1);
    }
    
    fread(&count, sizeof(count), 1, file);
    count = reverseInt(count);
    if (count < num_images) {
        fprintf(stderr, "Trying to read too many digits\n");
        exit(1);
    }

    fread(&num_rows, sizeof(num_rows), 1, file);
    num_rows = reverseInt(num_rows);

    fread(&num_cols, sizeof(num_cols), 1, file);
    num_cols = reverseInt(num_cols);

    /* int **images = (int**)malloc(sizeof(int*) * num_images); */
    /* for (int i = 0; i < num_images; i++) { */
        /* images[i] = (int*)malloc(sizeof(int) * num_rows * num_cols); */
    /* } */

    int *images = (int*)malloc(sizeof(int) * num_images * num_rows * num_cols);

    int i, j , k;
    for (i = 0; i < num_images; i++) {
        for (j = 0; j < num_rows; j++) {
            for (k = 0; k < num_cols; k++) {
                unsigned char temp = 0;
                fread(&temp, sizeof(temp), 1, file);
                /* images[i][num_rows * j + k] = ((double)temp / 255) > 0.5 ? 1 : 0; */
                images[num_rows * num_cols * i + num_rows * j + k] = ((double)temp / 255) > 0.5 ? 1 : 0;
            }
        }
    }
    fclose(file);

    /* read labes */
    file = fopen(labelfile, "r");
    if (file == NULL) {
        fprintf(stderr, "%s open failure\n", labelfile);
        exit(1);
    }

    fread(&header, sizeof(header), 1, file);
    header = reverseInt(header);
    if (header != 2049) {
        fprintf(stderr, "Invalid label file header\n");
        exit(1);
    }

    fread(&count, sizeof(count), 1, file);
    if (count < num_images) {
        fprintf(stderr, "Trying to read too many digits\n");
        exit(1);
    }

    int *labels = (int*)malloc(sizeof(int) * num_images);
    for (i = 0; i < num_images; i++) {
        unsigned char temp = 0;
        fread(&temp, sizeof(temp), 1, file);
        labels[i] = (int)temp;
    }
    fclose(file);

    *imagedata = images;
    *labeldata = labels;
}

