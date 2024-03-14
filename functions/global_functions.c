#include "global_functions.h"

void fillValues(double *mat, double dx, double dy, int width, int height) {
    double x, y;

    memset(mat, 0, height * width * sizeof(double));

    for (int i = 1; i < height - 1; i++) {
        y = i * dy; // y coordinate
        for (int j = 1; j < width - 1; j++) {
            x = j * dx; // x coordinate
            mat[j + i * width] = sin(M_PI * y) * sin(M_PI * x);
        }
    }
}

void fillValues3D(double *mat, int width, int height, int depth, double dx, double dy, double dz, int rank) {
    double x, y, z;

    // Assuming the data in the matrix is stored contiguously in memory
    memset(mat, 0, height * width * depth * sizeof(double));

    for (int i = 1; i < depth-1; i++) {
        z = (i + rank * (depth - 2)) * dz; // z coordinate
        for (int j = 1; j < height - 1; j++) {
            y = j * dy; // z coordinate
            for (int k = 1; k < width - 1; k++) {
                x = k * dx; // x coordinate
                mat[k +  j*width + i*width*height] = sin(M_PI * x) * sin(M_PI * y) * sin(M_PI * z);
            }
        }
    }
}
