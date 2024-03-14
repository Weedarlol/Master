#ifndef GLOBAL_FUNCTIONS_H
#define GLOBAL_FUNCTIONS_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

void fillValues(double *mat, double dx, double dy, int width, int height);

void fillValues3D(double *mat, int width, int height, int depth_node, double dx, double dy, double dz, int rank);

#endif
