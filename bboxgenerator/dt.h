#ifndef DT_H
#define DT_H

#ifdef __cplusplus
extern "C" {
#endif

void dt(double *m, int rows, int cols);
void dt_binary(unsigned char *bimg, int rows, int cols, int step);
void dt_gray(unsigned char *gray, int rows, int cols, int step);
        
#ifdef __cplusplus
}
#endif

#endif
