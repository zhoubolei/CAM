/**
 * Distance transform for binary image or gray-scale image.
 * @param 
 * @return 
 */

#include <stdlib.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif
        
#define INF 1E20

#define SQUARE(q) ((q)*(q))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define ROUND(t) ((int)((t) + 0.5))
#define BOUND_8U(t) ((t) < 0 ? 0 : (t) > 255 ? 255 : (t))

static void dt_row(const double *f, int n, double *d, double *z, int *v) {
        int k, q;

        v[0] = 0;
        z[0] = -INF; 
        z[1] = +INF;
        
        k = 0;
        for (q = 1; q < n; ++q) {
                double s = ((f[q]+SQUARE(q))-(f[v[k]]+SQUARE(v[k])))/(double)(2*q-2*v[k]);
                while (s <= z[k]) {
                        k--;
                        s = ((f[q]+SQUARE(q))-(f[v[k]]+SQUARE(v[k])))/(double)(2*q-2*v[k]);
                }
                k++;
                v[k] = q;
                z[k] = s;
                z[k+1] = +INF;
        }
        
        k = 0;
        for (q = 0; q < n; ++q) {
                while (z[k+1] < q) 
                        k++;
                d[q] = SQUARE(q-v[k]) + f[v[k]];
        }
}

void 
dt(double *m, int rows, int cols) {
        const int n = MAX(rows, cols);
        double *f = (double *)malloc(sizeof(f[0]) * n);
        double *d = (double *)malloc(sizeof(d[0]) * n);
        double *z = (double *)malloc(sizeof(z[0]) * (n+ 1));
        int *v = (int *)malloc(sizeof(v[0]) * n);
        int x, y;
        
        for (x = 0; x < cols; ++x) {
                for (y = 0; y < rows; ++y) {
                        f[y] = m[y*cols + x];
                }
                dt_row(f, rows, d, z, v);
                for (y = 0; y < rows; ++y) {
                        m[y*cols + x] = d[y];
                }
        }
        
        for (y = 0; y < rows; ++y) {
                for (x = 0; x < cols; ++x) {
                        f[x] = m[y*cols + x];
                }
                dt_row(f, cols, d, z, v);
                for (x = 0; x < cols; ++x) {
                        m[y*cols + x] = d[x];
                }
        }

        free(f);
        free(d);
        free(z);
        free(v);
}

static void 
min_max(const double *m, int sz, double *min, double *max) {
        double mi = m[0], ma = m[0];
        int i = 1;
        for (; i < sz; ++i) {
                if (m[i] > ma) {
                        ma = m[i];
                }
                else if (m[i] < mi) {
                        mi = m[i];
                }
        }
        *min = mi;
        *max = ma;
}

static void 
double_to_image(const double *m, int rows, int cols, unsigned char *data, int step) {
        int i, j;
        double mi, ma, scale;
        min_max(m, rows * cols, &mi, &ma);

        if (mi == ma) {
                return ;
        }
        
        scale = 255.0 / (ma - mi);

        for (i = 0; i < rows; ++i) {
                for (j = 0; j < cols; ++j) {
                        const double s = m[i*cols + j] * scale;
                        const int t = ROUND(s);
                        data[i*step + j] = BOUND_8U(t);
                }
        }
}

static void 
sqrt_m(double *m, int sz) {
        int i = 0;
        for (; i < sz; ++i) {
                m[i] = sqrt(m[i]);
        }
}

void 
dt_gray(unsigned char *gray, int rows, int cols, int step) {
        double *m = (double *)malloc(sizeof(m[0]) * rows * cols);
        int i, j;
        const double vstep = 100.0; /* big enough to transform the distance... */
        for (i = 0; i < rows; ++i) {
                for (j = 0; j < cols; ++j) {
                        m[i*cols + j] = vstep * (double)gray[i*step + j];
                }
        }

        dt(m, rows, cols);
        sqrt_m(m, rows * cols);
        double_to_image(m, rows, cols, gray, step);
        free(m);
}

void 
dt_binary(unsigned char *bimg, int rows, int cols, int step) {
        double *m = (double *)malloc(sizeof(m[0]) * rows * cols);
        int i, j;
        for (i = 0; i < rows; ++i) {
                for (j = 0; j < cols; ++j) {
                        m[i*cols + j] = bimg[i*step + j] > 0 ? +INF : 0.0;
                }
        }
        
        dt(m, rows, cols);
        sqrt_m(m, rows*cols);
        double_to_image(m, rows, cols, bimg, step);
        free(m);
}


#ifdef __cplusplus
}
#endif
