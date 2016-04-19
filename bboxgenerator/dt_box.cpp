/*
----------------------------------------
Given an heatmap, given out the bbox.

0. Get the DT-ed images
1. detect all contour in the bboxes
2. merge based on some rules.
3. output the bbox
----------------------------------------
*/
#include <stdio.h>
#include "dt.h"
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <assert.h>

using namespace cv;
using std::vector;

#define SCALE_NUM 3

struct Data
{
        Data() : size(SCALE_NUM)
        {
                for (int i = 0; i < SCALE_NUM; ++i)
                {
                        images[i] = NULL;
                }
        }

        ~Data()
        {
                for (int i = 0; i < SCALE_NUM; ++i)
                {
                        if (images[i])
                        {
                                cvReleaseImage(&(images[i]));
                                images[i] = NULL;
                        }
                }
        }
        int size;
        IplImage *images[SCALE_NUM];
};


static int g_Ths[SCALE_NUM] = {30, 90, 150};
                
static Data *
fromDT(const IplImage *gray)
{
        Data *data = new Data;
        for (int i = 0; i < data->size; ++i)
        {
                data->images[i] = cvCreateImage(cvGetSize(gray), 8, 1);
                cvThreshold(gray, data->images[i], g_Ths[i], 255, CV_THRESH_BINARY);
                dt_binary((unsigned char*)data->images[i]->imageData, data->images[i]->height, data->images[i]->width, data->images[i]->widthStep);
                cvThreshold(data->images[i], data->images[i], 10, 255, CV_THRESH_BINARY);
        }
        return data;
}


static int
LIMIT(int v, int L, int R)
{
        return v < L ? L : (v > R ? R : v);
}

static vector<CvRect>
getBBox(struct Data *data)
{
        vector<CvRect> bboxes;
        const int W = data->images[0]->width;
        const int H = data->images[0]->height;
        
        for (int i = 0; i < data->size; ++i)
        {
                cv::Mat a = cv::cvarrToMat(data->images[i]);
                vector< vector<cv::Point> >contours;
                vector<cv::Vec4i> hie;
                cv::findContours(a, contours, hie, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
                for (int j = 0; j < contours.size(); ++j)
                {
                        cv::Rect bb = cv::boundingRect( contours[j] );
                        CvRect cr;
                        cr.x = LIMIT(bb.x, 0, W-5);
                        cr.y = LIMIT(bb.y, 0, H-5);
                        cr.width = LIMIT(bb.width, 0, W - bb.x-5);
                        cr.height = LIMIT(bb.height, 0, H-bb.y-5);
                        //printf("%d, %d, %d, %d\n", W, H, cr.width, cr.height);
                        bboxes.push_back(cr);
                }
        }
        return bboxes;
}


/*
----------------------------------------
x_overlap = Math.max(0, Math.min(x12,x22) - Math.max(x11,x21));
y_overlap = Math.max(0, Math.min(y12,y22) - Math.max(y11,y21));
overlapArea = x_overlap * y_overlap;
----------------------------------------
*/
static bool
big_overlap(const CvRect &a, const CvRect &b)
{
        int t = (double)std::max(a.width * a.height, b.width * b.height) * 0.5;
        int x11, y11, x12, y12, x21, y21, x22, y22;
        x11 = a.x;
        y11 = a.y;
        x12 = a.x + a.width;
        y12 = a.y + a.height;
        x21 = b.x;
        y21 = b.y;
        x22 = b.x + b.width;
        y22 = b.y + b.height;
        int x_overlap = std::max(0, std::min(x12,x22) - std::max(x11,x21));
        int y_overlap = std::max(0, std::min(y12,y22) - std::max(y11,y21));
        int overlapArea = x_overlap * y_overlap;
        return overlapArea > t;
}

/*
----------------------------------------
1. Overlap > max(area(A), area(B)) * 0.5

0. rank BB
1. from big to small:
     
----------------------------------------
*/
static void
mergeBBox(vector<CvRect> &bboxes)
{
        for (int i = 0; i < bboxes.size(); ++i)
        {
                for (int j = i + 1; j < bboxes.size(); ++j)
                {
                        if (big_overlap(bboxes[i], bboxes[j]))
                        {
                                // remove small one
                                bboxes.erase(bboxes.begin() + j);
                        }
                }
        }
        return ;
}

static bool
my_cmp(const CvRect& a, const CvRect& b)
{
    return a.width * a.height > b.width * b.height;
}


static void
rankBBox(vector<CvRect> &bboxes)
{
        std::sort(bboxes.begin(), bboxes.end(), my_cmp);
}


static void
draw(const vector<CvRect> &rects, const char *iname)
{
        IplImage *img = cvLoadImage(iname, 1);
        const CvScalar color = cvScalar(0,0,255,0);
        
        for (int i = 0; i < rects.size(); ++i)
        {
                CvRect r = rects[i];
                cvRectangle(img, cvPoint(r.x, r.y), cvPoint(r.x + r.width, r.y + r.height), color, 3, 8, 0);
        }
        cvNamedWindow("draw", 1);
        cvShowImage("draw", img);
        cvWaitKey(0);
        cvReleaseImage(&img);
}


static void
output(const vector<CvRect> &rects, const char *filen)
{
        FILE *fp = fopen(filen, "w");
        assert(fp != NULL);
        for (int i = 0; i < rects.size(); ++i)
        {
                fprintf(fp, "%d %d %d %d ", rects[i].x, rects[i].y, rects[i].width, rects[i].height);
        }
        fclose(fp);
        return ;
}

static void
output(const vector<CvRect> &rects)
{
        for (int i = 0; i < rects.size(); ++i)
        {
                printf("%d %d %d %d ", rects[i].x, rects[i].y, rects[i].width, rects[i].height);
        }
        printf("\n");
}

int 
main(int argc, char *argv[])
{
        if (argc != 5 && argc != 6)
        {
                puts(">>>./program image.jpg th0 th1 th2\nor");
                puts(">>>./program image.jpg th0 th1 th2 output.txt");
                return -1;
        }

        IplImage *gray = cvLoadImage(argv[1], 0);
        if (!gray)
        {
                puts("Can not open image, dude!\n");
        }

        // set the thresholds
        {
                int t0, t1, t2;
                t0 = atoi(argv[2]);
                t1 = atoi(argv[3]);
                t2 = atoi(argv[4]);
                if (0 < t0 && t0 < t1 && t1 < t2 && t2 < 255)
                {
                        g_Ths[0] = t0;
                        g_Ths[1] = t1;
                        g_Ths[2] = t2;
                }
        }
        
        
        Data *data = fromDT(gray);
        vector<CvRect> rects = getBBox(data);
        rankBBox(rects);
        mergeBBox(rects);

        
        //if (argc == 4)
        //        draw(rects, argv[3]);

        
        
        if (argc == 6)
                output(rects, argv[5]);
        else
                output(rects);
        
        delete data;
        cvReleaseImage(&gray);        
        return 0;
}
