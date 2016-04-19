/*
----------------------------------------
Using Heat map as the foreground input of
the grabcut.

Update:
0. output the biggest bounding box
1. expose the two thresholding value to the command line.
----------------------------------------
*/
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using std::vector;

static int g_th0 = 10;
static int g_th1 = 40;

/*
----------------------------------------
Using the simplest thresholding to get the foreground.
----------------------------------------
*/
static Mat
foreground(const Mat &heatmap)
{
        Mat bm;
        Mat re = heatmap.clone();
        re.setTo(GC_BGD);
        
        threshold(heatmap, bm, g_th0, 255, THRESH_BINARY); 
        re.setTo(GC_PR_BGD, bm);
        threshold(heatmap, bm, g_th1, 255, THRESH_BINARY);
        re.setTo(GC_PR_FGD, bm);
        
        return re;
}


static Mat
cut(const Mat &src, const Mat &heatmap)
{
        Mat mask = foreground(heatmap);
        Mat bgModel,fgModel; 
        grabCut(src, mask, Rect(), bgModel,fgModel, 1, cv::GC_INIT_WITH_MASK);
        Mat1b mask_fgpf = ( mask == cv::GC_FGD) | (mask == cv::GC_PR_FGD);
        Mat3b tmp = Mat3b::zeros(src.rows, src.cols);
        src.copyTo(tmp, mask_fgpf);
        return tmp;
}

/*
----------------------------------------
The same with cut_mask, but save the segmented image
----------------------------------------
*/
static Mat
cut_mask_save(const Mat &src, const Mat &heatmap, const char *dstname)
{
        Mat mask = foreground(heatmap);
        Mat bgModel,fgModel; 
        grabCut(src, mask, Rect(), bgModel,fgModel, 1, cv::GC_INIT_WITH_MASK);
        Mat mask_fgpf = (mask == cv::GC_FGD) | (mask == cv::GC_PR_FGD);
        Mat tmp = Mat3b::zeros(src.rows, src.cols);
        src.copyTo(tmp, mask_fgpf);
        imwrite(dstname, tmp);
        return mask_fgpf;
}


/*
----------------------------------------
cut, return the mask.
----------------------------------------
*/
static Mat
cut_mask(const Mat &src, const Mat &heatmap)
{
        Mat mask = foreground(heatmap);
        Mat bgModel,fgModel; 
        grabCut(src, mask, Rect(), bgModel,fgModel, 1, cv::GC_INIT_WITH_MASK);
        return ( mask == cv::GC_FGD) | (mask == cv::GC_PR_FGD);
}


static bool
rect_cmp(const Rect& a, const Rect& b)
{
        return a.area()> b.area();
}


static vector<Rect>
bbox(Mat &mask)
{
        vector<Rect> rs;
        vector< vector<Point> >contours;
        vector<Vec4i> hie;
        findContours(mask, contours, hie, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
        for (int j = 0; j < contours.size(); ++j)
        {
                const Rect bb = boundingRect( contours[j] );
                if (bb.area() > 10)
                {
                        rs.push_back(bb);
                }
                
        }

        if (rs.size() == 0)
        {
                rs.push_back(Rect(0,0,mask.cols, mask.rows));
                return rs;
        }

        sort(rs.begin(), rs.end(), rect_cmp);
        return rs;
}


static void
output(const vector<Rect> &rs, const char *filen)
{
        FILE *fp = fopen(filen, "w");
        assert(fp != NULL);
        for (int i = 0; i < rs.size(); ++i)
        {
                fprintf(fp, "%d %d %d %d ", rs[i].x, rs[i].y, rs[i].width, rs[i].height);
        }

        fclose(fp);
        return ;
}

int 
main(int argc, char *argv[])
{
        if (argc != 4 && argc != 6 && argc != 7)
        {
                puts(">>./cut sample.jpg heat.jpg output.txt\nor");
                puts(">>./cut sample.jpg heat.jpg output.txt th1[=10] th2[=40]\nor");
                puts(">>./cut sample.jpg heat.jpg output.txt th1[=10] th2[=40] save_image_name.jpg");
                return 0;
        }

        if (argc == 6)
        {
                int t0 = atoi(argv[4]);
                int t1 = atoi(argv[5]);
                if (0 <= t0 && t0 < t1 && t1 <= 255)
                {
                        g_th0 = t0;
                        g_th1 = t1;
                }
        }
        
        Mat src = imread(argv[1], 1);
        Mat heat = imread(argv[2], 0);
        Mat m;

        if (argc == 7)
                m = cut_mask_save(src, heat, argv[6]);
        else
                m = cut_mask(src, heat);
        
        vector<Rect> bbs = bbox(m);
        output(bbs, argv[3]);
        //rectangle(src, box, Scalar(0,0,255));
        //imwrite(argv[3], src);
        //imshow("result", src);
        //waitKey(0);
        return 0;
}
