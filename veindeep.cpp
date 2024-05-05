#include "veindeep.h"

void VeinDeep::gen_pattern(Mat& input, Mat& output)
{
    output = Mat::zeros(input.size(), CV_16UC1);
    for (int i = 0; i < input.rows; ++i)
    {
        ushort previous_value = 0;
        for (int j = 0; j < input.cols; ++j)
        {
            ushort signal = input.at<ushort>(i, j);
            // Store value if sign changes to positive
            if (signal != previous_value)
            {
                previous_value = signal;
                if (signal > 0)
                {
                    Point2d intersect = Point2i(j, i);
                    output.at<ushort>(intersect) = numeric_limits<ushort>::max();
                }
            }
        }
    }
}

double VeinDeep::score(Mat& reference, Mat& test, double sigma)
{
    Mat outpattern1;
    VeinDeep::gen_pattern(reference, outpattern1);

    Mat outpattern2;
    VeinDeep::gen_pattern(test, outpattern2);

    double score = Kerneldist::score(outpattern1, outpattern2, sigma);

    return score;
}
