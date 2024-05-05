#include "kerneldist.h"

double Kerneldist::score(Mat& reference, Mat& test, double sigma)
{
    vector<Point2d> ref_points;
    Kerneldist::matrix_to_vector(reference, ref_points);

    vector<Point2d> test_points;
    Kerneldist::matrix_to_vector(test, test_points);

    double ref_dist = Kerneldist::vector_to_kernel(ref_points, ref_points, sigma);
    double test_dist = Kerneldist::vector_to_kernel(test_points, test_points, sigma);
    double combine_dist = Kerneldist::vector_to_kernel(ref_points, test_points, sigma);
    double score = ref_dist + test_dist - (2 * combine_dist);

    return score;
}

void Kerneldist::matrix_to_vector(Mat& input, vector<Point2d>& source_points)
{
    for (int i = 0; i < input.rows; ++i)
    {
        for (int j = 0; j < input.cols; ++j)
        {
            ushort current = input.at<ushort>(i, j);
            if (current > 0)
            {
                Point2d c_point = Point2d(double(j), double(i));
                source_points.push_back(c_point);
            }
        }
    }
}

double Kerneldist::vector_to_kernel(vector<Point2d>& points1, vector<Point2d>& points2, double sigma)
{
    double sum = 0;
    for (size_t i = 0; i < points1.size(); ++i)
    {
        Point2d p1 = points1.at(i);
        for (size_t j = 0; j < points2.size(); ++j)
        {
            Point2d p2 = points2.at(j);
            double euc_dist = norm(p1 - p2);
            double exponent = - (euc_dist * euc_dist) / (sigma * sigma);
            double k_dist = exp(exponent);
            sum += k_dist;
        }
    }
    return sum;
}
