#include "hausdorff.h"

double Hausdorff::score(Mat& reference, Mat& test)
{
    vector<Point2d> ref_points;
    Hausdorff::matrix_to_vector(reference, ref_points);

    vector<Point2d> test_points;
    Hausdorff::matrix_to_vector(test, test_points);

    Mat distances, distancesT;
    Hausdorff::vector_to_dist(ref_points, test_points, distances);
    transpose(distances, distancesT);

    double h_rt = Hausdorff::hausdorffdist(distances);
    double h_tr = Hausdorff::hausdorffdist(distancesT);

    double dist = h_rt;
    if (h_tr > h_rt)
    {
        dist = h_tr;
    }
    return dist;
}

double Hausdorff::hausdorffdist(Mat& distances)
{
    Rect col = Rect(0, 0, 1, distances.rows);
    Mat submatrix = distances(col);
    Mat smallestv = Mat::zeros(Size(distances.cols, 1), CV_64FC1);

    for (int j = 1; j < distances.cols; ++j)
    {
        col = Rect(j, 0, 1, distances.rows);
        submatrix = distances(col);
        double max, min;
        minMaxLoc(submatrix, &min, &max);
        smallestv.at<double>(0, j) = min;
    }
    double max2, min2;
    minMaxLoc(smallestv, &min2, &max2);
    return max2;
}

void Hausdorff::vector_to_dist(vector<Point2d>& points1, vector<Point2d>& points2, Mat& distances)
{
    Size distsize = Size(int(points1.size()), int(points2.size()));
    distances = Mat::zeros(distsize, CV_64FC1);
    for (int j = 0; j < int(points1.size()); ++j)
    {
        Point2d p1 = points1.at(j);
        for (int i = 0; i < int(points2.size()); ++i)
        {
            Point2d p2 = points2.at(i);
            double euc_dist = norm(p1 - p2);
            distances.at<double>(i, j) = euc_dist;
        }
    }
}

void Hausdorff::matrix_to_vector(Mat& input, vector<Point2d>& stored_points)
{
    for (int i = 0; i < input.rows; ++i)
    {
        for (int j = 0; j < input.cols; ++j)
        {
            ushort current = input.at<ushort>(i, j);
            if (current > 0)
            {
                Point2d c_point = Point2d(double(j), double(i));
                stored_points.push_back(c_point);
            }
        }
    }
}
