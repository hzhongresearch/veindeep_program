#ifndef KERNELDIST_H
#define KERNELDIST_H

#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

///
/// \brief The Kerneldist class
/// Computes the Gaussian kernel distance which measures the degree of similarity between 2 binary images, a smaller value indicates greater similarity.
///
class Kerneldist
{
public:
    ///
    /// \brief score Computes Gaussian kernel distance
    /// \param reference
    /// \param test
    /// \param sigma Penalty when 2 points are mis-aligned
    /// \return
    ///
    static double score(Mat& reference, Mat& test, double sigma);
protected:
    ///
    /// \brief matrix_to_vector Stores the coordinates of all non-zero pixels in binary image
    /// \param input
    /// \param stored_points Stored vector of non-zero pixels
    ///
    static void matrix_to_vector(Mat& input, vector<Point2d>& source_points);
    ///
    /// \brief vector_to_kernel Computes part of the kernel distance
    /// \param points1
    /// \param points2
    /// \param sigma Penalty when 2 points are mis-aligned
    /// \return Partial kernel distance score
    ///
    static double vector_to_kernel(vector<Point2d>& points1, vector<Point2d>& points2, double sigma);
};

#endif // KERNELDIST_H
