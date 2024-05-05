#ifndef HAUSDORFF_H
#define HAUSDORFF_H

#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

///
/// \brief The hausdorff class
/// Computes the Hausdorff distance which measures the degree of similarity between 2 binary images, a smaller value indicates greater similarity.
///
class Hausdorff
{
public:
    ///
    /// \brief score Computes Hausdorff distance
    /// \param reference
    /// \param test
    /// \return
    ///
	static double score(Mat& reference, Mat& test);
protected:
    ///
    /// \brief matrix_to_vector Stores the coordinates of all non-zero pixels in binary image
    /// \param input
    /// \param stored_points Stored vector of non-zero pixels
    ///
    static void matrix_to_vector(Mat& input, vector<Point2d>& stored_points);
    ///
    /// \brief vector_to_dist Computes the Euclidean distances between 2 sets of points
    /// \param points1
    /// \param points2
    /// \param distances Matrix which stores the distances
    ///
	static void vector_to_dist(vector<Point2d>& points1, vector<Point2d>& points2, Mat& distances);
    ///
    /// \brief hausdorffdist Computes the Hausdorff distance given a matrix of Euclidean distances
    /// \param distances
    /// \return Hausdorff distance
    ///
    static double hausdorffdist(Mat& distances);
};

#endif // HAUSDORFF_H
