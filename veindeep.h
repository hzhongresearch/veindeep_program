#ifndef VEINDEEP_H
#define VEINDEEP_H

#include <limits>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "kerneldist.h"

using namespace cv;
using namespace std;

///
/// \brief The VeinDeep class
/// Computes the VeinDeep distance which measures the degree of similarity between 2 binary images, a smaller value indicates greater similarity.
///
class VeinDeep
{
public:
    ///
    /// \brief score Computes VeinDeep distance
    /// \param reference
    /// \param test
    /// \param sigma Penalty when 2 points are mis-aligned
    /// \return
    ///
    static double score(Mat& reference, Mat& test, double sigma);
protected:
    ///
    /// \brief gen_pattern Line thinning algorithm to extract key points
    /// \param input
    /// \param output Vein pattern after thinning
    ///
    static void gen_pattern(Mat& input, Mat& output);
};

#endif // VEINDEEP_H
