#ifndef VEINFEATURE_H
#define VEINFEATURE_H

#include <iostream>
#include <limits>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

///
/// \brief The VeinFeature class
/// Applies a series of filters to preprocess the vein patterns for comparison
///
class VeinFeature
{
public:
    VeinFeature(int u1, int u2, int v1, int v2, ushort z1, ushort z2, double scale_x, double scale_y, int kernel_size, int min_vein_size, int connected, ushort x_angle_default, ushort y_angle_default, ushort z_angle_default, float f);
    ///
    /// \brief get_vein_pattern Applies a series of filters
    /// \param depth Input depth map of veins
    /// \param ir Input infrared (IR) image of veins
    /// \param output Final processed IR image of veins
    ///
    void get_vein_pattern(Mat& depth, Mat& ir, Mat& output);
    static const ushort max_ushort = numeric_limits<ushort>::max();
protected:
    Rect ref_roi;
    ushort z1, z2;
    double scale_x, scale_y;
    int kernel_size, min_vein_size, connected;
    ushort x_angle_default, y_angle_default, z_angle_default;
    float f;
protected:
    static void get_silhouette(Mat& depth, Mat& silhouette, ushort sub_value, ushort max_depth, ushort min_depth);
    static void clean_silhouette(Mat& silhouette, Mat& silhouette_cleaned, ushort sub_value, int connected);
    static void shrink_silhouette(Mat& silhouette, Mat& silhouette_scaled, double scaled_x, double scale_y);

    static void get_x_y_angles(Mat& depth, Mat& x_angle, Mat& y_angle);
    static void get_mean_x_y_angles(Mat& silhouette, Mat& x_angle, Mat& y_angle, ushort& x_angle_mean, ushort& y_angle_mean);
    static void get_z_angles(Mat& silhouette, Mat& z_angles);
    static void get_mean_z_angles(Mat& gradients, ushort& z_angle_mean);
    static void get_mean_depth(Mat& depth, Mat& silhouette_scaled, ushort& mean_depth);

    static void get_veins(Mat& ir, Mat& veins, ushort sub_value, int kernel_size);
    static void mask_veins(Mat& silhouette, Mat& veins, Mat& veins_masked);
    static void flatten_veins_3D(Mat& depth, Mat& veins, Mat& veins_flattened, ushort x_angle_mean, ushort y_angle_mean, ushort z_angle_mean, ushort x_angle_default, ushort y_angle_default, ushort z_angle_default, float fx, float fy, ushort sub_value);
    static void gaps(Mat& veins_fill, Mat& kernel, Mat& veins_cleaned, float summed, ushort sub_value);
    static void clean_veins(Mat& veins, Mat& veins_cleaned, int connected, int min_vein_size, ushort sub_value);
    static void trim_veins(Mat& veins, Mat& veins_trimmed);

    static void get_angle_matrix(Mat& source, Mat& kernel, Mat& derivative, Mat& angle);
    static ushort mean_with_mask(Mat& mask, Mat& values);
    static void rotate_matrix(float x_rotate, float y_rotate, float z_rotate, Mat& rotate);
    static void uvz_to_xyz(float u, float v, float z, float fx, float fy, float cx, float cy, Mat& x_y_z);
    static void xyz_to_uvz(float x, float y, float z, float fx, float fy, float cx, float cy, Mat& u_v_z);

    static void centre_veins(Mat& veins, Mat& silhouette, Mat& veins_centred);
};

#endif // VEINFEATURE_H
