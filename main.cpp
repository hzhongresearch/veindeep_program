#include <iostream>
#include <chrono>
#include <cstdlib>
#include <string>
#include <opencv2/core.hpp>

#include "hausdorff.h"
#include "kerneldist.h"
#include "veindeep.h"
#include "veinfeature.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    if (argc != 22)
    {
        cout << "Usage: \n./VeinDeep_exe <Reference IR input> <Reference depth input> <Test IR input> <Test depth input> <u1> <u2> <y1> <y2> ";
        cout << "<z1_min_depth> <z2_max_depth> <a_scale_x> <b_scale_y> <o_kernel_size> <g_min_vein_size> <connected>";
        cout << "<alpha_x_angle_default> <beta_y_angle_default> <gamma_z_angle_default> <kdist_sigma> <v_sigma>" << endl;
        return EXIT_SUCCESS;
    }

    // Reference image
    // Expected format is 16bit unsigned representing an IR image and depth map
    // This image should be the raw unprocessed output from a Kinect V2
    // The image should be centred on the back of a subject's enclosed fist with vein visible
    string ref_ir_path = string(argv[1]);
    Mat ref_ir = imread(ref_ir_path, IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);
    string ref_depth_path = string(argv[2]);
    Mat ref_depth = imread(ref_depth_path, IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);

    // Test image
    string test_ir_path = string(argv[3]);
    Mat test_ir = imread(test_ir_path, IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);
    string test_depth_path = string(argv[4]);
    Mat test_depth = imread(test_depth_path, IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);

    // ROI
    int u1 = atoi(argv[5]);
    int u2 = atoi(argv[6]);
    int v1 = atoi(argv[7]);
    int v2 = atoi(argv[8]);
    // Distance threshold parameters
    ushort z1_min_depth = ushort(atoi(argv[9]));
    ushort z2_max_depth = ushort(atoi(argv[10]));
    // Silhouette scale parameters
    double a_scale_x = atof(argv[11]);
    double b_scale_y = atof(argv[12]);
    // Vein kernel size parameter
    int o_kernel_size = atoi(argv[13]);
    // Remove vein segments smaller than the following
    int g_min_vein_size = atoi(argv[14]);
    // Segment variables typically 8
    int connected = atoi(argv[15]);
    // Rotation variables
    ushort alpha_x_angle_default = ushort(atoi(argv[16]));
    ushort beta_y_angle_default = ushort(atoi(argv[17]));
    ushort gamma_z_angle_default = ushort(atoi(argv[18]));
    // f is Kinect V2 lens focal length in mm typically 3.657
    float f = float(atof(argv[19]));

    // Kernel distance penalty
    double kdist_sigma = atof(argv[20]);

    // VeinDeep distance penalty
    double v_sigma = atof(argv[21]);

    // Initialise
    VeinFeature myveinfeature = VeinFeature(u1, u2, v1, v2, z1_min_depth, z2_max_depth, a_scale_x, b_scale_y, o_kernel_size, g_min_vein_size, connected, alpha_x_angle_default, beta_y_angle_default, gamma_z_angle_default, f);

    // Filter test and reference images
    Mat ref_pattern;
    myveinfeature.get_vein_pattern(ref_depth, ref_ir, ref_pattern);
    Mat test_pattern;
    myveinfeature.get_vein_pattern(test_depth, test_ir, test_pattern);

    // Hausdorff distance
    chrono::high_resolution_clock::time_point start1 = chrono::high_resolution_clock::now();
    double hausdorff_score = Hausdorff::score(ref_pattern, test_pattern);
    chrono::high_resolution_clock::time_point end1 = chrono::high_resolution_clock::now();
    auto duration1 = chrono::duration_cast<chrono::microseconds>( end1 - start1 ).count();
    // Kernel distance
    chrono::high_resolution_clock::time_point start2 = chrono::high_resolution_clock::now();
    double kerneldist_score = Kerneldist::score(ref_pattern, test_pattern, kdist_sigma);
    chrono::high_resolution_clock::time_point end2 = chrono::high_resolution_clock::now();
    auto duration2 = chrono::duration_cast<chrono::microseconds>( end2 - start2 ).count();
    // VeinDeep distance
    chrono::high_resolution_clock::time_point start3 = chrono::high_resolution_clock::now();
    double veindeep_score = VeinDeep::score(ref_pattern, test_pattern, v_sigma);
    chrono::high_resolution_clock::time_point end3 = chrono::high_resolution_clock::now();
    auto duration3 = chrono::duration_cast<chrono::microseconds>( end3 - start3 ).count();

    // Show the similarity scores and time taken for each algorithm
    cout << "Hausdorff distance: " << hausdorff_score << endl;
    cout << "Time: " << duration1 << " microseconds" << endl;
    cout << "Kernel distance: " << kerneldist_score << endl;
    cout << "Time: " << duration2 << " microseconds" << endl;
    cout << "VeinDeep distance: " << veindeep_score << endl;
    cout << "Time: " << duration3 << " microseconds" << endl;

    return EXIT_SUCCESS;
}
