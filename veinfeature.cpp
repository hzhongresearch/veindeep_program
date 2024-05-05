#include "veinfeature.h"

VeinFeature::VeinFeature(int u1, int u2, int v1, int v2, ushort z1, ushort z2, double scale_x, double scale_y, int kernel_size, int min_vein_size, int connected, ushort x_angle_default, ushort y_angle_default, ushort z_angle_default, float f)
{
    int roi_width = u2 - u1 + 1;
    int roi_height = v2 - v1 + 1;
    this->ref_roi = Rect(u1, v1, roi_width, roi_height);
    this->z2 = z2;
    this->z1 = z1;
    this->scale_x = scale_x;
    this->scale_y = scale_y;
    this->kernel_size = kernel_size;
    this->min_vein_size = min_vein_size;
    this->connected = connected;
    this->x_angle_default = x_angle_default;
    this->y_angle_default = y_angle_default;
    this->z_angle_default = z_angle_default;
    this->f = f;
}

void VeinFeature::get_vein_pattern(Mat& depth, Mat& ir, Mat& output)
{
    Mat ref_ir = ir(this->ref_roi).clone();
    Mat ref_depth = depth(this->ref_roi).clone();

    /**
     * Silhouette
     */
    Mat silhouette, silhouette_z_alignment, silhouette_scaled, silhouette_cleaned;
    VeinFeature::get_silhouette(ref_depth, silhouette, VeinFeature::max_ushort, this->z2, this->z1);
    VeinFeature::shrink_silhouette(silhouette, silhouette_scaled, this->scale_x, this->scale_y);
    VeinFeature::clean_silhouette(silhouette_scaled, silhouette_cleaned, VeinFeature::max_ushort, this->connected);
    VeinFeature::clean_silhouette(silhouette, silhouette_z_alignment, VeinFeature::max_ushort, this->connected);

    /**
     * Alignment
     */
    Mat x_angle, y_angle, z_angle;
    ushort x_angle_mean, y_angle_mean, z_angle_mean, depth_mean;
    VeinFeature::get_x_y_angles(ref_depth, x_angle, y_angle);
    VeinFeature::get_mean_x_y_angles(silhouette_cleaned, x_angle, y_angle, x_angle_mean, y_angle_mean);
    VeinFeature::get_z_angles(silhouette_z_alignment, z_angle);
    VeinFeature::get_mean_z_angles(z_angle, z_angle_mean);
    VeinFeature::get_mean_depth(ref_depth, silhouette_cleaned, depth_mean);

    /**
     * Veins
     */
    Mat veins, veins_masked, veins_flattened, veins_cleaned, veins_trimmed;
    VeinFeature::get_veins(ref_ir, veins, VeinFeature::max_ushort, this->kernel_size);
    VeinFeature::mask_veins(silhouette_cleaned, veins, veins_masked);
    VeinFeature::flatten_veins_3D(ref_depth, veins_masked, veins_flattened, x_angle_mean, y_angle_mean, z_angle_mean, this->x_angle_default, this->y_angle_default, this->z_angle_default, this->f, this->f, VeinFeature::max_ushort);
    VeinFeature::clean_veins(veins_flattened, veins_cleaned, this->connected, this->min_vein_size, VeinFeature::max_ushort);
    VeinFeature::trim_veins(veins_cleaned, veins_trimmed);

    // Output processed vein pattern
    output = veins_trimmed.clone();
}

void VeinFeature::get_silhouette(Mat& depth, Mat& silhouette, ushort sub_value, ushort max_depth, ushort min_depth)
{
    // Get silhouette
    Mat depth_temp, silhouette_temp;
    depth.convertTo(depth_temp, CV_32FC1);
    // Remove values above upper bound
    threshold(depth_temp, silhouette_temp, double(max_depth), double(sub_value), THRESH_TOZERO_INV);
    // Remove values equal to or below lower bound
    threshold(silhouette_temp, silhouette_temp, double(min_depth), double(sub_value), THRESH_BINARY);
    silhouette_temp.convertTo(silhouette, CV_16UC1);
}

void VeinFeature::clean_silhouette(Mat& silhouette, Mat& silhouette_cleaned, ushort sub_value, int connected)
{
    // Find connected components
    Mat silhouette_temp;
    silhouette.convertTo(silhouette_temp, CV_8UC1, double(1 / 256.0));
    Mat labels, stats, centroids;
    connectedComponentsWithStats(silhouette_temp, labels, stats, centroids, connected, CV_16UC1);

    // Detect largest non-zero segment
    int largest = 0, largest_index = 0;
    for (int i = 1; i < stats.rows; ++i)
    {
        int current_size = stats.at<int>(i, 4);
        if (current_size >= largest)
        {
            largest = current_size;
            largest_index = i;
        }
    }

    // Remove all but the largest non-zero segment
    silhouette_cleaned = Mat::zeros(silhouette.size(), CV_16UC1);
    for (int i = 0; i < labels.rows; ++i)
    {
        for (int j = 0; j < labels.cols; ++j)
        {
            ushort label_value = labels.at<ushort>(i, j);
            if (label_value == largest_index)
            {
                silhouette_cleaned.at<ushort>(i, j) = sub_value;
            }
        }
    }
}

void VeinFeature::shrink_silhouette(Mat& silhouette, Mat& silhouette_scaled, double scaled_x, double scale_y)
{
    // Find silhouette centroid
    Moments m1 = moments(silhouette, false);
    Point p1 = Point(cvFloor(m1.m10/m1.m00), cvFloor(m1.m01/m1.m00));
    if (m1.m00 == 0) { p1 = Point(0, 0); }

    // Shrink silhouette
    Mat scale_down;
    Size scale1 = Size(cvFloor(silhouette.cols * scaled_x), cvFloor(silhouette.rows * scale_y));
    resize(silhouette, scale_down, scale1, 0, 0, INTER_NEAREST);

    // Find small silhouette centroid
    Moments m2 = moments(scale_down, false);
    Point p2 = Point(cvFloor(m2.m10/m2.m00), cvFloor(m2.m01/m2.m00));
    if (m2.m00 == 0) { p2 = Point(0, 0); }

    // Align small silhouette centroid to original silhouette centroid
    silhouette_scaled = Mat::zeros(silhouette.size(), CV_16UC1);
    int left = p1.x - p2.x;
    int top = p1.y - p2.y;

    Rect location = Rect(left, top, scale_down.cols, scale_down.rows);
    scale_down.copyTo(silhouette_scaled(location));
}

void VeinFeature::get_veins(Mat& ir, Mat& veins, ushort sub_value, int kernel_size)
{
    // Filter veins
    Mat ir_temp;
    ir.convertTo(ir_temp, CV_8UC1, double(1 / 256.0));
    adaptiveThreshold(ir_temp, veins, double(sub_value), ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, kernel_size, 0);
    veins.convertTo(veins, CV_16UC1, 256);
}

void VeinFeature::mask_veins(Mat& silhouette, Mat& veins, Mat& veins_masked)
{
    // Crop out the veins using silhouette as mask
    Mat silhouette_temp;
    silhouette.convertTo(silhouette_temp, CV_8UC1, double(1 / 256.0));
    veins.copyTo(veins_masked, silhouette_temp);
}

void VeinFeature::gaps(Mat& veins_fill, Mat& kernel, Mat& veins_cleaned, float summed, ushort sub_value)
{
    filter2D(veins_fill, veins_fill, CV_32FC1, kernel, Point(-1, -1), 0, BORDER_DEFAULT);
    for (int i = 0; i < veins_fill.rows; ++i)
    {
        for (int j = 0; j < veins_fill.cols; ++j)
        {
            float fill = veins_fill.at<float>(i, j);
            if (fill > summed)
            {
                veins_cleaned.at<ushort>(i, j) = sub_value;
            }
        }
    }
}

void VeinFeature::clean_veins(Mat& veins, Mat& veins_cleaned, int connected, int min_vein_size, ushort sub_value)
{
    // Find connected components
    Mat veins_temp;
    veins.convertTo(veins_temp, CV_8UC1, double(1 / 256.0));
    Mat labels, stats, centroids;
    connectedComponentsWithStats(veins_temp, labels, stats, centroids, connected, CV_16UC1);

    // Remove every segment below limit
    veins_cleaned = Mat::zeros(veins.size(), CV_16UC1);
    for (int i = 0; i < labels.rows; ++i)
    {
        for (int j = 0; j < labels.cols; ++j)
        {
            int label_value = int(labels.at<ushort>(i, j));
            int label_occupancy = stats.at<int>(label_value, 4);
            if (label_occupancy > min_vein_size && label_value > 0)
            {
                veins_cleaned.at<ushort>(i, j) = sub_value;
            }
        }
    }

    // Fill in small gaps
    Mat veins_fill;
    veins_cleaned.convertTo(veins_fill, CV_32FC1, 1 / float(sub_value));

    // First filter
    float kernel_array1[] = {1, 1, 1,
                            1, 0, 1,
                            1, 1, 1};
    Mat kernel1 = Mat(Size(3, 3), CV_32FC1, kernel_array1);
    VeinFeature::gaps(veins_fill, kernel1, veins_cleaned, 4, sub_value);
}

void VeinFeature::centre_veins(Mat& veins, Mat& silhouette, Mat& veins_centred)
{
    // Get centroid of vein area
    Moments m1 = moments(silhouette, false);
    Point p1 = Point(cvFloor(m1.m10/m1.m00), cvFloor(m1.m01/m1.m00));
    if (m1.m00 == 0) { p1 = Point(0, 0); }
    // Get centre pixel of image
    int x_axis = cvFloor(veins.cols / 2);
    int y_axis = cvFloor(veins.rows / 2);
    Point p2 = Point(x_axis, y_axis);

    // Get pixels to shift to centre
    int left = p1.x - p2.x;
    int top = p1.y - p2.y;

    int region1_x = left, region1_y = top;
    int region2_x = 0, region2_y = 0;
    if (left < 0) { region1_x = 0; region2_x = -left; }
    if (top < 0) { region1_y = 0; region2_y = -top; }

    // Perform copy
    veins_centred = Mat::zeros(veins.size(), CV_16UC1);
    Rect region1 = Rect(region1_x, region1_y, veins.cols - abs(left), veins.rows - abs(top));
    Rect region2 = Rect(region2_x, region2_y, veins.cols - abs(left), veins.rows - abs(top));
    veins(region1).copyTo(veins_centred(region2));
}

void VeinFeature::get_angle_matrix(Mat& source, Mat& kernel, Mat& derivative, Mat& angle)
{
    filter2D(source, derivative, CV_32FC1, kernel, Point(-1, -1), 0, BORDER_DEFAULT);
    angle = Mat::zeros(derivative.size(), CV_32FC1);
    for (int i = 0; i < derivative.rows; ++i)
    {
        for (int j = 0; j < derivative.cols; ++j)
        {
            // Calculations for angle
            float D_j_i = derivative.at<float>(i, j);
            float angle_j_i = fastAtan2(D_j_i, 1);
            if (angle_j_i >= 180)
            {
                angle_j_i -= 180;
            }
            // Save angles
            angle.at<float>(i, j) = angle_j_i;
        }
    }
    angle.convertTo(angle, CV_16UC1);
}

void VeinFeature::get_x_y_angles(Mat& depth, Mat& x_angle, Mat& y_angle)
{
    // Apply filters to calculate derivatives
    Mat Dx, Dy, depth_temp;
    depth.convertTo(depth_temp, CV_32FC1);
    float kernel_array[] = {-0.5, 0, 0.5};
    Mat kernel_dx = Mat(Size(3, 1), CV_32FC1, kernel_array);
    Mat kernel_dy = Mat(Size(1, 3), CV_32FC1, kernel_array);
    // Calculate x derivative
    get_angle_matrix(depth_temp, kernel_dx, Dx, x_angle);
    // Calculate y derivative
    get_angle_matrix(depth_temp, kernel_dy, Dy, y_angle);
}

ushort VeinFeature::mean_with_mask(Mat& mask, Mat& values)
{
    unsigned long accumulate = 0;
    unsigned int size_of_mask = 0;
    for (int i = 0; i < mask.rows; ++i)
    {
        for (int j = 0; j < mask.cols; ++j)
        {
            ushort mask_value = mask.at<ushort>(i, j);
            if (mask_value > 0)
            {
                ushort angle = values.at<ushort>(i, j);
                accumulate += angle;
                ++size_of_mask;
            }
        }
    }

    ushort mean = ushort(accumulate/size_of_mask);
    return mean;
}

void VeinFeature::get_mean_x_y_angles(Mat& silhouette, Mat& x_angle, Mat& y_angle, ushort& x_angle_mean, ushort& y_angle_mean)
{
    x_angle_mean = VeinFeature::mean_with_mask(silhouette, x_angle);
    y_angle_mean = VeinFeature::mean_with_mask(silhouette, y_angle);
}

void VeinFeature::get_z_angles(Mat& silhouette, Mat& z_angles)
{
    // Find image derivative
    Mat Dx, Dy;
    float kernel_array[] = {-0.5, 0, 0.5};
    Mat kernel_dx = Mat(Size(3, 1), CV_32FC1, kernel_array);
    Mat kernel_dy = Mat(Size(1, 3), CV_32FC1, kernel_array);
    filter2D(silhouette, Dx, CV_32FC1, kernel_dx, Point(-1, -1), 0, BORDER_DEFAULT);
    filter2D(silhouette, Dy, CV_32FC1, kernel_dy, Point(-1, -1), 0, BORDER_DEFAULT);

    // Use derivative to find the angle of silhouette outline
    Mat magnitude;
    cartToPolar(Dx, Dy, magnitude, z_angles, true);
    z_angles.convertTo(z_angles, CV_16UC1);
}

void VeinFeature::get_mean_z_angles(Mat& gradients, ushort& z_angle_mean)
{
    z_angle_mean = VeinFeature::mean_with_mask(gradients, gradients);
}

void VeinFeature::get_mean_depth(Mat& depth, Mat& silhouette, ushort& depth_mean)
{
    Mat depth_temp, silhouette_temp;
    silhouette.convertTo(silhouette_temp, CV_8UC1, double(1 / 256.0));
    depth.copyTo(depth_temp, silhouette_temp);
    int non_zero = countNonZero(depth_temp);
    double sum_of_matrix = sum(depth_temp)[0];
    depth_mean = ushort(sum_of_matrix / non_zero);
}

void VeinFeature::rotate_matrix(float x_rot, float y_rot, float z_rot, Mat& rotate)
{
    float sin_x = float(sin(x_rot * CV_PI / 180));
    float cos_x = float(cos(x_rot * CV_PI / 180));
    float sin_y = float(sin(y_rot * CV_PI / 180));
    float cos_y = float(cos(y_rot * CV_PI / 180));
    float sin_z = float(sin(z_rot * CV_PI / 180));
    float cos_z = float(cos(z_rot * CV_PI / 180));

    // Rotate x
    float x_array[] = {1, 0, 0,
                       0, cos_x, sin_x,
                       0, -sin_x, cos_x};
    Mat x_matrix = Mat(Size(3, 3), CV_32FC1, x_array);
    // Rotate y
    float y_array[] = {cos_y, 0, -sin_y,
                       0, 1, 0,
                       sin_y, 0, cos_y};
    Mat y_matrix = Mat(Size(3, 3), CV_32FC1, y_array);
    // Rotate z
    float z_array[] = {cos_z, sin_z, 0,
                       -sin_z, cos_z, 0,
                       0, 0, 1};
    Mat z_matrix = Mat(Size(3, 3), CV_32FC1, z_array);

    Mat rotate_temp = x_matrix * y_matrix * z_matrix;
    rotate = rotate_temp.clone();
}

void VeinFeature::uvz_to_xyz(float u, float v, float z, float fx, float fy, float cx, float cy, Mat& x_y_z)
{
    float x = (u - cx) * z / fx;
    float y = (v - cy) * z / fy;
    float x_y_z_array[] = {x, y, z};
    Mat x_y_z_temp = Mat(Size(1, 3), CV_32FC1, x_y_z_array);
    x_y_z = x_y_z_temp.clone();
}

void VeinFeature::xyz_to_uvz(float x, float y, float z, float fx, float fy, float cx, float cy, Mat& u_v_z)
{
    float u = (x * fx / z) + cx;
    float v = (y * fy / z) + cy;
    float u_v_z_array[] = {u, v, z};
    Mat u_v_z_temp = Mat(Size(1, 3), CV_32FC1, u_v_z_array);
    u_v_z = u_v_z_temp.clone();
}

void VeinFeature::flatten_veins_3D(Mat& depth, Mat& veins, Mat& veins_flattened, ushort x_angle_mean, ushort y_angle_mean, ushort z_angle_mean, ushort x_angle_default, ushort y_angle_default, ushort z_angle_default, float fx, float fy, ushort sub_value)
{
    float x_rot_dy = float(y_angle_default - y_angle_mean);
    float y_rot_dx = float(x_angle_default - x_angle_mean);
    float z_rot_dz = -float(z_angle_default - z_angle_mean);
    float cx = float(cvFloor(depth.cols / 2));
    float cy = float(cvFloor(depth.rows / 2));
    Mat rotate;
    rotate_matrix(x_rot_dy, y_rot_dx, z_rot_dz, rotate);

    veins_flattened = Mat::zeros(veins.size(), CV_16UC1);
    for (int i = 0; i < veins_flattened.rows; ++i)
    {
        for (int j = 0; j < veins_flattened.cols; ++j)
        {
            ushort current_value = veins.at<ushort>(i, j);
            if (current_value > 0)
            {
                // Viewport coordinates
                float u = float(j);
                float v = float(i);
                float z = float(depth.at<ushort>(i, j));
                float u_v_z_array[] = {u, v, z};
                Mat u_v_z = Mat(Size(1, 3), CV_32FC1, u_v_z_array);

                // Convert to world coordinates and perform rotations
                Mat x_y_z;
                uvz_to_xyz(u, v, z, fx, fy, cx, cy, x_y_z);
                Mat x_y_z_rotate = rotate * x_y_z;
                float x = x_y_z_rotate.at<float>(0, 0);
                float y = x_y_z_rotate.at<float>(1, 0);

                // Convert back to viewport coordinates
                Mat u_v_z_new;
                xyz_to_uvz(x, y, z, fx, fy, cx, cy, u_v_z_new);
                float u_new = u_v_z_new.at<float>(0, 0);
                float v_new = u_v_z_new.at<float>(1, 0);

                // Write pixel if within the depth map
                int coord_x = int(u_new);
                int coord_y = int(v_new);
                if (0 <= coord_x && coord_x < veins_flattened.cols &&
                    0 <= coord_y && coord_y < veins_flattened.rows)
                {
                    veins_flattened.at<ushort>(coord_y, coord_x) = sub_value;
                }
            }
        }
    }
}

void VeinFeature::trim_veins(Mat& veins, Mat& veins_trimmed)
{
    int left = veins.cols - 1, right = 0;
    int top = veins.rows - 1, bottom = 0;

    for (int i = 0; i < veins.rows; ++i)
    {
        for (int j = 0; j < veins.cols; ++j)
        {
            ushort current_value = veins.at<ushort>(i, j);
            if (current_value > 0)
            {
                if (j < left) { left = j; }
                if (j > right) { right = j; }
                if (i < top) { top = i; }
                if (i > bottom) { bottom = i; }
            }
        }
    }

    Rect region = Rect(left, top, right - left + 1, bottom - top + 1);
    if (region.width < 1 || region.height < 1)
    {
        veins_trimmed = Mat::zeros(1, 1, CV_16UC1);
    }
    else
    {
        veins_trimmed = veins(region).clone();
    }
}
