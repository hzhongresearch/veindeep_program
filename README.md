# Description of VeinDeep program

VeinDeep is a system which uses an infrared (IR) depth sensor to extract vein patterns from a person’s hand. The idea is in the future smartphones will be equipped with Kinect V2 like IR depth sensors. Such sensors can be used to identify the smartphone owner using vein patterns and provide a way to unlock the phone. This project was created during my PhD. The associated research paper was presented at PerCom 2017.

This is the code for VeinDeep.

If you use this work please consider citing our paper.

```
@inproceedings{zhong2017veindeep,
  title={VeinDeep: Smartphone unlock using vein patterns},
  author={Zhong, Henry and Kanhere, Salil S and Chou, Chun Tung},
  booktitle={Pervasive Computing and Communications (PerCom), 2017 IEEE International Conference on},
  pages={2--10},
  year={2017},
  organization={IEEE}
}
```

The code, data and paper can be downloaded from the following page.

```
https://hzhongresearch.github.io/
```

### Licence
Copyright 2017 HENRY ZHONG. The code is released under MIT licence. See LICENCE.txt for details.

### Usage instructions
This version of the code has been tested under Debian Linux. Before use, first install prerequisite packages:

```
sudo apt update
sudo apt install build-essential cmake git libopencv-dev pkg-config
```

Download and compile.

```
git clone https://github.com/hzhongresearch/veindeep_program.git
cd veindeep_program
cmake .
make
```

Run with included sample test data using the following commands.

```
./VeinDeep_exe person_a_l_01_ir_raw.png person_a_l_01_depth_raw.png person_a_l_02_ir_raw.png person_a_l_02_depth_raw.png 217 296 173 252 500 600 0.7 0.8 11 9 4 90 90 180 3.657 5 5
./VeinDeep_exe person_a_l_01_ir_raw.png person_a_l_01_depth_raw.png person_b_r_01_ir_raw.png person_b_r_01_depth_raw.png 217 296 173 252 500 600 0.7 0.8 11 9 4 90 90 180 3.657 5 5
```

### Explanantion of parameters
```
./VeinDeep_exe <Reference IR input> <Reference depth input> <Test IR input> <Test depth input> <u1> <u2> <y1> <y2> <z1_min_depth> <z2_max_depth> <a_scale_x> <b_scale_y> <o_kernel_size> <g_min_vein_size> <connected> <alpha_x_angle_default> <beta_y_angle_default> <gamma_z_angle_default> <focal length> <kdist_sigma> <v_sigma>
```

1. ```<Reference IR input> <Reference depth input>``` : location of the reference IR and depth images.
2. ```<Test IR input> <Test depth input>``` : location of the test IR and depth images.
3. ```<u1> <u2> <y1> <y2>``` : Coordinates of the bounding box, pixels outside are discarded. Recommend keeping 217 296 173 252 to get centred image.
4. ```<z1_min_depth> <z2_max_depth>``` : Corresponds with params z1 z2 in our paper. Pixels whose depth value are outside this range are discarded. Recommend keeping between 500 600 for clean background separation.
5. ```<a_scale_x> <b_scale_y>``` : Corresponds with params a b in our paper. The vein pattern gets scaled and trimmed by the amount specified in the x and y axis.
6. ```<o_kernel_size>``` : Corresponds with param o in our paper. This is the adaptive threshold filter size. If future higher-res sensors are available increase value.
7. ```<g_min_vein_size>``` : Corresponds with param g in our paper. Removes vein patterns which consist of fewer than this number of pixels. Given higher-res images increase value. Given lower-noise sensor decrease value.
8. ```<connected>``` : The level of connectedness of a pixel. Recommend leave at 4 for 4 connected pixels. See [https://en.wikipedia.org/wiki/Pixel_connectivity](https://en.wikipedia.org/wiki/Pixel_connectivity) for explanation.
9. ```<focal length>``` : The Kinect V2 lens focal length in mm. Leave it at 3.657.
10. ```<alpha_x_angle_default> <beta_y_angle_default> <gamma_z_angle_default>``` : Corresponds with params alpha beta gamma in our paper. The pixels are rotated until the their mean values on each axis is equal to set values. Recommend keeping 90 90 180 so the vein pattern stays nice and flat. 
11. ```<kdist_sigma> <v_sigma>``` : kernel distance penalty for Kernel and VeinDeep distance algorithms. 5 5 seem to give very good results.
