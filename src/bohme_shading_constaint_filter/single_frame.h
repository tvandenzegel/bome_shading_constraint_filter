/* Copyright 2015 Toon Van den Zegel. All Rights Reserved.								*/
/*                                                                                      */
/* This file is part of bohme_shading_constraint_filter.								*/
/* 																						*/
/* bohme_shading_constraint_filter is free software :									*/
/* you can redistribute it and / or modify												*/
/* it under the terms of the GNU General Public License as published by					*/
/* the Free Software Foundation, either version 3 of the License, or					*/
/* (at your option) any later version.													*/
/* 																						*/
/* bohme_shading_constraint_filter is distributed in the hope that it will be useful,	*/
/* but WITHOUT ANY WARRANTY; without even the implied warranty of						*/
/* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the							*/
/* GNU General Public License for more details.											*/
/* 																						*/
/* You should have received a copy of the GNU General Public License					*/
/* along with bohme_shading_constraint_filter.                                          */
/* If not, see <http://www.gnu.org/licenses/>.					                     	*/

#pragma once

#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

//#include <sk_camera_geometry/pinhole_model.h>

struct camera_intrinsics
{
	int width;
	int height;
	double fx; // focal length expressed in pixel units
	double fy; // focal length expressed in pixel units
	double cx; // central point x
	double cy; // central point y
	double k1; // the first radial distortion coefficient
	double k2; // the second radial distortion coefficient
	double k3; // the third radial distortion coefficient
	double p1; // the first tangential distortion coefficient
	double p2; // the second tangential distortion coefficient
};

struct single_frame
{
	cv::Mat image_ideal_depth;
	cv::Mat image_ideal_intensity;

	cv::Mat image_noisy_depth;
	cv::Mat image_noisy_intensity;

	cv::Mat image_working_depth;
	cv::Mat image_rendered_intensity;

	double global_albedo;

	double palette_depth_min_max[2];
};

class single_frame_loader
{
public:
	static bool load(std::string directory, single_frame& out_frame, camera_intrinsics& intrinsics, const double scale = 1.0f);

private:
	static bool load_from_tiff(const char *fileName, cv::Mat &out_image);
	static bool import_albedo(const std::string filename, double& out_albedo);
	static bool import_camera_parameters_txt(const std::string filename, camera_intrinsics& out_intrinsics);
	static bool import_palette_min_max(const std::string filename, double& out_palette_min, double& out_palette_max);
};