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

#include <opencv2/core/core.hpp>

#include "single_frame.h"


struct bohme_shading_constraint_filter_parameters
{
	bohme_shading_constraint_filter_parameters();

	double global_albedo;

	double weight_shading_constraint_term;
	double weight_depth_regularization_term;
	double weight_normal_shape_prior_term;

	double gradient_delta;
};

class bohme_shading_constraint_filter
{
	struct pixel_back_up_storage
	{
		static const int patch_size = 3;

		pixel_back_up_storage()
		{
			patch_nx.create(patch_size, patch_size, CV_64FC1);
			patch_ny.create(patch_size, patch_size, CV_64FC1);
			patch_nz.create(patch_size, patch_size, CV_64FC1);
		}

		double value_x;
		double value_y;
		double value_z;
		cv::Mat patch_nx;
		cv::Mat patch_ny;
		cv::Mat patch_nz;
	};
public:
	bohme_shading_constraint_filter(
		const cv::Mat& image_depth_original, 
		const cv::Mat& image_intensity_original, 
		bohme_shading_constraint_filter_parameters& initial_parameters,
		camera_intrinsics& intr_params);
	bohme_shading_constraint_filter & operator=(const bohme_shading_constraint_filter &) {}
	~bohme_shading_constraint_filter();

	// 
	// Estimate albedo
	// - it just returns the estimate, it doesn't change the global_albedo parameter!
	double estimate_current_albedo(const cv::Mat& image_input_depth, const cv::Mat& image_input_intensity);

	// 
	// Energy function evaluation
	double calculate_energy(const cv::Mat& image_input_depth, cv::Mat* p_image_energy_per_pixel = nullptr /* stores the energy per pixel into this image */);

	//
	// Gradient evaluation
	void calculate_gradient(const cv::Mat& image_input_depth, cv::Mat& image_gradient);

	//
	//  Render the intensity image
	void render_intensity_image(cv::Mat& out_image_intensity);
		
	//
	// Accessors
	bohme_shading_constraint_filter_parameters& get_parameters() { return m_parameters; }
	int get_width() const { return m_image_depth_original.cols; }
	int get_height() const { return m_image_depth_original.rows; }
	int get_dimensions() const { return m_image_depth_original.rows * m_image_depth_original.cols; }

private:
	//
	// xz calculation from a depth image
	void regenerate_xz(const cv::Mat& intput_image_depth, cv::Mat& image_x, cv::Mat& image_y, cv::Mat& image_z, 
		const int selection_left, const int selection_right, 
		const int selection_top, const int selection_bottom);

	//
	// Actual energy calculation
	double calculate_energy(
		int selection_left, int selection_right, 
		int selection_top, int selection_bottom,
		cv::Mat* p_image_energy_per_pixel = nullptr);
	
	//
	// Pixel storage
	// - in this way we can modify 1 depth value and calculate the energy with this depth value (in the gradient calculation),
	//   and with these 2 functions we can just go back to the original state!
	void store_pixel(const int x, const int y);
	void restore_pixel(const int x, const int y);

	void create_roi(const int x, const int y, const int sel_width, const int w, const int h, cv::Point2i* roi);
private:
	cv::Point3d cv_normalize(const cv::Point3d &v)
	{
		double l = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
		return cv::Point3d(v.x / l, v.y / l, v.z / l);
	}

private:
	const cv::Mat m_image_depth_original;
	const cv::Mat m_image_intensity_original;
	bohme_shading_constraint_filter_parameters& m_parameters;
	pixel_back_up_storage m_pixel_back_up_storage;
	camera_intrinsics m_intrinsics;

public:
	// convention:
	// - x: left right
	// - y: depth
	// - z: up down
	cv::Mat m_image_x, m_image_y, m_image_z;
	cv::Mat m_image_nx, m_image_ny, m_image_nz;
	cv::Mat m_image_energy_per_pixel;
};