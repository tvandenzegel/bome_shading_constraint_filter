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

#include "normal_computer_grad.h"

#include <opencv2/core/core.hpp>

//==============================================================================================
void normal_computer_grad::compute_normals(
							const int width,
							const int height,
							const double focal_length,
							const double* input_y_image_ptr,
							double* output_nx_image_ptr,
							double* output_ny_image_ptr,
							double* output_nz_image_ptr,
							const int selection_left, const int selection_right, 
							const int selection_top, const int selection_bottom
							)
{
	int w_m1 = width - 1;
	int h_m1 = height - 1;
	const double inv_focal_length = 1.0f / focal_length;

	for (int y = selection_top; y <= selection_bottom; ++y)
	{
		for (int x = selection_left; x <= selection_right; ++x)
		{
			int i  = x + y * width;

			if ((input_y_image_ptr[i] > 0)
				&&
				((x != w_m1) && (input_y_image_ptr[i+1]>0))
				&&
				((y != h_m1) && (input_y_image_ptr[i+width]>0)))
			{
				double delta_x = input_y_image_ptr[i+1] - input_y_image_ptr[i];
				double delta_y = input_y_image_ptr[i+width] - input_y_image_ptr[i];
					
				//  a 1st approximation of the normal based on the image gradient of the depth image
				cv::Point3d n;
				n.x = delta_x;
				n.z = -delta_y;
				n.y = input_y_image_ptr[i] * -inv_focal_length;
				double nl = std::sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
				n.x /= nl;
				n.y /= nl;
				n.z /= nl;

				output_nx_image_ptr[i] = n.x;
				output_ny_image_ptr[i] = n.y;
				output_nz_image_ptr[i] = n.z;

			}
			else
			{
				output_nx_image_ptr[i] = 0.;
				output_ny_image_ptr[i] = -1.;
				output_nz_image_ptr[i] = 0.;
			}
			
		}
	}

}
//==============================================================================================
