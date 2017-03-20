/* Copyright 2015 Toon Van den Zegel. All Rights Reserved.								*/
/*                                                                                      */
/* This file is part of bohme_shading_constraint_filter.								*/
/* 																						*/
/* bohme_shading_constraint_filter is free software :									*/
/* you sscan redistribute it and / or modify												*/
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

#include "bohme_shading_constraint_filter.h"
#include "normal_computer_grad.h"

bohme_shading_constraint_filter_parameters::bohme_shading_constraint_filter_parameters()
{
	// just some default values
	global_albedo = 0.07;

	weight_shading_constraint_term = 2000.0;
	weight_depth_regularization_term = 50.0;
	weight_normal_shape_prior_term =  1.0;

	gradient_delta = 0.0000001;
}

bohme_shading_constraint_filter::bohme_shading_constraint_filter(const cv::Mat& image_depth_original, const cv::Mat& image_intensity_original, bohme_shading_constraint_filter_parameters& initial_parameters, camera_intrinsics& intr_params) :
	m_image_depth_original(image_depth_original),
	m_image_intensity_original(image_intensity_original),
	m_parameters(initial_parameters),
	m_intrinsics(intr_params)
{
	assert(image_depth_original.rows == image_intensity_original.rows);
	assert(image_depth_original.cols == image_intensity_original.cols);

	m_image_depth_original.copyTo(m_image_y);

	const int rows = m_image_depth_original.rows;
	const int cols = m_image_depth_original.cols;

	m_image_x.create(rows, cols, CV_64FC1);
	m_image_z.create(rows, cols, CV_64FC1);

	m_image_nx.create(rows, cols, CV_64FC1);
	m_image_ny.create(rows, cols, CV_64FC1);
	m_image_nz.create(rows, cols, CV_64FC1);

	m_image_energy_per_pixel.create(rows, cols, CV_64FC1);

	m_image_x = cv::Scalar(0.0);
	m_image_x = cv::Scalar(0.0);
	m_image_z = cv::Scalar(0.0);

	m_image_nx = cv::Scalar(0.0);
	m_image_ny = cv::Scalar(0.0);
	m_image_nz = cv::Scalar(0.0);

	m_image_energy_per_pixel = cv::Scalar(0.0);
}

bohme_shading_constraint_filter::~bohme_shading_constraint_filter()
{

}

double bohme_shading_constraint_filter::estimate_current_albedo(const cv::Mat& image_input_depth, const cv::Mat& image_input_intensity)
{
	const int w = image_input_depth.cols;
	const int h = image_input_depth.rows;

	regenerate_xz(image_input_depth, m_image_x, m_image_y, m_image_z,
		0, w - 1, 
		0, h - 1);

	const double *px = reinterpret_cast<const double*>(m_image_x.data);
	const double *py = reinterpret_cast<const double*>(m_image_y.data);
	const double *pz = reinterpret_cast<const double*>(m_image_z.data);
	const double *pi_ref = reinterpret_cast<const double*>(image_input_intensity.data);

	double max_i = 0.0;
	double estimated_albedo = 1.0;
	for (int y = 0; y < h; ++y)
	{
		for (int x = 0; x < w; ++x)
		{
			int idx = x + y * w;

			// valid pixel?
			if (py[idx] < 0.0) 
				continue;

			// maximum intensity?
			if (pi_ref[idx] < max_i)
				continue;
			
			max_i = pi_ref[idx];

			// calculate albedo
			cv::Point3d v(px[idx], py[idx], pz[idx]);
			double r2 = v.dot(v);
			estimated_albedo = r2 * pi_ref[idx];
		}
	}

	return estimated_albedo;
}

void bohme_shading_constraint_filter::render_intensity_image(cv::Mat& out_image_intensity)
{
	const int w = m_image_y.cols;
	const int h = m_image_y.rows;

	out_image_intensity.create(h,w, CV_64FC1);
	double* pI = reinterpret_cast<double*>(out_image_intensity.data);

	const double *px = reinterpret_cast<const double*>(m_image_x.data);
	const double *py = reinterpret_cast<const double*>(m_image_y.data);
	const double *pz = reinterpret_cast<const double*>(m_image_z.data);

	const double *pnx = reinterpret_cast<const double*>(m_image_nx.data);
	const double *pny = reinterpret_cast<const double*>(m_image_ny.data);
	const double *pnz = reinterpret_cast<const double*>(m_image_nz.data);

	const double global_albedo = m_parameters.global_albedo;

	for (int y = 0; y < h; ++y)
	{
		for (int x = 0; x < w; ++x)
		{
			const int idx = x + y * w;

			if (py[idx] <= 0.0)
			{
				pI[idx] = 0.0;
				continue;
			}

			cv::Point3d vp(px[idx], py[idx], pz[idx]);
			cv::Point3d vn(pnx[idx], pny[idx], pnz[idx]);

			double r2 = vp.dot(vp);
			cv::Point3d l = cv_normalize(vp) * -1.0f;
			double I = vn.dot(l) / r2;
			I *= global_albedo; 
			pI[idx] = I;
	
		}
	}				
}

double bohme_shading_constraint_filter::calculate_energy(const cv::Mat& image_input_depth, cv::Mat* p_image_energy_per_pixel)
{
	assert(image_input_depth.rows == m_image_y.rows);
	assert(image_input_depth.cols == m_image_y.cols);

	const int w = image_input_depth.cols;
	const int h = image_input_depth.rows;

	regenerate_xz(image_input_depth, m_image_x, m_image_y, m_image_z,
			0, w - 1, 
			0, h - 1);

	normal_computer_grad::compute_normals(w,
						h,
						m_intrinsics.fy,
						reinterpret_cast<const double*>(m_image_y.data),
						reinterpret_cast<double*>(m_image_nx.data),
						reinterpret_cast<double*>(m_image_ny.data),
						reinterpret_cast<double*>(m_image_nz.data),
						0, w-1,
						0, h-1);

	return calculate_energy(0, w-1, 0, h-1, p_image_energy_per_pixel);
}

void bohme_shading_constraint_filter::calculate_gradient(const cv::Mat& image_input_depth, cv::Mat& image_gradient)
{
	// gradient calculation with a finite differences approximation
	const double gradient_delta = m_parameters.gradient_delta;

	// first calculate the energy, but we store the energy for each pixel into an image
	calculate_energy(image_input_depth, &m_image_energy_per_pixel);

	// loop through each pixel.
	// - we are going to change the depth value of each pixel a little bit and recalculate the energy again
	// - we only need to calculate the energy of the pixels that are affected by this one depth value change
	double *pgrad = reinterpret_cast<double*>(image_gradient.data);
	
	double* py = reinterpret_cast<double*>(m_image_y.data);
	const double* p_e_per_pixel = reinterpret_cast<const double*>(m_image_energy_per_pixel.data);
	const int w = m_image_depth_original.cols;
	const int h = m_image_depth_original.rows;

	for (int y = 0; y < m_image_depth_original.rows ; ++y)
	{
		for (int x = 0; x < m_image_depth_original.cols; ++x)
		{
			const int idx = x + y * w;

			// valid pixel?
			if (py[idx] <= 0.0)
			{
				pgrad[idx] = 0.0;
				continue;
			}

			store_pixel(x, y);

			// change the depth of the pixel a little bit!
			py[idx] += gradient_delta;

			// recalculate x and z 
			regenerate_xz(m_image_y, m_image_x, m_image_y, m_image_z, x, x, y, y);

			// recalculate normals
			cv::Point2i roi1[2];
			create_roi(x, y, 1, w, h, roi1);

			normal_computer_grad::compute_normals(w,
						h,
						m_intrinsics.fy,
						reinterpret_cast<const double*>(m_image_y.data),
						reinterpret_cast<double*>(m_image_nx.data),
						reinterpret_cast<double*>(m_image_ny.data),
						reinterpret_cast<double*>(m_image_nz.data),
						roi1[0].x, roi1[1].x,
						roi1[0].y, roi1[1].y);
				
			// recalculate energy
			// - only over a patch that is influenced by the modified depth value 
			cv::Point2i roi2[2];
			create_roi(x, y, 2, w, h, roi2);
			double e_new_energy = calculate_energy(roi2[0].x, roi2[1].x, roi2[0].y, roi2[1].y);

			double e_org_energy = 0.f;
			for (int sub_y = roi2[0].y, sub_y_end = roi2[1].y; sub_y <= sub_y_end; ++sub_y)
			{
				const int idxy = sub_y * w;
				for (int sub_x = roi2[0].x, sub_x_end = roi2[1].x; sub_x <= sub_x_end; ++sub_x)
				{
					e_org_energy += p_e_per_pixel[sub_x + idxy];
				}
			}

			// the gradient!!
			pgrad[idx] = e_new_energy - e_org_energy;

			// clean up our modification!
			restore_pixel(x, y);

		}
	}
	
}

double bohme_shading_constraint_filter::calculate_energy(
	int selection_left, int selection_right, 
	int selection_top, int selection_bottom,
	cv::Mat* p_image_energy_per_pixel)
{
	const int w = m_image_depth_original.cols;
	const int h = m_image_depth_original.rows;

	const int w_m1 = w - 1;
	const int h_m1 = h - 1;

	const double *pi_original = reinterpret_cast<const double*>(m_image_intensity_original.data);
	const double *py_original = reinterpret_cast<const double*>(m_image_depth_original.data);

	const double *px = reinterpret_cast<const double*>(m_image_x.data);
	const double *py = reinterpret_cast<const double*>(m_image_y.data);
	const double *pz = reinterpret_cast<const double*>(m_image_z.data);

	const double *pnx = reinterpret_cast<const double*>(m_image_nx.data);
	const double *pny = reinterpret_cast<const double*>(m_image_ny.data);
	const double *pnz = reinterpret_cast<const double*>(m_image_nz.data);

	double* pe_per_pixel = (p_image_energy_per_pixel == nullptr) ? nullptr : reinterpret_cast<double*>(p_image_energy_per_pixel->data);

	const double global_albedo = m_parameters.global_albedo;
	const double weight_shading_constraint_term = m_parameters.weight_shading_constraint_term;
	const double weight_depth_regularization_term = m_parameters.weight_depth_regularization_term;
	const double weight_normal_shape_prior_term = m_parameters.weight_normal_shape_prior_term;

	int neighboursmasks[8];
	neighboursmasks[0] = 1;
	neighboursmasks[1] = 4;

	neighboursmasks[2] = 8 + 4;
	neighboursmasks[3] = 8 + 0;
	neighboursmasks[4] = 8 + 1 ;

	neighboursmasks[5] = 2 + 4;
	neighboursmasks[6] = 2 + 0;
	neighboursmasks[7] = 2 + 1;
	
	int neighboursids[8];
	neighboursids[0] = -1;
	neighboursids[1] = +1;

	neighboursids[2] = w + 1;
	neighboursids[3] = w + 0;
	neighboursids[4] = w - 1 ;

	neighboursids[5] = -w + 1;
	neighboursids[6] = -w + 0;
	neighboursids[7] = -w - 1 ;

	double e_shading_term = 0.;
	double e_depth_regulariation_term = 0.;
	double e_normal_shape_prior_term = 0.; 

	for (int y = selection_top; y <= selection_bottom; ++y)
	{
		for (int x = selection_left; x <= selection_right; ++x)
		{
			const int idx = x + y * w;

			// valid pixel?
			if (py[idx] < 0.0)
			{
				if (nullptr != pe_per_pixel)
					pe_per_pixel[idx] = 0.;

				continue;
			}
					
			cv::Point3d vp(px[idx], py[idx], pz[idx]);
			cv::Point3d vn(pnx[idx], pny[idx], pnz[idx]);

			// e - shading constraint term
			double e_pixel_shading_term;
			{
				double vp_inv_length = 1.0f / std::sqrt(vp.x * vp.x + vp.y * vp.y + vp.z * vp.z);
				cv::Point3d l(-vp.x * vp_inv_length, -vp.y * vp_inv_length, -vp.z * vp_inv_length);

				double r2 = vp.dot(vp);
				
				double I = l.dot(vn) / r2;
				I *= global_albedo; 

				double I_delta = pi_original[idx] - I;
				e_pixel_shading_term = I_delta * I_delta;

				e_pixel_shading_term = e_pixel_shading_term;
			}
			e_shading_term += e_pixel_shading_term;

			// e - depth regularization term
			double e_pixel_depth_regulariation_term;
			{
				e_pixel_depth_regulariation_term = (py_original[idx] - vp.y) * (py_original[idx] - vp.y);
			}
			e_depth_regulariation_term += e_pixel_depth_regulariation_term;

			// e - normal based shape prior term
			double e_pixel_normal_shape_prior_term = 0.f;
			unsigned char border_mask = 0;
			if (x == 0) 
				border_mask |= 1;
			if (y == 0)
				border_mask |= 2;
			if (x == w_m1) 
				border_mask |= 4;
			if (y == h_m1)
				border_mask |= 8;
		
			{
				int list [] = { 1, 2, 3 };
				for (int ni = 0; ni < 3; ++ni)
				{
					if (border_mask & neighboursmasks[list[ni]])
						continue;

					int idx_neighbour = idx + neighboursids[list[ni]];
					if (py[idx_neighbour] > 0.0)
					{
						cv::Point3d vn_neighbour(pnx[idx_neighbour], pny[idx_neighbour], pnz[idx_neighbour]);
						cv::Point3d diff = vn - vn_neighbour;
						e_pixel_normal_shape_prior_term += diff.dot(diff);
					}
				}
			}
			e_normal_shape_prior_term += e_pixel_normal_shape_prior_term;

		
			if (nullptr != pe_per_pixel)
			{
				pe_per_pixel[idx] = 
					weight_shading_constraint_term * e_pixel_shading_term + 
					weight_depth_regularization_term * e_pixel_depth_regulariation_term + 
					weight_normal_shape_prior_term * e_pixel_normal_shape_prior_term 
					; 

			}
		}
	}

	return weight_shading_constraint_term * e_shading_term + 
					weight_depth_regularization_term * e_depth_regulariation_term + 
					weight_normal_shape_prior_term * e_normal_shape_prior_term; 
}

void bohme_shading_constraint_filter::regenerate_xz(const cv::Mat& intput_image_depth, cv::Mat& image_x, cv::Mat& image_y, cv::Mat& image_z,
												const int selection_left, const int selection_right, 
												const int selection_top, const int selection_bottom)
{
	const int w = intput_image_depth.cols;
	double *pd_in = reinterpret_cast<double*>(intput_image_depth.data);
	double *py = reinterpret_cast<double*>(image_y.data);
	double *px = reinterpret_cast<double*>(image_x.data);
	double *pz = reinterpret_cast<double*>(image_z.data);

	for (int y = selection_top; y <= selection_bottom; ++y)
	{
		for (int x = selection_left; x <= selection_right; ++x)
		{
			int idx = x + y * w;
			if (pd_in[idx] > 0.)
			{
				px[idx] = (static_cast<double>(x) - m_intrinsics.cx)*pd_in[idx] / (m_intrinsics.fx);
				pz[idx] = (m_intrinsics.cy - static_cast<double>(y))*pd_in[idx] / (m_intrinsics.fy);
				py[idx] = pd_in[idx];
			}
			else
			{
				py[idx] = -1.0;
				px[idx] = -1.0;
				pz[idx] = -1.0;
			}
		}
	}
}

void bohme_shading_constraint_filter::store_pixel(const int x, const int y)
{
	const double *px = reinterpret_cast<const double*>(m_image_x.data);
	const double *py = reinterpret_cast<const double*>(m_image_y.data);
	const double *pz = reinterpret_cast<const double*>(m_image_z.data);

	const int w = m_image_y.cols;
	const int h = m_image_y.rows;
	const int idx = x + y * w;

	m_pixel_back_up_storage.value_x = px[idx];
	m_pixel_back_up_storage.value_y = py[idx];
	m_pixel_back_up_storage.value_z = pz[idx];

	const cv::Point2i plefttop(std::max(0, x-1), std::max(0,y-1));
	const cv::Point2i prightbottom(std::min(w, x+1), std::min(h,y+1));

	const double* pnx = reinterpret_cast<const double*>(m_image_nx.data);
	const double* pny = reinterpret_cast<const double*>(m_image_ny.data);
	const double* pnz = reinterpret_cast<const double*>(m_image_nz.data);

	double* patch_nx = reinterpret_cast<double*>(m_pixel_back_up_storage.patch_nx.data);
	double* patch_ny = reinterpret_cast<double*>(m_pixel_back_up_storage.patch_ny.data);
	double* patch_nz = reinterpret_cast<double*>(m_pixel_back_up_storage.patch_nz.data);

	int local_y = 0;
	for (int global_y = plefttop.y; global_y <= prightbottom.y; ++global_y)
	{
		int local_x = 0;
		for (int global_x = plefttop.x; global_x <= prightbottom.x; ++global_x)
		{
			int local_idx = local_x + pixel_back_up_storage::patch_size * local_y;
			int global_idx = global_x + w * global_y;
			patch_nx[local_idx] = pnx[global_idx]; 
			patch_ny[local_idx] = pny[global_idx]; 
			patch_nz[local_idx] = pnz[global_idx]; 

			++local_x;
		}
		++local_y;
	}
}

void bohme_shading_constraint_filter::restore_pixel(const int x, const int y)
{
	double *px = reinterpret_cast<double*>(m_image_x.data);
	double *py = reinterpret_cast<double*>(m_image_y.data);
	double *pz = reinterpret_cast<double*>(m_image_z.data);

	const int w = m_image_y.cols;
	const int h = m_image_y.rows;
	const int idx = x + y * w;

	px[idx] = m_pixel_back_up_storage.value_x;
	py[idx] = m_pixel_back_up_storage.value_y;
	pz[idx] = m_pixel_back_up_storage.value_z;

	cv::Point2i plefttop(std::max(0, x-1), std::max(0,y-1));
	cv::Point2i prightbottom(std::min(w-1, x+1), std::min(h-1,y+1));
	cv::Rect nrect(plefttop, prightbottom);

	double* pnx = reinterpret_cast<double*>(m_image_nx.data);
	double* pny = reinterpret_cast<double*>(m_image_ny.data);
	double* pnz = reinterpret_cast<double*>(m_image_nz.data);

	const double* patch_nx = reinterpret_cast<const double*>(m_pixel_back_up_storage.patch_nx.data);
	const double* patch_ny = reinterpret_cast<const double*>(m_pixel_back_up_storage.patch_ny.data);
	const double* patch_nz = reinterpret_cast<const double*>(m_pixel_back_up_storage.patch_nz.data);

	int local_y = 0;
	for (int global_y = plefttop.y; global_y <= prightbottom.y; ++global_y)
	{
		int local_x = 0;
		for (int global_x = plefttop.x; global_x <= prightbottom.x; ++global_x)
		{
			int local_idx = local_x + pixel_back_up_storage::patch_size * local_y;
			int global_idx = global_x + w * global_y;
			pnx[global_idx] = patch_nx[local_idx]; 
			pny[global_idx] = patch_ny[local_idx]; 
			pnz[global_idx] = patch_nz[local_idx]; 

			++local_x;
		}
		++local_y;
	}
}
		
void bohme_shading_constraint_filter::create_roi(const int x, const int y, const int sel_width, const int w, const int h, cv::Point2i* roi)
{
	roi[0].x = x - sel_width;
	roi[0].y = y - sel_width;
	roi[1].x = x + sel_width;
	roi[1].y = y + sel_width;

	roi[0].x = std::max(0, roi[0].x);
	roi[0].y  = std::max(0, roi[0].y);
	roi[1].x = std::min(w - 1, roi[1].x);
	roi[1].y = std::min(h - 1, roi[1].y);
}
