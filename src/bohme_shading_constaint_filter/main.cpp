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

#include "stdafx.h"
#include "single_frame.h"

#include "bohme_shading_constraint_filter.h"

#include <random>
#include <iostream>
#include <array>

single_frame g_frame;

bohme_shading_constraint_filter* g_shading_constraint_filter = nullptr;

class optimization_function: public cv::MinProblemSolver::Function
{
	bohme_shading_constraint_filter& filter;
	const int width;
	const int height;
	const int dimensions;
public:
	optimization_function(bohme_shading_constraint_filter& filter):
		filter(filter),
		width(filter.get_width()),
		height(filter.get_height()),
		dimensions(filter.get_dimensions())
	{
	}

	virtual int getDims() const
	{
		return dimensions;
	}

	virtual double getGradientEps() const
	{
		return 1.0;
	}

	virtual double calc(const double* x) const
	{
		const cv::Mat p(
			height,
			width,
			CV_64FC1,
			const_cast<double*>(x));
		double energy = g_shading_constraint_filter->calculate_energy(p);
		return energy;
	}
	int teller;

	virtual void getGradient(const double* x, double* grad)
	{
		cv::Mat image_gradient(g_frame.image_working_depth.rows, g_frame.image_working_depth.cols, CV_64FC1, grad);
		cv::Mat p(g_frame.image_working_depth.rows, g_frame.image_working_depth.cols, CV_64FC1, const_cast<double*>(x));
		g_shading_constraint_filter->calculate_gradient(p, image_gradient);
	}
};


int g_current_iteration = 0;

bool g_aborted = false;
bool g_reset = false;
bool g_done = false;

cv::Mat g_image_vis;
cv::Mat g_image_vis2;
const int g_image_vis_scale = 2; // visualization scale

void add_gaussian_noise(cv::Mat &image, double sigma);

void render_depth_map(const cv::Mat& in_image, cv::Mat& out_image);
void render_intensity_map(const cv::Mat& in_image, cv::Mat& out_image);
void visualize_frame();


bool func_iteration_done();
const char* keys =
{
	"{d| |directory}"
	"{m|0|median filter}"
	"{sigma_d|0.05|depth sigma in meters}"
	"{sigma_i|0.005|intensity sigma}"
};

int _tmain(int argc, _TCHAR* argv[])
{
	cv::CommandLineParser parser(argc, argv, keys);
	std::string dir = parser.get<std::string>("d");
	if (dir.length() == 0)
	{
		std::cout << "Please provide command line arguments: " << std::endl;
		std::cout << "e.g. -d=..\\data\\bohme_datasets\\2 -m=0" << std::endl;
		parser.printMessage();
		return -1;
	}
	
	bool enable_median_filter = (parser.get<int>("m") == 1) ? true : false;

	//
	// optimization stopping critera
	const int max_iterations = 10000;
	const double min_optimization_epsilon = 0.1;

	//
	// load data
	camera_intrinsics intrinsics;
	single_frame_loader::load(dir, g_frame, intrinsics,
		0.5f /* <<=== scale of images !!! (so you can run it faster for debugging) */);

	//
	// add some noisy & init
	const double depth_sigma = parser.get<double>("sigma_d");
	const double intensity_sigma = parser.get<double>("sigma_i");

	add_gaussian_noise(g_frame.image_noisy_depth, depth_sigma);
	add_gaussian_noise(g_frame.image_noisy_intensity, intensity_sigma);

	//
	// run!
	do
	{
		g_reset = false;
		g_done = false;

		//
		// init
		g_frame.image_noisy_depth.copyTo(g_frame.image_working_depth);

		//
		// median filter
		if (enable_median_filter)
			cv::medianBlur(g_frame.image_working_depth, g_frame.image_working_depth, 3);
	
		//
		// set-up parameters
		bohme_shading_constraint_filter_parameters params;
		params.global_albedo = g_frame.global_albedo;

		params.weight_shading_constraint_term = 1.0f / (2.0f * intensity_sigma * intensity_sigma);
		params.weight_depth_regularization_term = 1.0f / (2.0f * depth_sigma * depth_sigma);
		params.weight_normal_shape_prior_term = 1.0f;
		//params.weight_geometric_smoothness_term ;
	
		//
		// init filter
		delete g_shading_constraint_filter;
		g_shading_constraint_filter = new bohme_shading_constraint_filter(
			g_frame.image_working_depth,
			g_frame.image_noisy_intensity,
			params,
			intrinsics);

		//params.global_albedo = g_shading_constraint_filter->estimate_current_albedo(
		//	g_frame.image_working_depth, g_frame.image_noisy_intensity);

		double prev_energy = g_shading_constraint_filter->calculate_energy(g_frame.image_working_depth);
		g_shading_constraint_filter->render_intensity_image(g_frame.image_rendered_intensity);


		visualize_frame();
		cv::waitKey(100);
	
		//
		// optimize
		cv::Ptr<optimization_function> opt_function = cv::makePtr<optimization_function>(optimization_function(*g_shading_constraint_filter));
		cv::TermCriteria termcrit;
		termcrit.type = cv::TermCriteria::MAX_ITER;
		termcrit.maxCount = 1; // this is a trick, ignore this value, we want to visualize the result after each iteration, so will run the solver for a single iteration and render...run...runder

		cv::Ptr<cv::ConjGradSolver> solver = cv::ConjGradSolver::create(opt_function, termcrit);

		std::cout << "0 energy: " << prev_energy << std::endl;
		for (int g_current_iteration = 0; g_current_iteration < max_iterations /* max iterations */; ++g_current_iteration)
		{
			// optimize!
			cv::Mat x_val(g_frame.image_working_depth.cols * g_frame.image_working_depth.rows, 1, CV_64FC1, g_frame.image_working_depth.data);
			solver->minimize(x_val);

			// calculate energy
			double new_energy = g_shading_constraint_filter->calculate_energy(g_frame.image_working_depth);
			std::cout << g_current_iteration + 1 << " energy: " << new_energy << std::endl;
			double energy_epsilon = prev_energy - new_energy;
			prev_energy = new_energy;

			// display after each iteration the result
			// check if the user want to reset or cancel
			if (!func_iteration_done() || energy_epsilon < min_optimization_epsilon)
			{
				break;
			}
		}
		
		g_done = true;
	
		if (!g_aborted && !g_reset)
		{
			visualize_frame();
			cv::waitKey(0);
		}
		

	}
	while (g_reset);

	delete g_shading_constraint_filter;

	return 0;
}

void add_gaussian_noise(cv::Mat &image, double sigma)
{
	if (sigma == 0.f)
		return;

	std::random_device rd;
	std::default_random_engine generator(rd());
	std::normal_distribution<double> distribution(0.0, sigma);
	double* pdata = reinterpret_cast<double*>(image.data);
	const int w = image.cols;
	const int h = image.rows;

	int idx = 0;
	for (int y = 0; y < h; ++y)
	{
		for (int x = 0; x < w; ++x)
		{
			pdata[idx] += distribution(generator);
			++idx;
		}
	}

}

bool func_iteration_done()
{
	g_shading_constraint_filter->render_intensity_image(g_frame.image_rendered_intensity);

	visualize_frame();

	int key = cv::waitKey(10);

	if (114 == key  /* r */)
	{
		g_reset = true;
		std::cout << "resetting!!" << std::endl;
		return false; // rest!
	}
	else if ((97 == key  /* a */) || (27 == key  /* ESC */))
	{
		std::cout << "aborting!!"<< std::endl;
		g_aborted = true;
		return false; // abort!
	}
	
	return true; 
}

void visualize_frame()
{
	const int w = g_frame.image_working_depth.cols;
	const int h = g_frame.image_working_depth.rows;

	g_image_vis.create(h, w * 5, CV_8UC3);

	cv::Mat sub1(g_image_vis, cv::Rect(0,0, w, h));
	cv::Mat sub2(g_image_vis, cv::Rect(w, 0, w, h));
	cv::Mat sub3(g_image_vis, cv::Rect(w*2, 0, w, h));
	cv::Mat sub4(g_image_vis, cv::Rect(w*3, 0, w, h));
	cv::Mat sub5(g_image_vis, cv::Rect(w*4, 0, w, h));

	cv::Mat img_d;

	render_depth_map(g_frame.image_ideal_depth, img_d);
	img_d.copyTo(sub1);
	
	render_depth_map(g_frame.image_noisy_depth, img_d);
	img_d.copyTo(sub2);

	render_depth_map(g_frame.image_working_depth, img_d);
	img_d.copyTo(sub3);

	cv::Mat img_i;

	render_intensity_map(g_frame.image_rendered_intensity, img_i);
	img_i.copyTo(sub4);

	render_intensity_map(g_frame.image_noisy_intensity, img_i);
	img_i.copyTo(sub5);
	
	// up scale
	cv::Size newsize(
			g_image_vis.cols * g_image_vis_scale, 
			g_image_vis.rows * g_image_vis_scale
			);

	cv::resize(g_image_vis,g_image_vis2, newsize, 0.0, 0.0, CV_INTER_NN);
	
	const int ftpc = (g_image_vis2.cols / 5);
	const int fttop = 12;

	std::array<cv::String, 5> image_titles = { "GROUND TRUTH DEPTH", "INPUT DEPTH", "FILTERED DEPTH", "CALCULATED INTENSITY", "INPUT INTENSITY" };

	char chtext[128];
	if (g_done)
		sprintf_s(chtext, 128, "iteration: %d (done: tolerance too small or max iterations reached)", g_current_iteration);
	else
		sprintf_s(chtext, 128, "iteration: %d", g_current_iteration);

	// semi-trans boxes below text
	{
		cv::Mat overlay;
		double alpha = 0.8;

		// copy the source image to an overlay
		g_image_vis2.copyTo(overlay);

		// draw
		cv::Scalar color = cvScalar(50, 50, 50);
		int baseline;
		cv::Size txtsize;
		int boxheight = fttop + 2;
		
		txtsize = cv::getTextSize(image_titles[0], cv::FONT_HERSHEY_PLAIN, 1.0f, 1, &baseline);
		cv::rectangle(overlay, cv::Rect(0, 0, txtsize.width, boxheight), color, CV_FILLED);

		txtsize = cv::getTextSize(image_titles[1], cv::FONT_HERSHEY_PLAIN, 1.0f, 1, &baseline);
		cv::rectangle(overlay, cv::Rect(ftpc, 0, txtsize.width, boxheight), color, CV_FILLED);

		txtsize = cv::getTextSize(image_titles[2], cv::FONT_HERSHEY_PLAIN, 1.0f, 1, &baseline);
		cv::rectangle(overlay, cv::Rect(ftpc*2, 0, txtsize.width, boxheight), color, CV_FILLED);

		txtsize = cv::getTextSize(image_titles[3], cv::FONT_HERSHEY_PLAIN, 1.0f, 1, &baseline);
		cv::rectangle(overlay, cv::Rect(ftpc*3, 0, txtsize.width, boxheight), color, CV_FILLED);

		txtsize = cv::getTextSize(image_titles[4], cv::FONT_HERSHEY_PLAIN, 1.0f, 1, &baseline);
		cv::rectangle(overlay, cv::Rect(ftpc*4, 0, txtsize.width, boxheight), color, CV_FILLED);

		txtsize = cv::getTextSize(chtext, cv::FONT_HERSHEY_DUPLEX, 0.7f, 1, &baseline);
		cv::rectangle(overlay, cv::Rect(0, g_image_vis2.rows - 26, txtsize.width + 4, 26), color, CV_FILLED);
		
		// blend the overlay with the source image
		cv::addWeighted(overlay, alpha, g_image_vis2, 1 - alpha, 0, g_image_vis2);
	}

	// text
	{		
		cv::Scalar textcolor = cvScalar(240,240,240);

		putText(g_image_vis2, image_titles[0], cvPoint(0, fttop),
			cv::FONT_HERSHEY_PLAIN, 1.0f, textcolor, 1, CV_AA);

		putText(g_image_vis2, image_titles[1], cvPoint(ftpc, fttop),
		cv::FONT_HERSHEY_PLAIN, 1.0f, textcolor, 1, CV_AA);

		putText(g_image_vis2, image_titles[2], cvPoint(ftpc*2,fttop),
		cv::FONT_HERSHEY_PLAIN, 1.0f, textcolor, 1, CV_AA);

		putText(g_image_vis2, image_titles[3], cvPoint(ftpc*3,fttop),
		cv::FONT_HERSHEY_PLAIN, 1.0f, textcolor, 1, CV_AA);
	
		putText(g_image_vis2, image_titles[4], cvPoint(ftpc*4,fttop),
		cv::FONT_HERSHEY_PLAIN, 1.0f, textcolor, 1, CV_AA);

		putText(g_image_vis2, chtext, cvPoint(0,g_image_vis2.rows - 6), 
		cv::FONT_HERSHEY_DUPLEX, 0.7f, textcolor, 1, CV_AA);
	}

	cv::imshow("Shading Constraint Filter (r = reset, ESC = abort) (hold the key for a while)", g_image_vis2);
}

void render_intensity_map(const cv::Mat& in_image, cv::Mat& out_image)
{
	cv::Mat adjmap;
	in_image.convertTo(adjmap, CV_8UC3, 255.0f, 0.f);
	cv::cvtColor(adjmap, out_image, cv::COLOR_GRAY2BGR);
}

void render_depth_map(const cv::Mat& in_image, cv::Mat& out_image)
{
	const double min = g_frame.palette_depth_min_max[0];
	const double max = g_frame.palette_depth_min_max[1];

	// expand your range to 0..255
	double scale = 255.0f / (max - min);
	cv::Mat adjmap;
	in_image.convertTo(adjmap, CV_8UC1, scale, -min * scale);

	cv::Mat falseColorsMap;
	cv::applyColorMap(adjmap, out_image, cv::COLORMAP_OCEAN);
}
