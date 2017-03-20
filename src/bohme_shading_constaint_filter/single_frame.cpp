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

#include "single_frame.h"

#include <iostream>
#include <tiffio.h>
#include <fstream>

bool single_frame_loader::load(std::string directory, single_frame& out_frame, camera_intrinsics& intrinsics, const double scale)
{
	if (directory.length() < 2)
		return false;

	if (directory[directory.length() - 1]  != '/' && directory[directory.length() - 1] != '\\')
	{
		directory += "\\";
	}

	int frame_id = 0;
	char spf[512];


	sprintf_s(spf, 512, "%simage_%03d_depth.tiff", directory.c_str(), frame_id);
	if (load_from_tiff(spf, out_frame.image_ideal_depth) == false)
	{
		std::cout << "error::single_frame_loader::failed to load " << spf << std::endl;
		return false;
	}
	out_frame.image_ideal_depth.convertTo(out_frame.image_ideal_depth, CV_64FC1);
	out_frame.image_ideal_depth.copyTo(out_frame.image_noisy_depth);
	out_frame.image_ideal_depth.copyTo(out_frame.image_working_depth);

	// load intensity image
	// - convert to gray scale + scale it from 0 ---> 255 to 0 ---> 1
	cv::Mat image_load_intensity;
	sprintf_s(spf, 512, "%simage_%03d_intensity.tiff", directory.c_str(), frame_id);
	if (load_from_tiff(spf, image_load_intensity ) == false)
	{
		std::cout << "error::single_frame_loader::failed to load " << spf << std::endl;
		return false;
	}
	
	{
		cv::cvtColor(image_load_intensity, image_load_intensity, CV_BGRA2GRAY);
		image_load_intensity.convertTo(out_frame.image_ideal_intensity, CV_64FC1);
		double* pi = reinterpret_cast<double*>(out_frame.image_ideal_intensity.data);
		for (int i = 0, i_end = out_frame.image_ideal_intensity.rows * out_frame.image_ideal_intensity.cols; i < i_end; ++i)
		{
			pi[i] = pi[i] / 255.0f;
		}
	}
	out_frame.image_ideal_intensity.copyTo(out_frame.image_noisy_intensity);
	out_frame.image_ideal_intensity.copyTo(out_frame.image_rendered_intensity);


	sprintf_s(spf, 512, "%scamera_params.txt", directory.c_str());
	if (import_camera_parameters_txt(spf, intrinsics) == false)
	{
		std::cout << "error::single_frame_loader::failed to load " << spf << std::endl;
		return false;
	}

	sprintf_s(spf, 512, "%salbedo.txt", directory.c_str());
	if (import_albedo(spf, out_frame.global_albedo) == false)
	{
		std::cout << "error::single_frame_loader::failed to load " << spf << std::endl;
		return false;
	}

	sprintf_s(spf, 512, "%spalette.txt", directory.c_str());
	if (import_palette_min_max(spf, out_frame.palette_depth_min_max[0], out_frame.palette_depth_min_max[1]) == false)
	{
		std::cout << "error::single_frame_loader::failed to load " << spf << std::endl;
		return false;
	}

	assert(out_frame.image_ideal_intensity.rows == out_frame.image_ideal_depth.rows);
	assert(out_frame.image_ideal_intensity.cols == out_frame.image_ideal_depth.cols);
	assert(intrinsics.width == out_frame.image_ideal_intensity.cols);

	//
	// scale if needed
	if (scale != 1.0f)
	{
		cv::Size newsize(
			static_cast<int>(out_frame.image_ideal_depth.cols * scale), 
			static_cast<int>(out_frame.image_ideal_depth.rows * scale)
			);

		cv::resize(out_frame.image_ideal_depth, out_frame.image_ideal_depth, newsize, 0.0, 0.0, CV_INTER_NN);
		cv::resize(out_frame.image_ideal_intensity, out_frame.image_ideal_intensity, newsize, 0.0, 0.0, CV_INTER_NN);

		cv::resize(out_frame.image_noisy_depth, out_frame.image_noisy_depth, newsize, 0.0, 0.0, CV_INTER_NN);
		cv::resize(out_frame.image_noisy_intensity, out_frame.image_noisy_intensity, newsize, 0.0, 0.0, CV_INTER_NN);

		cv::resize(out_frame.image_working_depth, out_frame.image_working_depth, newsize, 0.0, 0.0, CV_INTER_NN);
		cv::resize(out_frame.image_rendered_intensity, out_frame.image_rendered_intensity, newsize, 0.0, 0.0, CV_INTER_NN);

		intrinsics.cx = intrinsics.cx * scale;
		intrinsics.cy = intrinsics.cy * scale;
		intrinsics.fx = intrinsics.fx * scale;
		intrinsics.fy = intrinsics.fy * scale;
		intrinsics.width =  static_cast<int>(std::floor(intrinsics.width * scale));
		intrinsics.height = static_cast<int>(std::floor(intrinsics.height * scale));
		intrinsics.k1 = intrinsics.k1;
		intrinsics.k2 = intrinsics.k2;
		intrinsics.k3 = intrinsics.k3;
		intrinsics.p1 = intrinsics.p1;
		intrinsics.p2 = intrinsics.p2;
	}


	

	return true;
}

bool single_frame_loader::load_from_tiff(const char *fileName, cv::Mat &out_image)
{
	uint32 width, height, bitspersample;
	uint32 samplesperpixel;
	uint32  stripbytecount;
	uint32 numberofstrips;
	unsigned long imagesize;
	TIFF *image;

	// Open the TIFF image
	if((image = TIFFOpen(fileName, "r")) == NULL)
	{
		std::cout << "TIFF: Failed to open " << fileName << std::endl;
		return false;
	}

	// Find the width and height of the image
	TIFFGetField(image, TIFFTAG_IMAGEWIDTH, &width);
	TIFFGetField(image, TIFFTAG_IMAGELENGTH, &height);
	TIFFGetField(image, TIFFTAG_BITSPERSAMPLE, &bitspersample);
	TIFFGetField(image, TIFFTAG_SAMPLESPERPIXEL, &samplesperpixel);
	uint32 iisRgb=0;
	TIFFGetField(image, TIFFTAG_PHOTOMETRIC, &iisRgb);

	//bool isRgb = (iisRgb == PHOTOMETRIC_RGB);

	uint32 sampleFormat=SAMPLEFORMAT_UINT;
	TIFFGetField(image, TIFFTAG_SAMPLEFORMAT, &sampleFormat);

	numberofstrips = TIFFNumberOfStrips(image);
	stripbytecount = (uint32)TIFFStripSize(image);

	imagesize = height * width + 1;


	int cvtype = CV_8U;
	
	bitspersample = bitspersample&0x000000FF;
	int nbchannels = samplesperpixel&0x000000FF;
	switch (bitspersample)
	{
	case 1:
		assert(0);
		break;
	case 8:
		{
			if (sampleFormat == SAMPLEFORMAT_INT)
				cvtype = CV_8SC(nbchannels);
			else
				cvtype = CV_8UC(nbchannels);

		}
		break;
	case 16:
		{
			if (sampleFormat == SAMPLEFORMAT_INT)
				cvtype = CV_16SC(nbchannels);
			else
				cvtype = CV_16UC(nbchannels);
		}
		break;
	case 32:
		{
			if (sampleFormat == SAMPLEFORMAT_IEEEFP)
				cvtype = CV_32FC(nbchannels);
			else
				cvtype = CV_32SC(nbchannels);
		}
		break;
	default:
		cvtype = CV_8UC(nbchannels);
		break;
	}


	// Resize image/reserve some space
	out_image.create(height, width, cvtype);
	uint32 totsize = static_cast<uint32>(out_image.total() * out_image.elemSize());


	// http://www.ibm.com/developerworks/linux/library/l-libtiff2/
	unsigned char *storage = (unsigned char*)out_image.data;

	//if (isRgb == true)
	//{
	//	::TIFFReadRGBAImage(image, imgInfo.width, imgInfo.height, (uint32*)storage, 0);
	//}
	//else
	{

		uint32 arrpos=0;
		for (uint32 istrip=0; istrip<numberofstrips; istrip++)
		{	
			//assert(	arrpos + stripbytecount <=totsize);

			uint32 finalstripbytecount = stripbytecount;
			if (arrpos + stripbytecount > totsize)
			{
				finalstripbytecount = totsize - arrpos;
			}


			//if(TIFFReadRawStrip(image, istrip, &storage[arrpos] ,  stripbytecount) == -1)
			if(TIFFReadEncodedStrip(image, istrip, &storage[arrpos] ,  finalstripbytecount) == -1)
			{
				std::cout << "IFF: Failed to read " <<  fileName << std::endl;
				return false;
			}

			arrpos += stripbytecount;
		}

		TIFFClose(image);
	}

	return true;
}


bool single_frame_loader::import_camera_parameters_txt(const std::string filename, camera_intrinsics& out_intrinsics)
{
	std::vector<std::string> lines;
	std::string line;
	std::ifstream myfile (filename);
	if (myfile.is_open())
	{
		while ( getline (myfile,line) )
		{
			lines.push_back(line);
		}
		myfile.close();
	}
	else 
	{
		std::cout << "Unable to open file"; 
		return false;
	}

	if (lines.size() < 11)
		return false;

	int i = -1;
	out_intrinsics.width = ::atoi(lines[++i].c_str());
	out_intrinsics.height = ::atoi(lines[++i].c_str());
	out_intrinsics.fx = static_cast<double>(::atof(lines[++i].c_str()));
	out_intrinsics.fy = static_cast<double>(::atof(lines[++i].c_str()));
	out_intrinsics.cx = static_cast<double>(::atof(lines[++i].c_str()));
	out_intrinsics.cy = static_cast<double>(::atof(lines[++i].c_str()));
	out_intrinsics.k1 = static_cast<double>(::atof(lines[++i].c_str()));
	out_intrinsics.k2 = static_cast<double>(::atof(lines[++i].c_str()));
	out_intrinsics.k3 = static_cast<double>(::atof(lines[++i].c_str()));
	out_intrinsics.p1 = static_cast<double>(::atof(lines[++i].c_str()));
	out_intrinsics.p2 = static_cast<double>(::atof(lines[++i].c_str()));
	
	return true;
}

bool single_frame_loader::import_albedo(const std::string filename, double& out_albedo)
{
	std::vector<std::string> lines;
	std::string line;
	std::ifstream myfile (filename);
	if (myfile.is_open())
	{
		while ( getline (myfile,line) )
		{
			lines.push_back(line);
		}
		myfile.close();
	}
	else 
	{
		std::cout << "Unable to open file"; 
		return false;
	}

	if (lines.size() < 1)
		return false;

	int i = -1;
	out_albedo = static_cast<double>(::atof(lines[++i].c_str()));

	return true;
}

bool single_frame_loader::import_palette_min_max(const std::string filename, double& out_palette_min, double& out_palette_max)
{
	std::vector<std::string> lines;
	std::string line;
	std::ifstream myfile(filename);
	if (myfile.is_open())
	{
		while (getline(myfile, line))
		{
			lines.push_back(line);
		}
		myfile.close();
	}
	else
	{
		std::cout << "Unable to open file";
		return false;
	}

	if (lines.size() < 2)
		return false;

	int i = -1;
	out_palette_min = static_cast<double>(::atof(lines[++i].c_str()));
	out_palette_max = static_cast<double>(::atof(lines[++i].c_str()));

	return true;
}
