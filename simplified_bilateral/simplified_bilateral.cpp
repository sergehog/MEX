/** 
*	@file simplified_bilateral.cpp
*	@date 11.03.2015
*	@author Sergey Smirnov <sergey.smirnov@tut.fi>
*/

#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}
#endif // __cplusplus

#include "../common/meximage.h"
//#include "../common/defines.h"
#include <vector>
#define _USE_MATH_DEFINES
#include <cmath>
#include <algorithm>
#include <memory>
#ifndef _DEBUG
#include <omp.h>
#endif

//#define M_PI       3.14159265358979323846
#define WEIGHTS_ORIGINAL

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")

typedef unsigned char uint8;
using namespace mymex;

float getWeight(const MexImage<float> &Image, const int x1, const int y1, const int x2, const int y2, const float sigma_color);

void recursive_bilateral(const MexImage<float> &Signal, const MexImage<float> &Filtered, const MexImage<float> &Image, const float sigma_color, const float sigma_spatial = 1.0)
{
	const int width = Signal.width;
	const int height = Signal.height;
	const int layers = Signal.layers;
	const int colors = Image.layers;

	MexImage<float> Temporal(width, height, layers);
	MexImage<float> Weights(width, height);
	MexImage<float> Weights2(width, height);
	Weights.setval(0.f);
	Weights2.setval(0.f);
	Temporal.setval(0.f);
	Filtered.setval(0.f);
	//WeightsTemporal.setval(0.f);

	//std::unique_ptr<float[]> weights_lookup(new float[256 * colors]);

	//for (int i = 0; i < 256 * colors; i++)
	//{
	//	weights_lookup[i] = float(exp(-double(i) / (255 * colors * sigma_color)));
	//}

	const long HW = Signal.layer_size;

	// horisontal passes
#pragma omp parallel
	{

		std::unique_ptr<float[]> temporal1(new float[Signal.layers]);
		std::unique_ptr<float[]> temporal2(new float[Signal.layers]);

		float temporal_weight1, temporal_weight2;

#pragma omp for
		for (int y = 0; y<height; y++)
		{
			for (int l = 0; l<layers; l++)
			{
				temporal1[l] = Signal(0, y, l);
				temporal2[l] = Signal(width - 1, y, l);
				Temporal(0, y, l) = temporal1[l];
				Temporal(width - 1, y, l) = temporal2[l];
			}
			temporal_weight1 = 1;
			temporal_weight2 = 1;
			Weights(0, y) = temporal_weight1;
			Weights(width - 1, y) = temporal_weight2;

			for (int x1 = 1; x1<width; x1++)
			{
				const int x2 = width - x1 - 1;
				//const float weight1 = 1.f;
				//const float weight2 = 1.f;
				const float weight1 = getWeight(Image, x1, y, x1 - 1, y, sigma_color);
				const float weight2 = getWeight(Image, x2, y, x2 + 1, y, sigma_color);

				temporal_weight1 = (1 + temporal_weight1 * sigma_spatial*weight1);
				temporal_weight2 = (1 + temporal_weight2 * sigma_spatial*weight2);
				Weights(x1, y) += temporal_weight1;
				Weights(x2, y) += temporal_weight2;

#pragma omp parallel for
				for (int l = 0; l<layers; l++)
				{
					temporal1[l] = (Signal(x1, y, l) + temporal1[l] * sigma_spatial*weight1);
					temporal2[l] = (Signal(x2, y, l) + temporal2[l] * sigma_spatial*weight2);
					Temporal(x1, y, l) += temporal1[l];
					Temporal(x2, y, l) += temporal2[l];
				}
			}

#pragma omp parallel for
			for (int x = 0; x<width; x++)
			{
				Weights(x, y) -= 1;
				for (int l = 0; l<layers; l++)
				{
					Temporal(x, y, l) -= Signal(x, y, l);
				}
			}
		}
	}

	//// intermediate normalization
	//for(int x=0; x<width; x++)
	//{
	//	for(int y=0; y<height; y++)
	//	{
	//		for(int l=0; l<layers; l++)		
	//		{
	//			Temporal(x,y,l) /= Weights(x,y);
	//		}			
	//	}
	//}	

	//Weights.setval(0.f);

	//vertical passes
#pragma omp parallel
	{

		std::unique_ptr<float[]> temporal1(new float[Signal.layers]);
		std::unique_ptr<float[]> temporal2(new float[Signal.layers]);

		float temporal_weight1, temporal_weight2;

#pragma omp for
		for (int x = 0; x<width; x++)
		{
			for (int l = 0; l<layers; l++)
			{
				temporal1[l] = Temporal(x, 0, l);
				temporal2[l] = Temporal(x, height - 1, l);
				Filtered(x, 0, l) = temporal1[l];
				Filtered(x, height - 1, l) = temporal2[l];
			}
			temporal_weight1 = Weights(x, 0);
			temporal_weight2 = Weights(x, height - 1);
			Weights2(x, 0) = temporal_weight1;
			Weights2(x, height - 1) = temporal_weight2;

			for (int y1 = 1; y1<height; y1++)
			{
				const int y2 = height - y1 - 1;
				//const float weight1 = 1.f;
				//const float weight2 = 1.f;
				const float weight1 = getWeight(Image, x, y1, x, y1 - 1, sigma_color);
				const float weight2 = getWeight(Image, x, y2, x, y2 + 1, sigma_color);

				temporal_weight1 = (Weights(x, y1) + temporal_weight1 * sigma_spatial*weight1);
				temporal_weight2 = (Weights(x, y2) + temporal_weight2 * sigma_spatial*weight2);
				Weights2(x, y1) += temporal_weight1;
				Weights2(x, y2) += temporal_weight2;

#pragma omp parallel for
				for (int l = 0; l<layers; l++)
				{
					temporal1[l] = (Temporal(x, y1, l) + temporal1[l] * sigma_spatial*weight1);
					temporal2[l] = (Temporal(x, y2, l) + temporal2[l] * sigma_spatial*weight2);
					Filtered(x, y1, l) += temporal1[l];
					Filtered(x, y2, l) += temporal2[l];
				}
			}

#pragma omp parallel for
			for (int y = 0; y<height; y++)
			{
				Weights2(x, y) -= Weights(x, y);
				for (int l = 0; l<layers; l++)
				{
					Filtered(x, y, l) -= Temporal(x, y, l);
				}
			}
		}
	}

	// final normalization
#pragma omp parallel for
	for (long i = 0; i<HW; i++)
	{
		const int x = i / height;
		const int y = i % height;
		for (int l = 0; l<layers; l++)
		{
			Filtered(x, y, l) /= Weights2(x, y);
		}
	}
}


void mexFunction(int nout, mxArray* output[], int in, const mxArray* input[])
{
#ifndef _DEBUG
	omp_set_num_threads(std::max<int>(4, omp_get_max_threads()));
	omp_set_dynamic(std::max<int>(2, omp_get_max_threads() / 2));
#endif

	if (in < 2 || in > 4 || nout != 1 || mxGetClassID(input[0]) != mxSINGLE_CLASS || mxGetClassID(input[1]) != mxSINGLE_CLASS)
	{
		mexErrMsgTxt("USAGE: [Filtered] = recursive_bilateral(single(Signal), single(Image), <sigma_color, sigma_spatial>);");
	}

	const float sigma_color = (in > 2) ? std::max<float>(0.f, mxGetScalar(input[2])) : 10.0;
	const float sigma_spatial = (in > 3) ? std::max<float>(0.f, mxGetScalar(input[3])) : 1.0;

	MexImage<float> Signal(input[0]);
	MexImage<float> Image(input[1]);
	
	const long HW = Image.layer_size;
	const int layers = Signal.layers;
	const int width = Image.width;
	const int height = Image.height;

	if (Signal.width != width || Signal.height != height)
	{
		mexErrMsgTxt("Signal and Image must have the same resolution");
	}
	const size_t dims[] = { (size_t)height, (size_t)width, (size_t)layers };
	output[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
	MexImage<float> Filtered(output[0]);
	//recursive_bilateral(Signal, Filtered, Image, sigma_color, sigma_spatial);
	//recursive_bilateral(Signal, Filtered, Image, sigma_color, sigma_spatial);
	
	int maxSegment = 0;
	std::vector<std::unique_ptr<float[]>> means;
	std::vector<unsigned int> pixels;
	means.push_back(std::unique_ptr<float[]>(new float[layers]))
	for ()
	for (long i = 1; i < HW; i++)
	{

	}
}



float getWeight(const MexImage<float> &Image, const int x1, const int y1, const int x2, const int y2, const float sigma_color)
{
#ifdef WEIGHTS_ORIGINAL
	float diff = 0;
	for (int c = 0; c < Image.layers; c++)
	{
		diff += abs(Image(x1, y1, c) - Image(x2, y2, c));
	}
	return diff / Image.layers > sigma_color ? 1.f : 0.f;
#else
	int diff = 0;
	for (int c = 0; c < Image.layers; c++)
	{
		diff += int(abs(Image(x1, y1, c) - Image(x2, y2, c)));
		//int diff = int(abs(Image(x1, y1, c) - Image(x2, y2, c)));
		//max_diff = diff > max_diff ? diff : max_diff;
	}
	return exp(-float(diff) / (255 * sigma_color*Image.layers));
#endif
}

