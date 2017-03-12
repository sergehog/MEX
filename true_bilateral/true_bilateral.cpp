/**
* @file true_bilateral.cpp
* @author Sergey Smirnov <sergey.smirnov@tut.fi>
* @date 09.07.2015
* @copyright 3D Media Group / Tampere University of Technology
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
#include <array>
#ifndef _DEBUG
#include <omp.h>
#endif

//#define M_PI       3.14159265358979323846

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")

#define colors 3

//typedef unsigned char uint8;
using namespace mymex;

struct general_bilateral 
{
	virtual void process(MexImage<float> &Signal, MexImage<uint8_t> &Image, MexImage<float> &SignalOut, const float sigma_color, const float sigma_distance) = 0;
};

#define DIAMETER(R) ((R) * 2 + 1)
#define WINDOW(R) (((R) * 2 + 1)*((R) * 2 + 1))

template<int radius>
struct optimized_bilateral : general_bilateral
{
	void process(MexImage<float> &Signal, MexImage<uint8_t> &Image, MexImage<float> &SignalOut, const float sigma_color, const float sigma_distance)
	{
		std::unique_ptr<float[]> color_lookup(new float[colors * 256]);
		for (int i = 0; i < colors * 256; i++)
		{ 
			color_lookup = exp(-sqrt(float(i) / sigma_color));
		}
		const int width = Signal.width;
		const int height = Signal.width;

		MexImage<float> Normalization(width, height);
		Normalization.setval(1.f);
		#pragma omp parallel for
		for (long i = 0; i < HW; i++)
		{
			SignalOut[i] = Signal[i];
		}
		#pragma omp parallel for
		for (int d = 0; d < WINDOW(radius); d++)
		{
			const int dx = d / DIAMETER(radius) - radius;
			const int dy = d % DIAMETER(radius) - radius;
			if (dx == dy && dy == 0)
			{
				continue;
			}
			const float distance_weight = exp(-sqrt(float((dx - x)*(dx - x) + (dy - y)*(dy - y))) / sigma_distance);

			#pragma omp parallel for
			for (long i = 0; i < HW; i++)
			{
				const int x = i / height;
				const int y = i % height;
				const int xw = x + dx;
				const int yw = y + dy;
				if (xw < 0 || xw >= width || yw < 0 || yw >= height)
				{
					continue;
				}
				int diff = 0;
				for (int c = 0; c < colors; c++)
				{
					diff += abs(int(Image(x, y, c)) - int(Image(xw, yw, c)));
				}
				const float weight = color_lookup[diff] * distance_weight;
				const float svalue = Signal(xw, yw) * weight;
				#pragma omp atomic
				SignalOut(x, y) += svalue;
				#pragma omp atomic
				Normalization(x, y) += weight;
			}
		}
#pragma omp parallel for
		for (long i = 0; i < HW; i++)
		{
			SignalOut[i] /= Normalization[i];
		}

	}
};

#define MAXRADIUS 20;
typedef general_bilateral*[MAXRADIUS] bilateral_list;

// Start of unroll
template<int radius, template<int> class T>
struct unroller {
	unroller<radius - 1, T> next_unroll;
	inline unroller(general_bilateral *) : next_unroll(val1), functor(val1) {}
};

// End of unroll
template<template<unsigned> class T>
struct unroll_constructor<1, T> {
	template<typename T1> inline unroll_constructor(T1 &) {}
};



void mexFunction(const int nout, mxArray* output[], const int in, const mxArray* input[])
{
#ifndef _DEBUG
	omp_set_num_threads(std::max<int>(4, omp_get_max_threads()));
	omp_set_dynamic(std::max<int>(2, omp_get_max_threads() / 2));
#endif

	if (in != 5 || nout != 4 || mxGetClassID(input[0]) != mxUINT8_CLASS || mxGetClassID(input[1]) != mxUINT8_CLASS || mxGetClassID(input[2]) != mxSINGLE_CLASS || mxGetClassID(input[3]) != mxSINGLE_CLASS)
	{
		mexErrMsgTxt("USAGE: [Li, Ri, DLi, DRi] = true_bilateral(uint8(L), uint8(R), single(DL), single(DR), radius);");
	}

	MexImage<uint8_t> Left(input[0]);
	MexImage<uint8_t> Right(input[1]);
	MexImage<float> DL(input[0]);
	MexImage<float> DR(input[1]);

	const unsigned radius = std::max<unsigned>(1u, std::min<unsigned>(100, mxGetScalar(input[4])));
	const int width = Left.width;
	const int height = Left.height;
	const long HW = Left.layer_size;
	const float nan = sqrt(-1.f);
	const size_t dims[] = { (size_t)height, (size_t)width, (size_t)colors };
	const size_t dims1[] = { (size_t)height, (size_t)width, 1 };

	output[0] = mxCreateNumericArray(3, dims, mxUINT8_CLASS, mxREAL);
	output[1] = mxCreateNumericArray(3, dims, mxUINT8_CLASS, mxREAL);
	output[2] = mxCreateNumericArray(3, dims1, mxSINGLE_CLASS, mxREAL);
	output[3] = mxCreateNumericArray(3, dims1, mxSINGLE_CLASS, mxREAL);


	MexImage<uint8_t> LeftI(output[1]);
	MexImage<uint8_t> RightI(output[2]);
	MexImage<float> DLI(output[3]);
	MexImage<float> DRI(output[4]);

	optimized_inpaint inpainting;




}