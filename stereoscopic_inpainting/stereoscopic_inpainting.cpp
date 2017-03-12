/**
* @file stereoscopic_inpainting.cpp
* @author Sergey Smirnov <sergey.smirnov@tut.fi>
* @date 20.05.2015
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


// Start of unroll
template<unsigned unroll_count, template<unsigned> class T>
struct unroll_constructor {
	unroll_constructor<unroll_count - 1, T> next_unroll;
	T<unroll_count - 1> functor;
	template<typename T1> inline unroll_constructor(T1 & val1) : next_unroll(val1), functor(val1) {}
};

// End of unroll
template<template<unsigned> class T>
struct unroll_constructor<1, T> {
	template<typename T1> inline unroll_constructor(T1 &) {}
};
struct general_inpaint {

	virtual void process(const int x, const int y, MexImage<uint8_t> &Image, MexImage<uint8_t> &Trimap, MexImage<float> &Alpha, MexImage<uint8_t> &Fg, MexImage<uint8_t> &Bg) = 0;
};


// The filters for each search variant (range_filters)
template<unsigned R>
struct specialized_inpaint : general_inpaint {

	void process(const int x, const int y, MexImage<uint8_t> &Image, MexImage<uint8_t> &Trimap, MexImage<float> &Alpha, MexImage<uint8_t> &Fg, MexImage<uint8_t> &Bg)
	{

	}
};

#define MAXR 20
class optimized_inpaint {
	// unroll templates
	template<unsigned N>
	struct T_unroll_find {
		template<typename T> T_unroll_find(T &mattings) {
			mattings[N].reset(new specialized_matting<N>());
		}
	};
	// -------------------------------------------------------------------------


	std::array<std::unique_ptr<general_inpaint>, MAXR> inpainters;
	unroll_constructor<MAXR, T_unroll_find> fill_filter;
public:
	optimized_inpaint() : fill_filter(inpainters) {}

	// C++ optimized search
	inline void process(const int x, const int y, MexImage<uint8_t> &Image, MexImage<uint8_t> &Trimap, MexImage<float> &Alpha, MexImage<uint8_t> &Fg, MexImage<uint8_t> &Bg, unsigned n)
	{
		auto const& inpainter = inpainters[n];
		inpainter->process(x, y, Image, Trimap, Alpha, Fg, Bg);
	}
};


void mexFunction(const int nout, mxArray* output[], const int in, const mxArray* input[])
{
#ifndef _DEBUG
	omp_set_num_threads(std::max<int>(4, omp_get_max_threads()));
	omp_set_dynamic(std::max<int>(2, omp_get_max_threads() / 2));
#endif

	if (in != 5 || nout != 4 || mxGetClassID(input[0]) != mxUINT8_CLASS || mxGetClassID(input[1]) != mxUINT8_CLASS || mxGetClassID(input[2]) != mxSINGLE_CLASS || mxGetClassID(input[3]) != mxSINGLE_CLASS)
	{
		mexErrMsgTxt("USAGE: [Li, Ri, DLi, DRi] = stereoscopic_inpainting(uint8(L), uint8(R), single(DL), single(DR), radius);");
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