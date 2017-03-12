/** Alpha-Matting with spiral sample selection and smooth alpha constraint
* @file spiral_matting.cpp
* @author Sergey Smirnov <sergey.smirnov@tut.fi>
* @date 21.07.2015
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

#define MAXN 200
#define colors 3
#define SMOOTH_ALPHA 11
#define BEST_ERRORS 10
#define SAMPLE_UNIQNESS_THR 3

//typedef unsigned char uint8;
using namespace mymex;

class color
{
	uint8_t values[colors];
public:
	color()
	{
		for (int c = 0; c < colors; c++)
		{
			values[c] = 0;
		}
	}

	color(const int x, const int y, MexImage<uint8_t> &img)
	{
		read(x, y, img);
	}

	uint8_t & operator[](int i)
	{
		return values[i];
	}

	void read(const int x, const int y, MexImage<uint8_t> &img)
	{
		for (int c = 0; c < colors; c++)
		{
			values[c] = img(x, y, c);
		}
	}

	// L1 distance (integer typed)
	unsigned diff(const color &a)
	{
		int diff = 0;
		for (int c = 0; c < colors; c++)
		{
			diff += abs(int(values[c]) - int(a.values[c]));
		}
		return unsigned(diff);
	}

	// L2 distance 
	float dist(const color &a)
	{
		float dist = 0;
		for (int c = 0; c < colors; c++)
		{
			int d = abs(int(values[c]) - int(a.values[c]));
			dist += (float(d)*(float(d)));
		}
		return sqrt(dist);
	}

	void write(MexImage<uint8_t> &img, const int x, const int y)
	{
		for (int c = 0; c < colors; c++)
		{
			img(x, y, c) = values[c];
		}
	}

	//static color readone(const int x, const int y, MexImage<uint8_t> &img)
	//{
	//	color sample;
	//	sample.read(x, y, img);
	//	return sample;
	//}
};


template<unsigned N>
struct samples
{
	color data[N];
	int number = 0;

	void add_sample(color sample)
	{
		if (number < N)
		{
			for (int i = 0; i < number; i++)
			{
				if (data[i].diff(sample) < SAMPLE_UNIQNESS_THR)
				{
					return;
				}
			}
			data[number++] = sample;
		}
	}

	void add_sample(const int x, const int y, MexImage<uint8_t> &img)
	{
		color sample;
		sample.read(x, y, img);
		add_sample(sample);
	}

	bool can_add()
	{
		return (number < N);
	}

	color &operator[](int i)
	{
		return data[i];
	}

};

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
struct general_matting {

	virtual void process(MexImage<uint8_t> &Image, MexImage<float> &CostVolume, MexImage<float> &Alpha, MexImage<uint8_t> &Fg, MexImage<uint8_t> &Bg, const float cost_thr) = 0;
};



// The filters for each search variant (range_filters)
template<unsigned N>
struct specialized_matting : general_matting {

	void process(MexImage<uint8_t> &Image, MexImage<float> &CostVolume, MexImage<float> &Alpha, MexImage<uint8_t> &Fg, MexImage<uint8_t> &Bg, const float cost_thr)
	{
		const int width = Image.width;
		const int height = Image.height;
		const long HW = Image.layer_size;
		//float const cost_thr = 30.f;

#pragma omp parallel for
		for (long i = 0; i < HW; i++)
		{
			const int x = i / height;
			const int y = i % height;

			if (!isnan(Alpha[i]))
			{
				for (int a = 0; a < SMOOTH_ALPHA; a++)
				{
					const float alpha = float(a) / (SMOOTH_ALPHA - 1); // 0; 0.1f; 0.2f; ..., 0.9f; 1.f.
					CostVolume(x, y, a) = abs(alpha - Alpha[i]) * cost_thr;
					//CostVolume(x, y, a) = abs(alpha - Alpha[i]) * 1.f;
				}
				continue;
			}

			//Collected Bg Samples 
			samples<N> bg_samples;			

			//Collected Fg Samples 
			samples<N> fg_samples;

			// radius of current window (will grow)
			int r = 1;

			// Spiral sample selection is here
			while (bg_samples.can_add() && fg_samples.can_add()) // fg_n < N && bg_n < N)
			{
				for (int di = -r; di < r; di++)
				{
					const int x1 = x + di;
					const int y1 = y - r;
					const int x2 = x + r;
					const int y2 = y + di;
					const int x3 = x - di;
					const int y3 = y + r;
					const int x4 = x - r;
					const int y4 = y - di;

					// upper left corner to right
					if (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height)
					{
						if (Alpha(x1, y1) == 0.f)
						{
							bg_samples.add_sample(x1, y1, Image);
						}
						else if (Alpha(x1, y1) == 1.f)
						{
							fg_samples.add_sample(x1, y1, Image);
						}
					}

					// upper right corner to down
					if (x2 >= 0 && x2 < width && y2 >= 0 && y2 < height)
					{
						if (Alpha(x2, y2) == 0.f)
						{
							bg_samples.add_sample(x2, y2, Image);
						}
						else if (Alpha(x2, y2) == 1.f)
						{
							fg_samples.add_sample(x2, y2, Image);
						}
					}

					// lower right corner to left
					if (x3 >= 0 && x3 < width && y3 >= 0 && y3 < height)
					{
						if (Alpha(x3, y3) == 0.f)
						{
							bg_samples.add_sample(x3, y3, Image);
						}
						else if (Alpha(x3, y3) == 1.f)
						{
							fg_samples.add_sample(x3, y3, Image);
						}
					}

					// lower left corner to up
					if (x4 >= 0 && x4 < width && y4 >= 0 && y4 < height)
					{
						if (Alpha(x4, y4) == 0.f)
						{
							bg_samples.add_sample(x4, y4, Image);
						}
						else if (Alpha(x4, y4) == 1.f)
						{
							fg_samples.add_sample(x4, y4, Image);
						}
					}
				}
				r++;
			}

			//color bg_alpha_samples[SMOOTH_ALPHA];
			//color fg_alpha_samples[SMOOTH_ALPHA];

#pragma omp parallel for
			for (int a = 0; a < SMOOTH_ALPHA; a++)
			{
				const float alpha = float(a) / (SMOOTH_ALPHA - 1); // 0; 0.1f; 0.2f; ..., 0.9f; 1.f.
				float best_err[BEST_ERRORS];
				for (int j = 0; j < BEST_ERRORS; j++)
				{
					best_err[j] = FLT_MAX;
				}

				for (int bg_n = 0; bg_n < N; bg_n++)
				{
					for (int fg_n = 0; fg_n < N; fg_n++)
					{
						float err = 0.f;
						for (int c = 0; c < colors; c++)
						{
							float diff = float(Image(x, y, c)) - float(alpha * fg_samples[fg_n][c] + (1 - alpha) * bg_samples[bg_n][c]);
							err += diff*diff;
						}


						if (err < best_err[BEST_ERRORS - 1])
						{
							best_err[BEST_ERRORS - 1] = err;

							for (int j = BEST_ERRORS - 2; j >= 0; j--)
							{
								if (best_err[j + 1] < best_err[j])
								{
									const float tmp = best_err[j];
									best_err[j] = best_err[j + 1];
									best_err[j + 1] = tmp;
								}
							}
						}
						
						
					}
				}
				
				float best_e = 0;
				for (int j = 0; j < BEST_ERRORS; j++)
				{
					best_e += sqrt(best_err[j]);
				}
				best_e /= BEST_ERRORS;

				CostVolume(x, y, a) = best_e > cost_thr ? cost_thr : best_e;
			}

		}
	}
};


class optimized_matting {
	// unroll templates
	template<unsigned N>
	struct T_unroll_find {
		template<typename T> T_unroll_find(T &mattings) {
			mattings[N].reset(new specialized_matting<N>());
		}
	};
	// -------------------------------------------------------------------------


	std::array<std::unique_ptr<general_matting>, MAXN> mattings;
	unroll_constructor<MAXN, T_unroll_find> fill_filter;
public:
	optimized_matting() : fill_filter(mattings) {}

	// C++ optimized search
	inline void process(MexImage<uint8_t> &Image, MexImage<float> &CostVolume, MexImage<float> &Alpha, MexImage<uint8_t> &Fg, MexImage<uint8_t> &Bg, const unsigned n, const float cost_thr)
	{
		auto const& matting = mattings[n];
		matting->process(Image, CostVolume, Alpha, Fg, Bg, cost_thr);
	}
};


void mexFunction(const int nout, mxArray* output[], const int in, const mxArray* input[])
{
#ifndef _DEBUG
	omp_set_num_threads(std::max<int>(4, omp_get_max_threads()));
	omp_set_dynamic(std::max<int>(2, omp_get_max_threads() / 2));
#endif

	if (in != 4 || nout != 4 || mxGetClassID(input[0]) != mxUINT8_CLASS || mxGetClassID(input[1]) != mxUINT8_CLASS)
	{
		mexErrMsgTxt("USAGE: [Alpha, Fg, Bg, Cost] = spiral_matting(uint8(Image), uint8(Trimap), N, thr);");
	}

	MexImage<uint8_t> Image(input[0]);
	MexImage<uint8_t> Trimap(input[1]);

	//const float alpha = std::max<float>(0.f, std::min<float>(1.f, mxGetScalar(input[2])));
	const unsigned n = std::max<unsigned>(2u, std::min<unsigned>(100, mxGetScalar(input[2])));
	const float cost_thr = float(mxGetScalar(input[3]));
	const int width = Image.width;
	const int height = Image.height;
	const long HW = Image.layer_size;
	const float nan = sqrt(-1.f);
	const size_t dims[] = { (size_t)height, (size_t)width, (size_t)colors };
	const size_t dims1[] = { (size_t)height, (size_t)width, 1 };
	const size_t dims11[] = { (size_t)height, (size_t)width, (size_t)SMOOTH_ALPHA};
	output[0] = mxCreateNumericArray(3, dims1, mxSINGLE_CLASS, mxREAL);
	output[1] = mxCreateNumericArray(3, dims, mxUINT8_CLASS, mxREAL);
	output[2] = mxCreateNumericArray(3, dims, mxUINT8_CLASS, mxREAL);
	output[3] = mxCreateNumericArray(3, dims11, mxSINGLE_CLASS, mxREAL);
	MexImage<float> Alpha(output[0]);
	MexImage<uint8_t> Fg(output[1]);
	MexImage<uint8_t> Bg(output[2]);
	//MexImage<float> DistFg(width, height);
	//MexImage<float> DistBg(width, height);
	//MexImage<float> CostVolume(width, height, SMOOTH_ALPHA);
	MexImage<float> CostVolume(output[3]);

	optimized_matting matting;

	//DistFg.setval(nan);
	//DistBg.setval(nan);
	//unsigned UNKNOWN = 0;
	//MexImage<unsigned> UMap(width, height);

#pragma omp parallel for
	for (long i = 0; i < HW; i++)
	{
		if (Trimap[i] == 0)
		{
			Alpha[i] = 0.f;
			for (int c = 0; c < colors; c++)
			{
				Bg[i + HW*c] = Image[i + HW*c];
			}
		}
		else if (Trimap[i] == 255 || Trimap[i] == 2)
		{
			Alpha[i] = 1.f;
			for (int c = 0; c < colors; c++)
			{
				Fg[i + HW*c] = Image[i + HW*c];
			}
		}
		else
		{
			Alpha[i] = nan;
		}
	}

	matting.process(Image, CostVolume, Alpha, Fg, Bg, n, cost_thr);
}