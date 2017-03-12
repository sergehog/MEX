//simple_matting
/**
* @file recursive_gaussian.cpp
* @author Sergey Smirnov <sergey.smirnov@tut.fi>
* @date 29.10.2014
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
#define BEST_ERRORS 5
#define MAXN 500
#define SAMPLE_UNIQNESS_THR 5
#define DISTANCE_WEIGHT 100
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
	int Xs[N];
	int Ys[N];
	int number = 0;

	void add_sample(const int x, const int y, MexImage<uint8_t> &img)
	{
		color sample;
		sample.read(x, y, img);
		if (number < N)
		{
			for (int i = 0; i < number; i++)
			{
				if (data[i].diff(sample) < SAMPLE_UNIQNESS_THR)
				{
					return;
				}
			}
			data[number] = sample;
			Xs[number] = x;
			Ys[number] = y;
			number++;
		}
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

	virtual void process(MexImage<uint8_t> &Image, MexImage<uint8_t> &Trimap, MexImage<float> &Alpha, MexImage<uint8_t> &Fg, MexImage<uint8_t> &Bg) = 0;
};

void add_sample_pair(const float err, const float alpha, const int bg_n, const int fg_n, float * const best_err, float * const best_alphas, int * const best_bg, int * const best_fg)
{
	if (err < best_err[BEST_ERRORS - 1])
	{
		best_err[BEST_ERRORS - 1] = err;
		best_alphas[BEST_ERRORS - 1] = alpha;
		best_bg[BEST_ERRORS - 1] = bg_n;
		best_fg[BEST_ERRORS - 1] = fg_n;

		for (int j = BEST_ERRORS - 2; j >= 0 && (best_err[j + 1] < best_err[j]); j--)
		{
			const float err_tmp = best_err[j];
			const float alpha_tmp = best_alphas[j];
			const float bg_tmp = best_bg[j];
			const float fg_tmp = best_fg[j];
			best_err[j] = best_err[j + 1];
			best_alphas[j] = best_alphas[j + 1];
			best_bg[j] = best_bg[j + 1];
			best_fg[j] = best_fg[j + 1];
			best_err[j + 1] = err_tmp;
			best_alphas[j + 1] = alpha_tmp;
			best_bg[j + 1] = bg_tmp;
			best_fg[j + 1] = fg_tmp;
		}
	}
}

// The filters for each search variant (range_filters)
template<unsigned N>
struct specialized_matting : general_matting {

	void process(MexImage<uint8_t> &Image, MexImage<uint8_t> &Trimap, MexImage<float> &Alpha, MexImage<uint8_t> &Fg, MexImage<uint8_t> &Bg)
	{
		const int width = Image.width;
		const int height = Image.height;
		const long HW = Image.layer_size;
		int zero_error_pixels = 1;
		MexImage<float> Error(width, height);
		Error.setval(FLT_MAX);

		while (zero_error_pixels > 0)
		{
			zero_error_pixels = 0;

			#pragma omp parallel for
			for (long i = 0; i < HW; i++)
			{
				const int x = i / height;
				const int y = i % height;

				if (!isnan(Alpha[i]))
				{
					continue;
				}

				//! Collected Samples 
				samples<N> bg_samples;
				samples<N> fg_samples;

				//! radius of current window
				int r = 1;

				// Spiral sample selection is here
				while (bg_samples.can_add() && fg_samples.can_add())
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
							if (Trimap(x1, y1) == 0)
							{
								bg_samples.add_sample(x1, y1, Image);
							}
							else if ((Trimap(x1, y1) == 2 || Trimap(x1, y1) == 255))
							{
								fg_samples.add_sample(x1, y1, Image);
							}
						}

						// upper right corner to down
						if (x2 >= 0 && x2 < width && y2 >= 0 && y2 < height)
						{
							if (Trimap(x2, y2) == 0)
							{
								bg_samples.add_sample(x2, y2, Image);
							}
							else if ((Trimap(x2, y2) == 2 || Trimap(x2, y2) == 255))
							{
								fg_samples.add_sample(x2, y2, Image);
							}
						}

						// lower right corner to left
						if (x3 >= 0 && x3 < width && y3 >= 0 && y3 < height)
						{
							if (Trimap(x3, y3) == 0)
							{
								bg_samples.add_sample(x3, y3, Image);
							}
							else if ((Trimap(x3, y3) == 2 || Trimap(x3, y3) == 255))
							{
								fg_samples.add_sample(x3, y3, Image);
							}
						}

						// lower left corner to up
						if (x4 >= 0 && x4 < width && y4 >= 0 && y4 < height)
						{
							if (Trimap(x4, y4) == 0)
							{
								bg_samples.add_sample(x4, y4, Image);
							}
							else if ((Trimap(x4, y4) == 2 || Trimap(x4, y4) == 255))
							{
								fg_samples.add_sample(x4, y4, Image);
							}
						}
					}
					r++;
				}


				float best_err[BEST_ERRORS];
				for (int j = 0; j < BEST_ERRORS; j++)
				{
					best_err[j] = FLT_MAX;
				}
				float best_alphas[BEST_ERRORS];
				int best_bg[BEST_ERRORS];
				int best_fg[BEST_ERRORS];

				//float best_err = FLT_MAX;

				// check for alpha = 0
				for (int bg_n = 0; bg_n < N; bg_n++)
				{
					const float err = color(x, y, Image).dist(bg_samples[bg_n]);
					const float dist_bg = sqrt(float((bg_samples.Xs[bg_n] - x)*(bg_samples.Xs[bg_n] - x) + (bg_samples.Ys[bg_n] - y)*(bg_samples.Ys[bg_n] - y)));
					const float energy = sqrt(err) + dist_bg / DISTANCE_WEIGHT;
					add_sample_pair(energy, 0.f, bg_n, 0, best_err, best_alphas, best_bg, best_fg);
				}

				// check for alpha = 1
				for (int fg_n = 0; fg_n < N; fg_n++)
				{
					float err = color(x, y, Image).dist(fg_samples[fg_n]);
					const float dist_fg = sqrt(float((fg_samples.Xs[fg_n] - x)*(fg_samples.Xs[fg_n] - x) + (fg_samples.Ys[fg_n] - y)*(fg_samples.Ys[fg_n] - y)));
					const float energy = sqrt(err) + dist_fg / DISTANCE_WEIGHT;

					add_sample_pair(energy, 1.f, 0, fg_n, best_err, best_alphas, best_bg, best_fg);
				}

				// Now check all Fg/Bg pairs with O(N^2) complexity
				for (int bg_n = 0; bg_n < N; bg_n++)
				{
					for (int fg_n = 0; fg_n < N; fg_n++)
					{
						float alpha = 0.f;
						for (int c = 0; c < colors; c++)
						{
							const float nom = (float(Image(x, y, c)) - bg_samples[bg_n][c])*(float(fg_samples[fg_n][c]) - bg_samples[bg_n][c]);
							const float denom = (float(fg_samples[fg_n][c]) - bg_samples[bg_n][c])*(float(fg_samples[fg_n][c]) - bg_samples[bg_n][c]);
							const float alpha_c = nom / denom;
							alpha += alpha_c < 0 ? 0 : (alpha_c > 1 ? 1 : alpha_c);
						}
						alpha /= 3;
						float err = 0.f;
						for (int c = 0; c < colors; c++)
						{
							const float diff = float(Image(x, y, c)) - float(alpha * fg_samples[fg_n][c] + (1 - alpha) * bg_samples[bg_n][c]);
							err += diff*diff;
						}
						const float dist_bg = sqrt(float((bg_samples.Xs[bg_n] - x)*(bg_samples.Xs[bg_n] - x) + (bg_samples.Ys[bg_n] - y)*(bg_samples.Ys[bg_n] - y)));
						const float dist_fg = sqrt(float((fg_samples.Xs[fg_n] - x)*(fg_samples.Xs[fg_n] - x) + (fg_samples.Ys[fg_n] - y)*(fg_samples.Ys[fg_n] - y)));
						const float energy = sqrt(err) + 0.5 * dist_fg / DISTANCE_WEIGHT + 0.5 * dist_bg / DISTANCE_WEIGHT;

						add_sample_pair(sqrt(err), alpha, bg_n, fg_n, best_err, best_alphas, best_bg, best_fg);
					}
				}

				// estimte weighted-averaged alpha over all best samples. 
				float weighted_alpha = 0;
				float accum_weight = 0;
				for (int j = 0; j < BEST_ERRORS; j++)
				{
					const float weight = 1 / (best_err[j] + 0.00001);
					weighted_alpha += best_alphas[j] * weight;
					accum_weight += weight;
				}

				// now use weighted alpha to select exact sample (among best ones)
				// So, we can also recover Bg and Fg!!
				weighted_alpha /= accum_weight;
				accum_weight = FLT_MAX;
				for (int j = 0; j < BEST_ERRORS; j++)
				{
					if (abs(best_alphas[j] - weighted_alpha) < accum_weight)
					{
						accum_weight = abs(best_alphas[j] - weighted_alpha);
						Alpha[i] = best_alphas[j];
						const int bg_n = best_bg[j];
						const int fg_n = best_fg[j];
						for (int c = 0; c < colors; c++)
						{
							Bg[i + c*HW] = bg_samples[bg_n][c];
							Fg[i + c*HW] = fg_samples[fg_n][c];
						}
					}
				}
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
	inline void process(MexImage<uint8_t> &Image, MexImage<uint8_t> &Trimap, MexImage<float> &Alpha, MexImage<uint8_t> &Fg, MexImage<uint8_t> &Bg, unsigned n)
	{
		auto const& matting = mattings[n];
		matting->process(Image, Trimap, Alpha, Fg, Bg);
	}
};


void mexFunction(const int nout, mxArray* output[], const int in, const mxArray* input[])
{
#ifndef _DEBUG
	omp_set_num_threads(std::max<int>(4, omp_get_max_threads()));
	omp_set_dynamic(std::max<int>(2, omp_get_max_threads() / 2));
#endif

	if (in != 3 || nout != 3 || mxGetClassID(input[0]) != mxUINT8_CLASS || mxGetClassID(input[1]) != mxUINT8_CLASS)
	{
		mexErrMsgTxt("USAGE: [Alpha, Fg, Bg] = simple_matting(uint8(Image), uint8(Trimap), N);");
	}

	MexImage<uint8_t> Image(input[0]);
	MexImage<uint8_t> Trimap(input[1]);
	//const float alpha = std::max<float>(0.f, std::min<float>(1.f, mxGetScalar(input[2])));
	const unsigned n = std::max<unsigned>(2u, std::min<unsigned>(MAXN, mxGetScalar(input[2])));
	const int width = Image.width;
	const int height = Image.height;
	const long HW = Image.layer_size;
	const float nan = sqrt(-1.f);
	const size_t dims[] = { (size_t)height, (size_t)width, (size_t)colors };
	const size_t dims1[] = { (size_t)height, (size_t)width, 1 };
	output[0] = mxCreateNumericArray(3, dims1, mxSINGLE_CLASS, mxREAL);
	output[1] = mxCreateNumericArray(3, dims, mxUINT8_CLASS, mxREAL);
	output[2] = mxCreateNumericArray(3, dims, mxUINT8_CLASS, mxREAL);
	MexImage<float> Alpha(output[0]);
	MexImage<uint8_t> Fg(output[1]);
	MexImage<uint8_t> Bg(output[2]);
	MexImage<float> DistFg(width, height);
	MexImage<float> DistBg(width, height);

	optimized_matting matting;	

	DistFg.setval(nan);
	DistBg.setval(nan);
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

	matting.process(Image, Trimap, Alpha, Fg, Bg, n);	
}