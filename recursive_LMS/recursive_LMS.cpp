/** recursive_LMS - analogue of recursive gaussian, but can also handle texures, similarly as NLM filter
* @file recursive_LMS.cpp
* @author Sergey Smirnov <sergey.smirnov@tut.fi>
* @date 06.05.2015
* @copyright 3D Media Group / Tampere University of Technology
*/


#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}
#endif // __cplusplus

#include "../common/meximage.h"
//#include "../common/defines.h"
//#include <vector>
#include <queue>
#include <algorithm>
#include <memory>
#include <array>
#define _USE_MATH_DEFINES
#include <cmath>
#ifndef _DEBUG
#include <omp.h>
#endif

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")

using namespace mymex;

float getWeight(const MexImage<float> &Image, const int x1, const int y1, const int x2, const int y2, const float * const weights_lookup);


template<unsigned N>
class fixed_queue
{
	float queue[N];

public:
	
	fixed_queue() : set(0.f)
	{
	}

	void set(const float value)
	{
		for (int n = 0; n < N; n++)
		{
			queue[n] = value;
		}
	}

	void push(const float value)
	{
		for (int n = N - 1; n > 0; n--)
		{
			queue[n] = queue[n - 1];
		}
		queue[0] = value;
	}

	float &operator[](const int n)
	{
		return queue[n];
	}
};
// base (general) struct for polymorpic filter access
struct base_filter
{
	virtual void filter_signal(MexImage<float> &Signal, MexImage<float> &Image, MexImage<float> &Filtered, const float alpha, const float sigma_color)
	{};
};


// inherited filtering-struct, optimized for particular "colors" and "N"
template<unsigned colors, unsigned N>
struct optimized_filter : base_filter
{
	void filter_signal(MexImage<float> &Signal, MexImage<float> &Image, MexImage<float> &Filtered, const float alpha, const float sigma_color)
	{
		const int layers = Signal.layers;
		const int width = Signal.width;
		const int height = Signal.height;
		const int HW = Signal.layer_size;

		MexImage<float> Temporal(width, height, layers);
		MexImage<float> Weights(width, height);
		MexImage<float> Weights2(width, height);
		Weights.setval(0.f);
		Weights2.setval(0.f);
		Temporal.setval(0.f);
		Filtered.setval(0.f);

		std::unique_ptr<float[]> weights_lookup(new float[256 * colors]);

		for (int i = 0; i < 256 * colors; i++)
		{
			weights_lookup[i] = float(exp(-double(i) / (255 * colors * sigma_color)));
		}

		#pragma omp parallel 
		{
			// for each signal's layer we need its own fixed_queue for forward and reverse filtering
			std::unique_ptr<fixed_queue<N>[]> y_fwd(new fixed_queue<N>[layers]);
			std::unique_ptr<fixed_queue<N>[]> y_rev(new fixed_queue<N>[layers]);
			float weight_fwd, weight_rev;
			#pragma omp for
			for (int x = 0; x<width; x++)
			{
				// setup y-s and intermediate (Temporal) filtering result.
				for (int l = 0; l<layers; l++)
				{
					y_fwd[l].set(Signal(x, 0, l));
					y_rev[l].set(Signal(x, height - 1, l));
					Temporal(x, 0, l) = y_fwd[l][0];
					Temporal(x, height - 1, l) = y_rev[l][0];
				}
				weight_fwd = 1.f;
				weight_rev = 1.f;
	
				for (int y = 1; y < height; y++)
				{
					for (int l = 0; l < layers; l++)
					{
						float y = Signal(x, y, l);
						
						for (int n = 0; n < N && y > n; n++)
						{
							const int yn = y - 1 - n;

							//y_fwd[l][n]
							
							
						}

					}
				}

				/*for (int y = 2; y<height; y++)
				{
					for (int l = 0; l<layers; l++)
					{
						float y_0 = Signal(x, y, l) + a*y_1[l];
						Temporal(x, y, l) += y_0;
						y_2[l] = y_1[l];
						y_1[l] = y_0;
	
						float ry_0 = Signal(x, height - y - 1, l) + a*ry_1[l];
						Temporal(x, height - y - 1, l) += ry_0;
						ry_2[l] = ry_1[l];
						ry_1[l] = ry_0;
					}
				}*/
			}
		}
	
		#pragma omp parallel for
		for (long i = 0; i<HW*layers; i++)
		{
			Temporal[i] -= Signal[i];
		}
		/*
		#pragma omp parallel 
		{
			std::unique_ptr<float[]> y_1(new float[layers*N]);
			std::unique_ptr<float[]> y_2(new float[layers*N]);
	
			std::unique_ptr<float[]> ry_1(new float[layers*N]);
			std::unique_ptr<float[]> ry_2(new float[layers*N]);
	
			#pragma omp for
			for (int y = 0; y<height; y++)
			{
				for (int l = 0; l<layers; l++)
				{
					y_2[l] = Temporal(0, y, l);
					y_1[l] = Temporal(1, y, l) + a * y_2[l];
					Filtered(0, y, l) += y_2[l];
					Filtered(1, y, l) += y_1[l];
	
					ry_2[l] = Temporal(width - 1, y, l);
					ry_1[l] = Temporal(width - 2, y, l) + a * ry_2[l];
					Filtered(width - 1, y, l) += ry_2[l];
					Filtered(width - 2, y, l) += ry_1[l];
				}
	
				for (int x = 2; x<width; x++)
				{
					for (int l = 0; l<layers; l++)
					{
						float y_0 = Temporal(x, y, l) + a*y_1[l];
						Filtered(x, y, l) += y_0;
						y_2[l] = y_1[l];
						y_1[l] = y_0;
	
						float ry_0 = Temporal(width - x - 1, y, l) + a*ry_1[l];
						Filtered(width - x - 1, y, l) += ry_0;
						ry_2[l] = ry_1[l];
						ry_1[l] = ry_0;
					}
				}
			}
		}
	
		#pragma omp parallel for
		for (long i = 0; i<HW*layers; i++)
		{
			Filtered[i] -= Temporal[i];
		}*/
	}
};

#define MAXCOLORS 3
#define MAXN 20
typedef std::array<base_filter*, (MAXCOLORS)*(MAXN)> filter_array;


template<unsigned colors, unsigned N>
class unroll_N
{
private:
	optimized_filter<colors, N> * filter;
	unroll_N<colors, N - 1> unroller;
public:
	unroll_N(filter_array table) : unroller(table), filter(new optimized_filter<colors, N>())
	{
		table[(colors-1)*MAXN + (N-1)] = filter;
	}

	~unroll_N()
	{
		delete filter;
	}

};

template<unsigned colors>
class unroll_N<colors, 0>
{	
public:
	unroll_N(filter_array table)
	{
		//end of unrolling
	}
};

template<unsigned colors, unsigned N>
struct unroll_color
{
	unroll_N<colors, N> unroller;
	unroll_color<colors-1, N> unroller_color;
	unroll_color(filter_array table) : unroller(table), unroller_color(table)
	{		
	}
};

template< unsigned N>
struct unroll_color<0, N>
{
	unroll_color(filter_array table)
	{
		//end of unrolling
	}
};


void mexFunction(int nout, mxArray* output[], int in, const mxArray* input[])
{
	static filter_array table;	
	unroll_color<MAXCOLORS, MAXN> unroller(table);
	

#ifndef _DEBUG
	omp_set_num_threads(std::max<int>(4, omp_get_max_threads()));
	omp_set_dynamic(std::max<int>(2, omp_get_max_threads() / 2));
#endif

	//if (in != 2 || nout < 1 || nout > 2 || mxGetClassID(input[0]) != mxSINGLE_CLASS)
	if (in < 3 || in > 4 || nout != 1 || mxGetClassID(input[0]) != mxSINGLE_CLASS || mxGetClassID(input[1]) != mxSINGLE_CLASS)
	{
		mexErrMsgTxt("USAGE: [Filtered] = recursive_LMS(single(Signal), single(Image), N, alpha);");
	}
	
	MexImage<float> Signal(input[0]);
	MexImage<float> Image(input[1]);
	const unsigned N = (unsigned)mxGetScalar(input[2]);
	const float alpha = (in > 3) ? std::max<float>(0.f, std::min<float>(1.f, mxGetScalar(input[3]))) : 0.8;
	const float sigma_color = 0.9;

	const int width = Signal.width;
	const int height = Signal.height;
	const int layers = Signal.layers;
	const int colors = Image.layers;
	const long HW = Signal.layer_size;

	if (Image.width != width || Image.height != height)
	{
		mexErrMsgTxt("Resolution of input images must be the same!");
	}

	if (colors > MAXCOLORS)
	{
		mexErrMsgTxt("Too high number of colors in the Image!");
	}

	if (N > MAXN)
	{
		mexErrMsgTxt("Too big N value!");
	}

	const size_t dims[] = { (size_t)height, (size_t)width, (size_t)layers };
	output[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
	MexImage<float> Filtered(output[0]);	
	
	// polymorphic access to a proper filtering function
	table[(colors - 1)*MAXN + N - 1]->filter_signal(Signal, Image, Filtered, alpha, sigma_color);

}



float getWeight(const MexImage<float> &Image, const int x1, const int y1, const int x2, const int y2, const float * const weights_lookup)
{
	int max_diff = 0;
	for (int c = 0; c < Image.layers; c++)
	{
		int diff = int(abs(Image(x1, y1, c) - Image(x2, y2, c)));
		max_diff = diff > max_diff ? diff : max_diff;
	}
	return weights_lookup[max_diff];
}

