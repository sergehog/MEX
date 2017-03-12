/** Interpolation between rectified stereo frames using Gaussian Aggregation (holes allowed)
*	@file interpolate_stereo.cpp
*	@date 27.10.2015
*	@author Sergey Smirnov <sergey.smirnov@tut.fi>
*/
/**
*	@file translate_stereo.cpp
*	@date 27.04.2012
*	@author Sergey Smirnov <sergey.smirnov@tut.fi>
*/

#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}
#endif // __cplusplus
//#include "../common/defines.h"
#include "../common/meximage.h"
#include <cmath>
#include <algorithm>
#include <memory>

typedef unsigned char uint8;
using namespace mymex;

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")


void fast_filtering(const MexImage<float> &Signal, const MexImage<float> &Temporal, const float sigma)
{
	const int width = Signal.width;
	const int height = Signal.height;
	const int layers = Signal.layers;
	const long HW = Signal.layer_size;
	std::unique_ptr<float[]> temporal1(new float[std::max(width, height)]);
	std::unique_ptr<float[]> temporal2(new float[std::max(width, height)]);

	for (int l = 0; l<layers; l++)
	{
		// Temporal.setval(0.f);
		// Horisontal (left-to-right & right-to-left) pass
		for (int y = 0; y<height; y++)
		{
			temporal1[0] = Signal(0, y, l);
			temporal2[width - 1] = Signal(width - 1, y, l);
			for (int x1 = 1; x1<width; x1++)
			{
				const int x2 = width - x1 - 1;
				temporal1[x1] = (Signal(x1, y, l) + temporal1[x1 - 1] * sigma);
				temporal2[x2] = (Signal(x2, y, l) + temporal2[x2 + 1] * sigma);
			}
			for (int x = 0; x<width; x++)
			{
				Temporal(x, y) = (temporal1[x] + temporal2[x]) - Signal(x, y, l);
			}
		}

		// Vertical (up-to-down & down-to-up) pass
		for (int x = 0; x<width; x++)
		{
			temporal1[0] = Temporal(x, 0);
			temporal2[height - 1] = Temporal(x, height - 1);
			for (int y1 = 1; y1<height; y1++)
			{
				const int y2 = height - y1 - 1;
				temporal1[y1] = (Temporal(x, y1) + temporal1[y1 - 1] * sigma);
				temporal2[y2] = (Temporal(x, y2) + temporal2[y2 + 1] * sigma);
			}
			for (int y = 0; y<height; y++)
			{
				Signal(x, y, l) = (temporal1[y] + temporal2[y]) - Temporal(x, y);
			}
		}
	}
}


//template<typename T> void render(T*, float*, T*, float*, bool*, size_t, size_t, size_t);

const float skip_delta = 3;
const unsigned skip_numb = 1;

void mexFunction(int nout, mxArray* output[], int in, const mxArray* input[])
{
	if (in < 4 || in > 10 || nout != 2
		|| mxGetClassID(input[0]) != mxSINGLE_CLASS || mxGetClassID(input[1]) != mxSINGLE_CLASS
		|| mxGetClassID(input[2]) != mxSINGLE_CLASS || mxGetClassID(input[3]) != mxSINGLE_CLASS)
	{
		mexErrMsgTxt("USAGE: [Virtual, Disp] = interpolate_stereo(single(L), single(R), single(DispL), single(DispR), <position=0...1, radius, sigma_distance, fill_depth, fill_image, average>); ");
	}

	MexImage<float> Left(input[0]);
	MexImage<float> Right(input[1]);
	MexImage<float> DispL(input[2]);
	MexImage<float> DispR(input[3]);

	const int height = Left.height;
	const int width = Left.width;
	const long HW = Left.layer_size;
	const int colors = Left.layers;
	const float nan = sqrt(-1.f);
	const float disp_thr = 2.f;
	if (height != DispL.height || width != DispL.width || height != DispR.height || width != DispR.width)
	{
		mexErrMsgTxt("ERROR: Sizes of all images must be the same.");
	}
	if (height != Right.height || width != Right.width || colors != Right.layers)
	{
		mexErrMsgTxt("ERROR: Sizes of all images must be the same.");
	}

	const float position = (in > 4) ? std::max(-1.f, std::min(2.f, (float)mxGetScalar(input[4]))) : 0.5f;
	const unsigned radius = (in > 5) ? (unsigned)mxGetScalar(input[5]) : 1;
	const float sigma = (in > 6) ? (float)mxGetScalar(input[6]) : 0.8f;
	const bool fill_depth = (in > 7) ? (bool)mxGetScalar(input[7]) : true;
	const bool fill_images = (in > 8) ? (bool)mxGetScalar(input[8]) : true;
	const bool average = (in > 9) ? (bool)mxGetScalar(input[9]) : false;

	size_t dims[] = { (unsigned)height, (unsigned)width, (unsigned)colors };
	size_t dims2d[] = { (unsigned)height, (unsigned)width, 1 };

	output[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
	output[1] = mxCreateNumericArray(3, dims2d, mxSINGLE_CLASS, mxREAL);

	MexImage<float> Rendered(output[0]);
	MexImage<float> Disparity(output[1]);

	if (position == 0.f)
	{
		for (int i = 0; i < HW; i++)
		{
			Disparity[i] = DispL[i];
			for (int c = 0; c < colors; c++)
			{
				Rendered[i + c*HW] = Left[i + c*HW];
			}
		}
		return;
	}

	if (position == 1.f)
	{
		for (int i = 0; i < HW; i++)
		{
			Disparity[i] = DispR[i];
			for (int c = 0; c < colors; c++)
			{
				Rendered[i + c*HW] = Right[i + c*HW];
			}
		}
		return;
	}


	Disparity.setval(nan);

	// check for mindisp/maxdisp
	float mindisp = DispL[0];
	float maxdisp = DispL[0];
#pragma omp for
	for (int i = 1; i < HW; i++)
	{
		mindisp = !isnan(DispL[i]) && (DispL[i] < mindisp || isnan(mindisp)) ? DispL[i] : mindisp;
		maxdisp = !isnan(DispL[i]) && (DispL[i] > maxdisp || isnan(maxdisp)) ? DispL[i] : maxdisp;
		mindisp = !isnan(DispR[i]) && (DispR[i] < mindisp || isnan(mindisp)) ? DispR[i] : mindisp;
		maxdisp = !isnan(DispR[i]) && (DispR[i] > maxdisp || isnan(maxdisp)) ? DispR[i] : maxdisp;
	}
	
	// Left -> Virtual pass
#pragma omp for
	for (long i = 0; i < HW; i++)
	{
		const int x = i / height;
		const int y = i % height;
		const float disp = DispL[i];
		if (isnan(disp))
		{
			continue;
		}

		const int xv = std::round(x - disp*position);
		if (xv < 0 || xv >= width)
		{
			continue;
		}
		const float dispv = Disparity(xv, y);
		if (isnan(dispv) || dispv < disp)
		{
			Disparity(xv, y) = disp;
		}
	}

	// Right -> Virtual pass
#pragma omp for
	for (long i = 0; i < HW; i++)
	{
		const int x = i / height;
		const int y = i % height;
		const float disp = DispR[i];
		if (isnan(disp))
		{
			continue;
		}

		const int xv = std::round(x + disp*(1 - position));
		if (xv < 0 || xv >= width)
		{
			continue;
		}
		const float dispv = Disparity(xv, y);
		if (isnan(dispv) || dispv < disp)
		{
			Disparity(xv, y) = disp;
		}
	}

	// fill holes in the virtual disparity map
	MexImage<float> Temporal3(width, height, 3);
	MexImage<float> Weights(width, height);	
	MexImage<float> TemporalC(width, height);

	if (fill_depth)
	{
		MexImage<float> DispTmp(Temporal3, 0);
		//DispTmp.setval(11.f);
		//Weights.setval(0.f);
		//TemporalC.setval(0.f);
		// background-favored weighting scheme
#pragma omp for
		for (long i = 0; i < HW; i++)
		{			
			Weights[i] = isnan(Disparity[i]) ? 0.f : (maxdisp - Disparity[i]);
			DispTmp[i] = isnan(Disparity[i]) ? 0.f : Disparity[i] * Weights[i];
		}
		fast_filtering(DispTmp, TemporalC, sigma);
		fast_filtering(Weights, TemporalC, sigma);
				
		// predict disparity if absent
#pragma omp for
		for (long i = 0; i < HW; i++)
		{			
			Disparity[i] = isnan(Disparity[i]) ? DispTmp[i] / Weights[i] : Disparity[i];			
		}
	}
	Rendered.setval(nan);
	Weights.setval(nan);

#pragma omp for
	for (long i = 0; i < HW; i++)
	{
		const int x = i / height;
		const int y = i % height;
		const float disp = Disparity[i];

		const float xl = x + disp*position;
		if (!isnan(disp) && xl > 0.f && xl <= float(width - 2))
		{
			const int xl1 = std::floor(xl);
			const float wl = (xl - xl1);
			const float dispL = DispL(xl1, y)*(1 - wl) + DispL(xl1 + 1, y)*wl;
			const float w = abs(disp - dispL);
			if (!isnan(dispL) && dispL < disp + 1)
			{
				Weights(x, y) = w;	
				for (int c = 0; c < colors; c++)
				{
					Rendered(x, y, c) = (Left(xl1, y, c)*(1 - wl) + Left(xl1 + 1, y, c)*wl);
				}
			}
		}

		const float xr = x - disp*(1 - position);
		if (xr > 0.f && xr <= float(width - 2))
		{
			const int xr1 = std::floor(xr);
			const float wr = (xr - xr1);
			const float dispR = DispR(xr1, y)*(1 - wr) + DispR(xr1 + 1, y)*wr;
			const float wu = abs(disp - dispR);
			if (!isnan(dispR) && dispR < disp + 1)
				//if (w<1)
				if (average && !isnan(Weights(x, y)))
				{
					const float w1 = 1.f - Weights(x, y) / (Weights(x, y) + wu);
					const float w2 = 1.f - wu / (Weights(x, y) + wu);
					for (int c = 0; c < colors; c++)
					{
						const float right = (Right(xr1, y, c)*(1 - wr) + Right(xr1 + 1, y, c)*wr);
						Rendered(x, y, c) = w1 * Rendered(x, y, c) + w2 * right;
					}
				}
				else if (isnan(Weights(x, y)) || wu < Weights(x, y))
				{
					Weights(x, y) = wu;
					for (int c = 0; c < colors; c++)
					{
						Rendered(x, y, c) = (Right(xr1, y, c)*(1 - wr) + Right(xr1 + 1, y, c)*wr);
					}
				}
		}
	}
	
	if (fill_images)
	{
		MexImage<float> ToFilter(width, height, colors);

#pragma omp for
		for (long i = 0; i < HW; i++)
		{
			if (/*isnan(Weights[i]) ||*/ isnan(Disparity[i]) || isnan(Rendered[i]))
			{
				for (int c = 0; c < colors; c++)
				{
					ToFilter[i + c*HW] = 0;
				}
				Weights[i] = 0;
			}
			else
			{
				Weights[i] = (maxdisp - Disparity[i]);
				for (int c = 0; c < colors; c++)
				{
					ToFilter[i + c*HW] = Rendered[i + c*HW] * Weights[i];
				}
			}
		}
		fast_filtering(Weights, TemporalC, sigma);
		fast_filtering(ToFilter, Temporal3, sigma);

#pragma omp for
		for (long i = 0; i < HW; i++)
		{
			if (isnan(Rendered[i]))
			{
				for (int c = 0; c < colors; c++)
				{
					Rendered[i + c*HW] = ToFilter[i + c*HW] / Weights[i];					
				}
			}
		}
	}
	/*else
	{
#pragma omp for
		for (long i = 0; i < HW; i++)
		{
			if (isnan(Rendered[i]))
			{
				for (int c = 0; c < colors; c++)
				{
					Rendered[i + c*HW] /= Weights[i];
				}
			}
		}
	}*/

}

