/** Inpainting for RGB+D images
*	@file inpaint_rgbd.cpp
*	@date 18.05.2016
*	@author Sergey Smirnov <sergey.smirnov@tut.fi>
*/

#include "../common/meximage.h"
#include <float.h>
#include <cmath>
#include <algorithm>
#include <vector>
#ifndef _DEBUG
#include <omp.h>
#endif

#include <string.h>

#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}
#endif // __cplusplus

typedef signed char int8;

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")

using namespace mymex;

template<int radius>
float calculateAverage(MexImage<float> &Confidence, const int x0, const int y0)
{
	const int width = Confidence.width;
	const int height = Confidence.height;

	int pixels = 0;
	float value = 0;
	for (int x = std::max(0, x0 - radius); x <= std::min(width - 1, x0 + radius); x++)
	{
		for (int y = std::max(0, y0 - radius); y <= std::min(height - 1, y0 + radius); y++)
		{
			value += Confidence(x, y);
			pixels++;
		}
	}
	return value / pixels;
}

template<int radius>
void calculateSource(MexImage<bool> &Valid, MexImage<bool> &Source)
{
	const int width = Valid.width;
	const int height = Valid.height;
	const int HW = Valid.layer_size;
	constexpr int diameter = radius * 2 + 1;
	constexpr int window = diameter*diameter;

	Source.setval(0);

#pragma omp parallel for
	for (int x = radius; x <= width - radius - 1; x++)
	{
		for (int y = radius; y <= height - radius - 1; y++)
		{
			const long index = Valid.Index(x, y);

			if (!Valid[index])
				continue;

			int pixels = 0;

			for (int i = 0; i<window; i++)
			{
				const int sx = x + i / diameter - radius;
				const int sy = y + i%diameter - radius;
				const long sindex = Valid.Index(sx, sy);

				if (!Valid[sindex])
					break;

				pixels++;
			}
			
			if (pixels == window)
			{
				Source[index] = true;
			}
		}
	}
}


void mexFunction(int nout, mxArray* output[], int in, const mxArray* input[])
{
#ifndef _DEBUG
	omp_set_num_threads(std::max<int>(4, omp_get_max_threads()));
	omp_set_dynamic(std::max<int>(2, omp_get_max_threads() / 2));;
#endif	

	if (in != 5 || nout != 1 || mxGetClassID(input[0]) != mxSINGLE_CLASS || mxGetClassID(input[1]) != mxSINGLE_CLASS || mxGetClassID(input[2]) != mxSINGLE_CLASS)
	{
		mexErrMsgTxt("USAGE: [FilledColor] = inpaint_rgbd(Color_NaNs, FullDepth, Priority, search, radius);\n");
	}

	MexImage<float> Color(input[0]);
	MexImage<float> Depth(input[1]);
	MexImage<float> Priority(input[2]);

	const int height = Color.height;
	const int width = Color.width;
	const int HW = Color.layer_size;
	const int colors = Color.layers;


	if (Depth.width != width || Depth.height != height || Depth.layers != 1)
	{
		mexErrMsgTxt("Wrong Depth dimensions!");
	}

	if (Priority.width != width || Priority.height != height || Priority.layers != 1)
	{
		mexErrMsgTxt("Wrong Priority dimensions!");
	}

	const int search = std::abs((int)mxGetScalar(input[3]));
	//const int radius = std::max(1, (int)mxGetScalar(input[4]));
	constexpr int radius = 5;
	constexpr int diameter = radius * 2 + 1;
	constexpr int window = diameter*diameter;
	const float depth_factor = 0.1;
	const float nan = sqrt(-1.f);

	matlab_size dims3d[] = { (matlab_size)height, (matlab_size)width, colors };
	//matlab_size dims2d[] = { (matlab_size)height, (matlab_size)width, 1 };

	output[0] = mxCreateNumericArray(3, dims3d, mxSINGLE_CLASS, mxREAL);

	MexImage<float> Filled(output[0]);
	MexImage<float> Confidence(width, height);

	MexImage<bool> Valid(width, height);
	MexImage<bool> Source(width, height);
	
	Valid.setval(false);
	Source.setval(false);

	unsigned long holes = 0;

	#pragma omp parallel for reduction(+:holes)
	for (long i = 0; i<HW; i++)
	{
		bool ishole = false;
		for (int c = 0; c<colors; c++)
		{
			ishole |= isnan(Color[i + c*HW]);
		}

		if (!ishole)
		{
			for (int c = 0; c<colors; c++)
			{
				Filled[i+c*HW] = Color[i + c*HW];
			}
		}
		holes += ishole;
		Valid[i] = !ishole;
	}

	for (long i = 0; i<HW; i++)
	{		
		float confIn = Priority[i];
		confIn = isnan(confIn) ? 0.f : confIn;
		confIn = confIn < 0.f ? 0.f : (confIn > 1.f ? 1.f : confIn);
		Confidence.data[i] = float(Valid[i]) * confIn;
	}

	calculateSource<radius>(Valid, Source);


	//bool hasHoles = true;
	unsigned iter = 0;
	long updated = 1;

	unsigned long long searches = 0;
	const unsigned long holes_saved = holes;

	MexImage<float> ConfidenceIntegral(width, height);

	// process goes iteratively
	while (holes > 0 && updated > 0)
	{
		updated = 0;
		
		ConfidenceIntegral.IntegralFrom(Confidence, true);
		long best_index = -1;
		float best_priority = -1.f;

		// order of inpainting - find appropriate hole :-)
		#pragma omp parallel for schedule(dynamic)
		for (long i = 0; i<HW; i++)
		{
			const int x = i / height;
			const int y = i % height;

			if (Valid[i])
			{
				continue;
			}

			//const float priority = sqrt(calculateAverage<radius>(Confidence, x, y));
			const float priority = sqrt(ConfidenceIntegral.getIntegralAverage(x, y, radius));

			if (priority > best_priority)
			{
				#pragma omp critical
				{
					best_priority = priority;
					best_index = i;
				}
			}
		}

		if (best_index < 0)
		{
			mexPrintf("ERROR: No appropriate patch to inpaint was found :-(\n");
			break;
		}

		const int best_x = best_index / height;
		const int best_y = best_index % height;
		
		// seek for the best valid patch over whole image o_O
		float best_diff = 1000000.f;
		long best_sindex = -1;
		
		unsigned long checked = 0;
		unsigned long runned = 0;

		#pragma omp parallel for reduction(+:checked,runned) schedule(dynamic)
		for(long i=0; i<HW; i++)
		{
			if (!Source[i])
			{
				continue;
			}
			const int x = i / height;
			const int y = i % height;
			if (abs(x - best_x) > search || abs(y - best_y) > search)
			{
				continue;
			}
			runned ++;
			
			float diff = 0;
			int pixels = 0;
			for (int w = 0; w < window; w++)
			{
				const int dx = w / diameter - radius;
				const int dy = w % diameter - radius;

				const int sx = x + dx;
				const int bx = best_x + dx;

				

				const int sy = y + dy;
				const int by = best_y + dy;
				
				if (bx < 0 || bx >= width || by < 0 || by >= height || !Valid(bx, by))
				{
					continue;
				}

				for (int c = 0; c<colors; c++)
				{
					diff += abs(Color(sx, sy, c) - Filled(bx, by, c));
				}
							
				diff += abs(Depth(sx, sy) - Depth(bx, by)) * depth_factor;

				if (diff > best_diff)
				{
					break;
				}
				//pixels++;
			}
			//for (int dx = -radius; dx <= radius; dx++)
			//{
			//	const int sx = x + dx;
			//	const int bx = best_x + dx;
			//	if (bx < 0 || bx >= width)
			//		continue;

			//	for (int dy = -radius; dy <= radius; dy++)
			//	{
			//		const int sy = y + dy;
			//		const int by = best_y + dy;

			//		if (by < 0 || by >= height)
			//			continue;

			//		const long sindex = Filled.Index(sx, sy);
			//		const long bindex = Filled.Index(bx, by);

			//		if (!Valid[bindex])
			//		{
			//			continue;
			//		}

			//		for (int c = 0; c<colors; c++)
			//		{
			//			diff += abs(Color[sindex + c*HW] - Filled[bindex + c*HW]);
			//		}
			//		
			//		diff += abs(Depth[sindex] - Depth[bindex]) * depth_factor;

			//		pixels++;
			//	}
			//}

			//diff /= pixels;

			if (diff < best_diff)
			{
				#pragma omp critical
				{
					best_diff = diff;
					best_sindex = i;
				}			
			}

			checked++;
		}

		if (best_sindex < 0)
		{
			mexPrintf("ERROR: No appropriate match for patch at (%d, %d) was found (%d (%d) patches were checked, ) \n", best_x, best_y, checked, runned);
			break;
		}

		searches += checked;

		const int best_sx = best_sindex / height;
		const int best_sy = best_sindex % height;

		#pragma omp parallel for
		for (int dx = -radius; dx <= radius; dx++)
		{
			const int sx = best_sx + dx;
			const int bx = best_x + dx;
			if (bx < 0 || bx >= width)
				continue;

			for (int dy = -radius; dy <= radius; dy++)
			{
				const int sy = best_sy + dy;
				const int by = best_y + dy;

				if (by < 0 || by >= height)
					continue;

				const long sindex = Filled.Index(sx, sy);
				const long bindex = Filled.Index(bx, by);

				if (Valid[bindex])
				{
					continue;
				}

				Valid[bindex] = true;
				Confidence[bindex] = best_priority;

				for (int c = 0; c<colors; c++)
				{
					Filled[bindex + c*HW] = Color[sindex + c*HW];
				}				

				updated++;
			}
		}


	}

	mexPrintf("Inpainted pixels: %d; using patches: %d; total matches: %d. \n", holes_saved - holes, iter, searches);

}

