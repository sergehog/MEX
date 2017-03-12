/**
*	@file get_hierarchical_reduction.cpp
*	@date 07.04.2016
*	@author Sergey Smirnov <sergey.smirnov@tut.fi>
*/

#include "../common/meximage.h"
#include "../get_laplacian/sparse_matrix.h"
#include <float.h>
#include <cmath>
#include <atomic>
//#include <algorithm>
#include <vector>
#include <memory>

#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}
#endif // __cplusplus
#ifndef _DEBUG
#include <omp.h>
#endif

#ifdef WIN32
#define isnan _isnan
#endif

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")

//#define IsNonZero(d) ((d)!=0.0)
//#define colors 3

using namespace mymex;

//#define radius 1
//#define diameter (radius*2 + 1)
//#define window (diameter*diameter)

#define colors 3

void mexFunction(int nout, mxArray* output[], int in, const mxArray* input[])
{
#ifndef _DEBUG
	omp_set_num_threads(std::max<int>(8, omp_get_max_threads()));
	omp_set_dynamic(std::max<int>(4, omp_get_max_threads() / 2));;
#endif	

	if (in < 1 || in > 3 || nout != 1 || mxGetClassID(input[0]) != mxSINGLE_CLASS)
	{
		mexErrMsgTxt("USAGE: [ReductionMatrix] = get_hierarchical_reduction(single(I), <factor, threshold>);\n");
	}

	MexImage<float> Image(input[0]);

	const uint8_t factor = in > 1 ? std::max<uint8_t>(2, uint8_t(mxGetScalar(input[1]))) : 2;
	const float threshold = in > 2 ? std::max<float>(1.f, mxGetScalar(input[2])) : 1.f;

	const int height = Image.height;
	const int width = Image.width;
	const size_t HW = size_t(width) * height;
	
	MexImage<int64_t> Indexes(width, height);
	for (size_t i = 0; i < HW; i++)
	{
		Indexes[i] = i;
	}

	const int new_width = width % factor ? width / factor + 1 : width / factor;
	const int new_height = height % factor ? height / factor + 1 : height / factor;

	MexImage<float> NewImage(new_width, new_height, colors);
	NewImage.setval(0.f);

	for (int x = 0; x < width; x++)
	{
		for (int y = 0; y < height; y++)
		{
			const int new_x = x / factor;
			const int new_y = y / factor;
			const int factor_x = (x < width - factor || width % factor == 0) ? factor : width-factor*(width / factor);
			const int factor_y = (y < height - factor || height % factor == 0) ? factor : height - factor*(height / factor);
			//const int factor_x = width % factor && (x == width - 1) ? 1 : factor;
			//const int factor_y = height % factor && (y == height - 1) ? 1 : factor;
			for (int c = 0; c < colors; c++)
			{
				NewImage(new_x, new_y, c) += Image(x, y, c) / (factor_x * factor_y);
			}
		}
	}

	// calculate error 
	MexImage<float> Error(new_width, new_height);
	Error.setval(0.f);

	for (int x = 0; x < width; x++)
	{
		for (int y = 0; y < height; y++)
		{
			const int new_x = x / factor;
			const int new_y = y / factor;
			const int factor_x = (x < width - factor || width % factor == 0) ? factor : width - factor*(width / factor);
			const int factor_y = (y < height - factor || height % factor == 0) ? factor : height - factor*(height / factor);
			float err = 0;
			for (int c = 0; c < colors; c++)
			{
				float diff = Image(x, y, c) - NewImage(new_x, new_y, c);
				err += diff *diff;
			}
			Error(new_x, new_y) += sqrt(err) / (factor_x*factor_y);
		}
	}


	// required too see if some pixel already shown up
	int64_t prev_index = 0;
	int64_t max_seen_index = -1;
	size_t elements = 0;
	// threshold and re-calculate indexes
	for (int x = 0; x < width; x++)
	{
		for (int y = 0; y < height; y++)
		{
			const size_t index = size_t(x) * height + y;
			const int new_x = x / factor;
			const int new_y = y / factor;
			const int factor_x = (x < width - factor || width % factor == 0) ? factor : width - factor*(width / factor);
			const int factor_y = (y < height - factor || height % factor == 0) ? factor : height - factor*(height / factor);
			const float err = Error(new_x, new_y);
			int64_t new_index = Indexes(x, y);
			if (err < threshold)
			{
				new_index = Indexes(new_x * factor, new_y * factor);
			}			
			if (new_index > max_seen_index)
			{
				max_seen_index = new_index;
				Indexes(x, y) = prev_index;
				prev_index++;
				if (err < threshold)
				{
					elements += size_t(factor_x * factor_y);
				}
				else
				{
					elements++;
				}
			}
			else
			{
				Indexes(x, y) = new_index;
			}
		}
	}

	// new number of supra-pixels (combining original and decimated)
	const size_t puxels = prev_index;
	mexPrintf("Puxels = %d; NNZ=%d \n", puxels, elements);
	output[0] = mxCreateSparse(HW, puxels, elements + 1, mxComplexity::mxREAL);
	double* const vals = mxGetPr(output[0]);
	mwIndex * const row_inds = mxGetIr(output[0]);
	mwIndex * const col_inds = mxGetJc(output[0]);

	size_t col_index = 0;
	size_t s = 0;
	max_seen_index = -1;

	for (size_t i = 0; i < HW; i++)
	{
		const int x = int(i / height);
		const int y = int(i % height);
		const int new_x = x / factor;
		const int new_y = y / factor;
		if (Indexes(x, y) <= max_seen_index)
			continue;
		int64_t puxel = Indexes(x, y);
		max_seen_index = puxel;
		col_inds[puxel] = s;
		if (Error(new_x, new_y) < threshold)
		{
			for (int xx = x; xx < std::min(width, x + factor); xx++)
			{
				for (int yy = y; yy < std::min(height, y + factor); yy++)
				{
					size_t indexx = size_t(xx)*height + yy;
					vals[s] = 1.0;
					row_inds[s] = indexx;
					s++;
				}
			}			
		}
		else
		{
			vals[s] = 1.0;
			row_inds[s] = i;
			s++;
		}
	}
	col_inds[max_seen_index+1] = s;

}