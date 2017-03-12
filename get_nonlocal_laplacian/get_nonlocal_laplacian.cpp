/**
*	@file get_nonlocal_laplacian.cpp
*	@date 18.12.2015
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

#define radius 1
//#define diameter (radius*2 + 1)
//#define window (diameter*diameter)

#define colors 3


template<int largest>
static void compute_adjustency(MexImage<int64_t> &Largest, MexImage<float> &Weights, const MexImage<uint8_t> &Image, const int search, const float sigma)
{
	const int height = Image.height;
	const int width = Image.width;
	const int64_t HW = Image.layer_size;
	const int search_diameter = search * 2 + 1;
	const int search_window = (search * 2 + 1) * (search * 2 + 1);
	
	MexImage<float> Diff(width, height);
	
	//for (int s = 0; s < search_window / 2; s++)
	for (int s = 0; s < search_window; s++)
	{
		Diff.setval(10000.f);
		const int dx = s / search_diameter - search;
		const int dy = s % search_diameter - search;
		if (abs(dx) + abs(dy) <= radius)
		{
			continue;
		}
		
		#pragma omp parallel for
		for (int64_t i = 0; i < HW; i++)
		{
			const int x = i / height;
			const int y = i % height;
			const int xs = std::max(0, std::min(width-1, x + dx));
			const int ys = std::max(0, std::min(height - 1, y + dy));
			
			float diff = 0;
			for (int c = 0; c < colors; c++)
			{
				diff += abs(float(Image(x, y, c)) - float(Image(xs, ys, c)));
			}
			Diff(x, y) = (diff/colors);
		}

		Diff.IntegralImage(true);

		#pragma omp parallel for
		for (int64_t i = 0; i < HW; i++)
		{
			const int x = i / height;
			const int y = i % height;
			
			const int xs = x + dx;
			const int ys = y + dy;
			if (xs < 0 || ys < 0 || xs >= width || ys >= height)
			{
				continue;
			}
			
			int64_t index = i;
			float diff = Diff.getIntegralAverage(x, y, radius);
			//const float diff = Diff(x, y);

			int64_t indexs = xs*height + ys;
			float diffs = diff;

			// sorted insert
			for (int l = 0; l < largest; l++)
			{
				// update current structure
				if (diffs < Weights(x, y, l))
				{
					float diff2 = Weights(x, y, l);
					int64_t index2 = Largest(x, y, l);
					Largest(x, y, l) = indexs;
					Weights(x, y, l) = diffs;
					indexs = index2;
					diffs = diff2;
				}
				//// update at position (xs,ys) 
				//if (diff < Weights(xs, ys, l))
				//{
				//	float diff2 = Weights(xs, ys, l);
				//	int64_t index2 = Largest(xs, ys, l);
				//	Largest(x, y, l) = index;
				//	Weights(x, y, l) = diff;
				//	index = index2;
				//	diff = diff2;
				//}
			}
		}
	}	

	// pre-sort values by index for sparse-matrix structure
	#pragma omp parallel for
	for (long i = 0; i < HW; i++)
	{
		const int x = i / height;
		const int y = i % height;
		
		for (int l = 0; l < largest; l++)
		{
			for (int l2 = l+1; l < largest; l++)
			{
				if (Largest(x, y, l2) < Largest(x, y, l))
				{
					long index = Largest(x, y, l);
					float weight = Weights(x, y, l);
					Largest(x, y, l) = Largest(x, y, l2);
					Weights(x, y, l) = Weights(x, y, l2);
					Largest(x, y, l2) = index;
					Weights(x, y, l2) = weight;
				}
			}
		}
	}
}


void mexFunction(int nout, mxArray* output[], int in, const mxArray* input[])
{
#ifndef _DEBUG
	omp_set_num_threads(std::max<int>(8, omp_get_max_threads()));
	omp_set_dynamic(std::max<int>(4, omp_get_max_threads() / 2));;
#endif	

	if (in < 2 || in > 5 || nout != 1 || mxGetClassID(input[0]) != mxUINT8_CLASS)
	{
		mexErrMsgTxt("USAGE: [NL] = get_nonlocal_laplacian(uint8(I), <search, largest, sigma>);\n");
	}

	MexImage<uint8_t> Image(input[0]);	
	const int search = in > 1 ? std::max<int>(1, int(mxGetScalar(input[1]))) : 3;
	const int largest = in > 2 ? std::max<int>(2, int(mxGetScalar(input[2]))) : 8;
	const float sigma = in > 3 ? std::max<float>(0.0, mxGetScalar(input[3])) : 1.0;

	const int height = Image.height;
	const int width = Image.width;
	const int HW = Image.layer_size;	

	if (largest > 20)
	{
		mexErrMsgTxt("Too many largest!"); 
	}
	size_t tlen = HW * largest;

	MexImage<int64_t> Largest(width, height, largest);
	MexImage<float> Weights(width, height, largest);

	//const mwSize dims[] = { (unsigned)height, (unsigned)width, largest };	
	//output[1] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
	//output[2] = mxCreateNumericArray(3, dims, mxINT64_CLASS, mxREAL);
	//MexImage<float> Weights(output[1]);
	//MexImage<int64_t> Largest(output[2]);
	Largest.setval(-1);	
	Weights.setval(9999.f);

	switch (largest)
	{
	case 1: compute_adjustency<1>(Largest, Weights, Image, search, sigma); break;
	case 2: compute_adjustency<2>(Largest, Weights, Image, search, sigma); break;
	case 3: compute_adjustency<3>(Largest, Weights, Image, search, sigma); break;
	case 4: compute_adjustency<4>(Largest, Weights, Image, search, sigma); break;
	case 5: compute_adjustency<5>(Largest, Weights, Image, search, sigma); break;
	case 6: compute_adjustency<6>(Largest, Weights, Image, search, sigma); break;
	case 7: compute_adjustency<7>(Largest, Weights, Image, search, sigma); break;
	case 8: compute_adjustency<8>(Largest, Weights, Image, search, sigma); break;
	case 9: compute_adjustency<9>(Largest, Weights, Image, search, sigma); break;
	case 10: compute_adjustency<10>(Largest, Weights, Image, search, sigma); break;
	case 11: compute_adjustency<11>(Largest, Weights, Image, search, sigma); break;
	case 12: compute_adjustency<12>(Largest, Weights, Image, search, sigma); break;
	case 13: compute_adjustency<13>(Largest, Weights, Image, search, sigma); break;
	case 14: compute_adjustency<14>(Largest, Weights, Image, search, sigma); break;
	case 15: compute_adjustency<15>(Largest, Weights, Image, search, sigma); break;
	case 16: compute_adjustency<16>(Largest, Weights, Image, search, sigma); break;
	case 17: compute_adjustency<17>(Largest, Weights, Image, search, sigma); break;
	case 18: compute_adjustency<18>(Largest, Weights, Image, search, sigma); break;
	case 19: compute_adjustency<19>(Largest, Weights, Image, search, sigma); break;
	case 20: compute_adjustency<20>(Largest, Weights, Image, search, sigma); break;
	default: mexErrMsgTxt("Too many connections (reduce 'largest') !"); break;
	}

	// number of adjastency elements in "entries" (some of them will be summed up)
	output[0] = mxCreateSparse(HW, HW, tlen+1, mxComplexity::mxREAL);

	double* const vals = mxGetPr(output[0]);
	mwIndex * const row_inds = mxGetIr(output[0]);
	mwIndex * const col_inds = mxGetJc(output[0]);

	// current column index (goes from 0 to HW-1 and then HW)
	//size_t col = 0;

	// current index in sorted values list
	//size_t k = 0;

	// current index in the sparse matrix
	size_t s = 0;
	for (int col = 0; col < HW; col++)
	{
		const int x = col / height;
		const int y = col % height;
		col_inds[col] = col*largest;
		for (int l = 0; l < largest; l++, s++)
		{
			vals[s] = exp(-Weights(x, y, l) / (255 * sigma));
			row_inds[s] = Largest(x, y, l);
		}
	}
	col_inds[HW] = HW*largest;
}
