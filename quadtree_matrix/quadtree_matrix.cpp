/**
*	@file quadtree_matrix.cpp
*	@date 11.04.2016
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


void mexFunction(int nout, mxArray* output[], int in, const mxArray* input[])
{
#ifndef _DEBUG
	omp_set_num_threads(std::max<int>(8, omp_get_max_threads()));
	omp_set_dynamic(std::max<int>(4, omp_get_max_threads() / 2));;
#endif	

	if (in < 1 || in > 3 || nout < 1 || nout > 2 || mxGetClassID(input[0]) != mxSINGLE_CLASS)
	{
		mexErrMsgTxt("USAGE: [ReductionMatrix, <DecomposedImage>] = quadtree_matrix(single(Image), <levels, threshold>);\n");
	}

	MexImage<float> Image(input[0]);

	const uint8_t levels = in > 1 ? std::max<uint8_t>(2, uint8_t(mxGetScalar(input[1]))) : 2;
	const float threshold = in > 2 ? std::max<float>(0.f, mxGetScalar(input[2])) : 1.f;

	const int height = Image.height;
	const int width = Image.width;
	const int colors = Image.layers;
	const size_t HW = size_t(width) * height;

	
	const size_t dims[] = { (size_t)height, (size_t)width, (size_t)1 };
	
	output[0] = mxCreateNumericArray(3, dims, mxUINT64_CLASS, mxREAL);
	MexImage<uint64_t> Indexes(output[0]);
	for (size_t i = 0; i < HW; i++)
	{
		Indexes[i] = i;
	}
	MexImage<float> *_Decomposed;
	if (nout > 1)
	{
		const size_t dimC[] = { (size_t)height, (size_t)width, (size_t)colors};
		output[1] = mxCreateNumericArray(3, dimC, mxSINGLE_CLASS, mxREAL);
		_Decomposed = new MexImage<float>(output[1]);
	}
	else
	{
		_Decomposed = new MexImage<float>(width, height, colors);
	}
	MexImage<float> &Decomposed = *_Decomposed;
	for (long i = 0; i < HW; i++)
	{
		for (int c = 0; c < colors; c++)
		{
			Decomposed[i + c*HW] = Image[i + c*HW];
		}		
	}	


	std::unique_ptr<MexImage<float> *[]> Images = std::unique_ptr<MexImage<float> *[]>(new MexImage<float> *[levels]);
	Images[0] = &Image;
	std::unique_ptr<MexImage<float> *[]> Errors = std::unique_ptr<MexImage<float> *[]>(new MexImage<float> *[levels]);	

	// construct image pyramid and errors
	for (uint8_t l = 1; l < levels; l++)
	{
		MexImage<float> &Prev = *Images[l - 1];
		const int prev_width = Prev.width;
		const int prev_height = Prev.height;
		const int new_width = prev_width % 2 ? prev_width / 2 + 1 : prev_width / 2;
		const int new_height = prev_height % 2 ? prev_height / 2 + 1 : prev_height / 2;

		Images[l] = new MexImage<float>(new_width, new_height, colors);
		MexImage<float> &New = *Images[l];		
		New.setval(0.f);

		// decimate image
		for (int x = 0; x < prev_width; x++)
		{
			for (int y = 0; y < prev_height; y++)
			{
				const int new_x = x / 2;
				const int new_y = y / 2;
				const int factor_x = prev_width % 2 && (x == prev_width - 1) ? 1 : 2;
				const int factor_y = prev_height % 2 && (y == prev_height - 1) ? 1 : 2;
				for (int c = 0; c < colors; c++)
				{
					New(new_x, new_y, c) += Prev(x, y, c) / (factor_x * factor_y);
				}
			}
		}

		Errors[l] = new MexImage<float>(new_width, new_height);
		MexImage<float> &Error = *Errors[l];
		Error.setval(0.f);
		
		// calculate error due to decimation
		for (int x = 0; x < prev_width; x++)
		{
			for (int y = 0; y < prev_height; y++)
			{
				const int new_x = x / 2;
				const int new_y = y / 2;
				const int factor_x = prev_width % 2 && (x == prev_width - 1) ? 1 : 2;
				const int factor_y = prev_height % 2 && (y == prev_height - 1) ? 1 : 2;
				
				float err = 0;
				if (l == 1)
				{
					for (int c = 0; c < colors; c++)
					{
						float diff = Prev(x, y, c) - New(new_x, new_y, c);
						err += diff *diff;
					}
					err = sqrt(err);// / (factor_x*factor_y);
				}
				else
				{
					MexImage<float> &PrevError = *Errors[l-1];
					err = PrevError(x, y);
				}
				Error(new_x, new_y) += err ;
			}
		}
	}
	
	//Indexes.setval(0);
	uint64_t max_seen_index = 0;
	uint64_t curr_index = 0;
	for (int x = 0; x < width; x++)
	{
		for (int y = 0; y < height; y++)
		{			
			int new_x = x;
			int new_y = y;
			
			// find best level for the pixel
			uint8_t best_level = 0;
			for (uint8_t level = 1; level < levels; level++)
			{		
				MexImage<float> &Error = *Errors[level];
				MexImage<float> &Img = *Images[level];
				new_x = new_x / 2;
				new_y = new_y / 2;				
				if (Error(new_x, new_y) < threshold)
				{
					best_level = level;
										
					for (uint8_t c = 0; c < colors; c++)
					{
						Decomposed(x, y, c) = Img(new_x, new_y, c);
					}					
				}
				else
				{
					break;
				}
			}

			new_x = x;
			new_y = y;
			
			// find corresponding index in the finer image
			for (uint8_t l = 1; l <= best_level; l++)
			{
				new_x = new_x / 2;
				new_y = new_y / 2;					
			}
			for (uint8_t l = best_level; l >= 1; l--)
			{
				new_x = new_x * 2;
				new_y = new_y * 2;
			}
			uint64_t index = new_x * height + new_y;
			
			if (index > max_seen_index)
			{
				max_seen_index = index;
				curr_index = curr_index ++;
				Indexes(x, y) = curr_index;
			}
			else
			{
				Indexes(x, y) = Indexes(new_x, new_y);
			}			
		}
	}
	
//	mexPrintf("HW=%d, PUX=%d, RATIO=%f \n", HW, curr_index, float(HW) / curr_index);
	//for (uint8_t l = levels-1; l >= 1; l--)
	//{
	//	MexImage<float> &Prev = *Images[l - 1];
	//	MexImage<uint8_t> &PrevLevel = *Levels[l-1];

	//	const int prev_width = Prev.width;
	//	const int prev_height = Prev.height;
	//	
	//	MexImage<float> &NewError = *Errors[l];
	//	//MexImage<float> &NewImage = *Images[l];
	//	MexImage<uint8_t> &NewLevel = *Levels[l];

	//	for (int x = 0; x < prev_width; x++)
	//	{
	//		for (int y = 0; y < prev_height; y++)
	//		{
	//			const int new_x = x / 2;
	//			const int new_y = y / 2;
	//			const float err = NewError(new_x, new_y);
	//			
	//			PrevLevel(x, y) = (err < threshold) ? NewLevel(new_x, new_y) + 1 : 0;
	//		}
	//	}		
	//}

	//for (uint8_t l = 1; l < levels; l++)
	//{
	//	MexImage<float> &Prev = *Images[l - 1];
	//	MexImage<uint8_t> &PrevLevel = *Levels[l - 1];
	//	MexImage<float> &NewError = *Errors[l];
	//	NewError.setval(0.f);

	//	const int prev_width = Prev.width;
	//	const int prev_height = Prev.height;
	//	
	//	for (int x = 0; x < prev_width; x++)
	//	{
	//		for (int y = 0; y < prev_height; y++)
	//		{
	//			const int new_x = x / 2;
	//			const int new_y = y / 2;
	//			

	//		}
	//	}
	//}

	for (uint8_t l = 1; l < levels; l++)
	{
		MexImage<float> &NewError = *Errors[l];
		MexImage<float> &NewImage = *Images[l];
		//MexImage<uint8_t> &NewLevel = *Levels[l];

		//delete &NewLevel;
		delete &NewError;
		delete &NewImage;
	}

	/*
	MexImage<float> NewImage(new_width, new_height, colors);
	NewImage.setval(0.f);

	for (int x = 0; x < width; x++)
	{
		for (int y = 0; y < height; y++)
		{
			const int new_x = x / factor;
			const int new_y = y / factor;
			const int factor_x = (x < width - factor || width % factor == 0) ? factor : width - factor*(width / factor);
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
	col_inds[max_seen_index + 1] = s;
	*/
}