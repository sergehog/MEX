/**
*	@file get_hierarchical_laplacian.cpp
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
#include <Eigen\Dense>

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

	if (in < 1 || in > 4 || nout < 2 || nout > 3 || mxGetClassID(input[0]) != mxSINGLE_CLASS)
	{
		mexErrMsgTxt("USAGE: [Laplacian, Indexes, {Image}] = get_hierarchical_laplacian(single(I), <factor, threshold, sigma>);\n");
	}

	MexImage<float> Image(input[0]);
	
	const uint8_t factor = in > 1 ? std::max<uint8_t>(1, uint8_t(mxGetScalar(input[1]))) : 2;
	const double threshold = in > 2 ? std::max<double>(1.0, mxGetScalar(input[2])) : 0.5;
	const double sigma = in > 3 ? std::max<double>(0.0, mxGetScalar(input[3])) : 0.1;
	
	const int height = Image.height;
	const int width = Image.width;
	const size_t HW = size_t(width) * height;
	size_t dims[3] = {height, width, 1};
	size_t dims3[3] = { height, width, 3};
	
	output[1] = mxCreateNumericArray(3, dims, mxINT64_CLASS, mxREAL);	
	if (nout > 2)
	{
		output[2] = mxCreateNumericArray(3, dims3, mxSINGLE_CLASS, mxREAL);
	}

	MexImage<float> &Coarsened = *((nout > 2) ? new MexImage<float>(output[2]) : new MexImage<float>(width, height, colors));				
	
	MexImage<int64_t> Indexes(output[1]);
	for (size_t i = 0; i < HW; i++)
	{
		Indexes[i] = i;
		//Hierarchy[i] = 1;
		for (int c = 0; c < colors; c++)
		{
			Coarsened[i + HW*c] = Image[i + HW*c];
		}
	}
	
	const int new_width = width % 2 ? width / 2 + 1 : width / 2;
	const int new_height = height % 2 ? height / 2 + 1 : height / 2;	
	
	// decimate image to a coarser level
	//size_t dims3[3] = { new_height, new_width, colors };
	//output[0] = mxCreateNumericArray(3, dims3, mxSINGLE_CLASS, mxREAL);
	//MexImage<float> NewImage(output[0]);

	MexImage<float> NewImage(new_width, new_height, colors);
	NewImage.setval(0.f);

	for (int x = 0; x < width; x++)
	{
		for (int y = 0; y < height; y++)
		{
			const int new_x = x / 2;
			const int new_y = y / 2;
			const float factor_x = width % 2 && (x == width - 1) ? 1.f : 2.f;
			const float factor_y = height % 2 && (y == height - 1) ? 1.f : 2.f;
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
			const int new_x = x / 2;
			const int new_y = y / 2;
			const float factor_x = width % 2 && (x == width - 1) ? 1.f : 2.f;
			const float factor_y = height % 2 && (y == height - 1) ? 1.f : 2.f;
			float err = 0;
			for (int c = 0; c < colors; c++)
			{
				float diff = Image(x, y, c) - NewImage(new_x, new_y, c);				
				err += diff *diff;				
			}			
			Error(new_x, new_y) += sqrt(err) / (factor_x*factor_y);
		}
	}
		

	// required too see if somne pixel already shown up
	long prev_index = 0;
	long max_seen_index = -1;

	// threshold and re-calculate indexes
	for (int x = 0; x < width; x++)
	{
		for (int y = 0; y < height; y++)
		{
			const size_t index = size_t(x) * height + y;
			const int new_x = x / 2;
			const int new_y = y / 2;
			const float err = Error(new_x, new_y);			
			int64_t new_index = Indexes(x, y);
			if (err < threshold)
			{
				//Indexes(x, y) = Indexes(new_x * 2, new_y * 2);
				new_index = Indexes(new_x * 2, new_y * 2);
				for (int c = 0; c < colors; c++)
				{
					Coarsened(x, y, c) = NewImage(new_x, new_y, c);
				}				
			}			
			if (new_index > max_seen_index)
			{
				max_seen_index = new_index;
				Indexes(x, y) = prev_index;
				prev_index++;				
			}
			else
			{
				Indexes(x, y) = new_index;
			}
		}
	}
	
	// new number of supra-pixels (combining original and decimated)
	const size_t puxels = prev_index;
	mexPrintf("Puxels = %d\n", puxels);
	
// each neighbourhood can contain at max 12  plus one current
#define maxneihgb 13

	// back mapping from puxels to pixels of the fine grid
	std::unique_ptr<size_t[]> back_mapping = std::unique_ptr<size_t[]>(new size_t[puxels]);

	// each puxel neighbours
	std::unique_ptr<int64_t[]> neighbourhoods = std::unique_ptr<int64_t[]>(new int64_t[puxels*maxneihgb]);
	//int64_t *neighbourhoods = new int64_t[puxels*maxneihgb];
	
	for (size_t i = 0; i < puxels*maxneihgb; i++)
	{
		neighbourhoods[i] = -1;
	}

	std::unique_ptr<uint8_t[]> neigh_sizes = std::unique_ptr<uint8_t[]>(new uint8_t[puxels]);

	//std::unique_ptr<float[]> neighb_colors = std::unique_ptr<float[]>(new float[puxels*maxneihgb*colors]);
	max_seen_index = -1;
	size_t overall_length = 0;
	for (size_t i = 0; i < HW; i++)
	{
		const int x = int(i / height);
		const int y = int(i % height);
		const int new_x = x / 2;
		const int new_y = y / 2;
		if (Indexes(x, y) <= max_seen_index)
			continue;
		
		const int64_t puxel = Indexes(x, y);
		max_seen_index = puxel;
		back_mapping[puxel] = i;
		
		neighbourhoods[puxel*maxneihgb] = puxel;
		// is puxel big (2x2) or small (1x1)
		const int r2 = (Error(new_x, new_y) < threshold) ? 2 : 1;
		int neigh_size = 1;
		// go over neighbourhood
		for (int xn = std::max(0, x - 1); xn <= std::min(width - 1, x + r2); xn++)
		{
			for (int yn = std::max(0, y - 1); yn <= std::min(height - 1, y + r2); yn++)
			{
				int64_t neigh_puxel = Indexes(xn, yn);
				// sorted insert
				for (int n = 0; n < maxneihgb; n++)
				{
					neigh_size = std::max(neigh_size, n + 1);
					if (neighbourhoods[puxel*maxneihgb + n] == -1 || neighbourhoods[puxel*maxneihgb + n] == neigh_puxel)
					{
						neighbourhoods[puxel*maxneihgb + n] = neigh_puxel;
						break;
					}
					else if (neighbourhoods[puxel*maxneihgb + n] > neigh_puxel)
					{
						int64_t temp = neighbourhoods[puxel*maxneihgb + n];
						neighbourhoods[puxel*maxneihgb + n] = neigh_puxel;
						neigh_puxel = temp;
					}							
				}

			}			
		}
		//mexPrintf("%d, ", neigh_size);
		//if (puxel % 100 == 0)
		//{
		//	mexPrintf("\n", neigh_size);
		//}
		neigh_sizes[puxel] = neigh_size;
		overall_length += neigh_size;
	}
	//delete[] neighbourhoods;

	// number of adjastency elements in "entries" (some of them will be summed up)
	output[0] = mxCreateSparse(puxels, puxels, overall_length + 1, mxComplexity::mxREAL);

	double* const vals = mxGetPr(output[0]);
	mwIndex * const row_inds = mxGetIr(output[0]);
	mwIndex * const col_inds = mxGetJc(output[0]);


	size_t col_index = 0;
	size_t s = 0;
	for (int64_t puxel = 0; puxel < puxels; puxel++)
	{
		const int8_t heigh_size = neigh_sizes[puxel];	
		col_inds[puxel] = col_index;
		col_index += heigh_size;
		size_t index = back_mapping[puxel];
		for (int8_t n = 0; n < heigh_size; n++, s++)
		{
			const int64_t neib_puxel = neighbourhoods[puxel*maxneihgb + n];
			const  size_t neib_index = back_mapping[neib_puxel];
			float err = 0;
			for (int c = 0; c < colors; c++)
			{
				float diff = (Coarsened[index + c*HW] - Coarsened[neib_index + c*HW]);
				err += diff*diff;
			}
			
			vals[s] = exp(-sqrt(err) / sigma);
			row_inds[s] = neib_puxel;
		}
	}
	col_inds[puxels] = col_index;

	//Eigen::Matrix<double, maxneihgb, maxneihgb> I0 = Eigen::Matrix<double, maxneihgb, maxneihgb>::Identity();
	//Eigen::Matrix<double, maxneihgb, maxneihgb> I = Eigen::Matrix<double, maxneihgb, maxneihgb>::Identity();
	//I0(maxneihgb - 1, maxneihgb - 1) = 0.0;
	
}