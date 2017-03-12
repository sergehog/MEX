/**
*	@file get_learning_laplacian.cpp
*	@date 29.09.2015
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
using namespace Eigen;

//#define radius 1
#define diameter (radius*2 + 1)
#define window (diameter*diameter)

template<int radius>
static void compute_adjustency(sparseEntry * const entries, MexImage<uint8_t> &Image, MexImage<bool> &ProcessArea, const size_t elements_num, const double lambda, const double trick_sigma)
{
	switch (Image.layers)
	{
	case 1: compute_color_adjustency<radius, 1>(entries, Image, ProcessArea, elements_num, lambda, trick_sigma); break;
	case 2: compute_color_adjustency<radius, 2>(entries, Image, ProcessArea, elements_num, lambda, trick_sigma); break;
	case 3: compute_color_adjustency<radius, 3>(entries, Image, ProcessArea, elements_num, lambda, trick_sigma); break;
	case 4: compute_color_adjustency<radius, 4>(entries, Image, ProcessArea, elements_num, lambda, trick_sigma); break;
	default: mexErrMsgTxt("Too many colors in your image");
	};	
}


template<int radius, int colors>
static void compute_color_adjustency(sparseEntry * const triplets, const MexImage<uint8_t> &Image, const MexImage<bool> &ProcessArea, const size_t elements_num, const double lambda, const double trick_sigma)
{
	if (isnan(trick_sigma) || trick_sigma == 0.0)
	{
		compute_alpha_trick<radius, colors, 0>(triplets, Image, ProcessArea, elements_num, lambda, trick_sigma);
	}
	else
	{
		compute_alpha_trick<radius, colors, 1>(triplets, Image, ProcessArea, elements_num, lambda, trick_sigma);
	}
}

template<int radius, int colors, bool trick>
void compute_alpha_trick(sparseEntry * const triplets, const MexImage<uint8_t> &Image, const MexImage<bool> &ProcessArea, const size_t elements_num, const double lambda, const double trick_sigma)
{
	const int height = Image.height;
	const int width = Image.width;
	const int HW = Image.layer_size;

	// in order to parallelize further processing, let's write all indexes in a separate array
	std::vector<long> elements(elements_num);
	for (long i = 0, j = 0; i < HW; i++)
	{
		const int x = i / height;
		const int y = i % height;

		if (!ProcessArea[i])
		{
			continue;
		}
		elements[j] = i;
		j++;
	}
	Eigen::Matrix<double, window, window> I0 = Eigen::Matrix<double, window, window>::Identity();
	Eigen::Matrix<double, window, window> I = Eigen::Matrix<double, window, window>::Identity();
	I0(window - 1, window - 1) = 0.0;

#pragma omp parallel 
	{
		Eigen::Matrix<double, window, window> F;
		Matrix<double, window, colors + 1> Xi;
		for (int i = 0; i < window; i++)
		{
			Xi(i, colors) = 1.0;
		}
		#pragma omp for //schedule(dynamic)
		for (long je = 0; je < elements_num; je++)
		{
			const long index = elements[je];
			const int x = index / height;
			const int y = index % height;

			// data matrix
			#pragma omp parallel for
			for (int i = 0; i < window; i++)
			{
				const int dx = i / diameter - radius;
				const int dy = i % diameter - radius;
				const int xi = x + dx;
				const int yi = y + dy;
				for (int c = 0; c < colors; c++)
				{
					Xi(i, c) = double(Image(xi, yi, c)) / 255;
				}
			}

			// kernel-trick weights
			if (trick) 
			{
				for (int i1 = 0; i1 < window; i1++)
				{
					F(i1, i1) = 1.0;
					for (int i2 = i1 + 1; i2 < window; i2++)
					{
						double diff = 0.0;
						for (int c = 0; c < colors; c++)
						{
							diff += (Xi(i1, c) - Xi(i2, c)) * (Xi(i1, c) - Xi(i2, c));
						}

						const double value = exp(-diff / trick_sigma);
						F(i1, i2) = value;
						F(i2, i1) = value;
					}
				}
			}			
			// normal (linear) weights
			else
			{
				F = (Xi*Xi.transpose());				
			}

			F = F * ((F + I0*lambda).inverse()); // fenmu = F + I0*lambda;
			F = (I - F).transpose() * (I - F); // normalization
			//F = (I - F); // normalization


			long k = je*window*window;
			for (int i = 0; i < window; i++)
			{
				const int dxi = i / diameter - radius;
				const int dyi = i % diameter - radius;
				const long indexi = (x + dxi)*height + (y + dyi);

				for (int j = 0; j < window; j++, k++)
				{
					const int dxj = j / diameter - radius;
					const int dyj = j % diameter - radius;
					const long indexj = (x + dxj)*height + (y + dyj);
					triplets[k].value = F(i, j);
					triplets[k].row = indexj;
					triplets[k].col = indexi;					
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

	if (in < 2 || in > 5 || nout != 1 || mxGetClassID(input[0]) != mxUINT8_CLASS || mxGetClassID(input[1]) != mxUINT8_CLASS)
	{
		mexErrMsgTxt("USAGE: [L] = get_learning_laplacian(uint8(I), uint8(Trimap), <radius, kernel_trick_sigma, lambda>);\n");
	}

	MexImage<uint8_t> Image(input[0]);
	MexImage<uint8_t> Trimap(input[1]);
	const int radius = in > 2 ? std::max<int>(1, int(mxGetScalar(input[2]))) : 2;
	const double trick_sigma = in > 3 ? std::max<double>(0.0, mxGetScalar(input[3])) : 0.0;	
	const double lambda = in > 4 ? std::max<double>(0.0, mxGetScalar(input[4])) : 0.0000001;
	//const double c = in > 5 ? std::max<double>(0.000001, mxGetScalar(input[5])) : 800.0;
	
	const int height = Image.height;
	const int width = Image.width;
	const int HW = Image.layer_size;
	//const int colors = Image.layers;	

	MexImage<bool> ProcessArea(width, height);
	ProcessArea.setval(0);

#pragma omp parallel for schedule(dynamic)
	for (long i = 0; i < HW; i++)
	{
		const int x = i / height;
		const int y = i % height;
		if (Trimap(x, y) != 0 && Trimap(x, y) != 2 && Trimap(x, y) != 255)
		{
			for (int xi = std::max<int>(radius, x - radius); xi <= std::min<int>(width - radius - 1, x + radius); xi++)
			{
				for (int yi = std::max<int>(radius, y - radius); yi <= std::min<int>(height - radius - 1, y + radius); yi++)
				{
					ProcessArea(xi, yi) = true;
				}
			}
		}
	}
	
	// number of pixels, used for adjastency calculation
	size_t elements_num = 0;
	#pragma omp parallel for reduction(+: elements_num)
	for (long i = 0; i < HW; i++)
	{
		elements_num += ProcessArea[i];
	}

	
	// number of adjastency elements in "entries" (some of them will be summed up)
	size_t const tlen = elements_num*window*window;
	std::unique_ptr<sparseEntry[]> entries(new sparseEntry[tlen]);// tlen);
	switch (radius)
	{
	case 1: compute_adjustency<1>(entries.get(), Image, ProcessArea, elements_num, lambda, trick_sigma); break;
	case 2: compute_adjustency<2>(entries.get(), Image, ProcessArea, elements_num, lambda, trick_sigma); break;
	case 3: compute_adjustency<3>(entries.get(), Image, ProcessArea, elements_num, lambda, trick_sigma); break;
	default: compute_adjustency<4>(entries.get(), Image, ProcessArea, elements_num, lambda, trick_sigma); break;
	}

	//const size_t allocate = size_t(tlen / (radius * 2 + 1));
	const size_t allocate = size_t(tlen)/2;
	output[0] = mxCreateSparse(HW, HW, allocate, mxComplexity::mxREAL);

	double* const vals = mxGetPr(output[0]);
	mwIndex * const row_inds = mxGetIr(output[0]);
	mwIndex * const col_inds = mxGetJc(output[0]);

	fillSparseMatrix(entries.get(), tlen, vals, row_inds, col_inds, HW);

	entries.reset();
	mexPrintf("Number of non-empty entries: %d (allocated %d) \n", col_inds[HW], allocate);
}
