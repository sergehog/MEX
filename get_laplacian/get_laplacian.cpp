/** Calculate Fast Learning-based Laplacian Matrix
*	@file get_laplacian.cpp
*	@date 22.04.2016
*	@author Sergey Smirnov <sergey.smirnov@tut.fi>
*/

#include "../common/meximage.h"
//#include "sparse_matrix.h"
#include <memory>
//#include <float.h>
#include <cmath>
//#include <atomic>
//#include <vector>
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

using namespace mymex;
using namespace Eigen;


#define diameter (radius*2 + 1)
#define window (diameter*diameter)

template<int radius>
static void compute_adjustency(const MexImage<float> &Image, double * const entries, const int64_t tlen, const double sigma, const double epsilon)
{
	switch (Image.layers)
	{
	case 1: compute_color_adjustency<radius, 1>(Image, entries, tlen, sigma, epsilon); break;
	case 2: compute_color_adjustency<radius, 2>(Image, entries, tlen, sigma, epsilon); break;
	case 3: compute_color_adjustency<radius, 3>(Image, entries, tlen, sigma, epsilon); break;
	case 4: compute_color_adjustency<radius, 4>(Image, entries, tlen, sigma, epsilon); break;
	case 5: compute_color_adjustency<radius, 5>(Image, entries, tlen, sigma, epsilon); break;
	case 6: compute_color_adjustency<radius, 6>(Image, entries, tlen, sigma, epsilon); break;
	case 7: compute_color_adjustency<radius, 7>(Image, entries, tlen, sigma, epsilon); break;
	case 8: compute_color_adjustency<radius, 8>(Image, entries, tlen, sigma, epsilon); break;
	default: mexErrMsgTxt("Too many colors in your image");
	};
}

template<int radius, int colors>
static void compute_color_adjustency(const MexImage<float> &Image, double * const entries, const int64_t tlen, const double sigma, const double epsilon)
{
	const int radius2 = radius * 2;
	const int diameter2 = radius2 * 2 + 1;
	const int window2 = diameter2 * diameter2;

	const int height = Image.height;
	const int width = Image.width;
	const int64_t HW = Image.layer_size;

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
		for (int64_t index = 0; index < HW; index++)
		{
			const int x = static_cast<int>(index / height);
			const int y = static_cast<int>(index % height);
			if (x < radius || x >= width - radius || y < radius || y >= height - radius)
			{
				continue;
			}

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

			if (sigma > 0)
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

						const double value = exp(-sqrt(diff) / sigma);
						F(i1, i2) = value;
						F(i2, i1) = value;
					}
				}
			}
			else
			{
				F = (Xi*Xi.transpose());
			}

			F = F * ((F + I0*epsilon).inverse()); // fenmu = F + I0*lambda;
			F = (I - F).transpose() * (I - F); // normalization			
			
			for (int i = 0; i < window; i++)
			{				
				const int dxi = i / diameter - radius;
				const int dyi = i % diameter - radius;
				const int64_t indexi = (x + dxi)*int64_t(height) + (y + dyi);
				const int xi = x + dxi;
				const int yi = y + dyi;
				//const int xi = indexi / height;
				//const int yi = indexi % height;

				for (int j = 0; j < window; j++)
				{
					const int dxj = j / diameter - radius;
					const int dyj = j % diameter - radius;
					//const long indexj = (x + dxj)*height + (y + dyj);
					const int xj = x + dxj;
					const int yj = y + dyj;
					//const int xj = indexj / height;
					//const int yj = indexj % height;

					// indexing within window2 required to write value in proper place inside "entries" array
					const int dxe = xj - xi;
					const int dye = yj - yi;
					const int indexe = (dxe + radius2) * diameter2 + (dye + radius2);

					entries[indexi*window2 + indexe] += F(i, j);					
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

	if (in < 1 || in > 4 || nout != 1 || mxGetClassID(input[0]) != mxSINGLE_CLASS)
	{
		mexErrMsgTxt("USAGE: [Laplacian] = get_laplacian(single(I), <radius, sigma, epsilon>);\n");
	}

	MexImage<float> Image(input[0]);
	const int radius = in > 1 ? std::max<int>(1, int(mxGetScalar(input[1]))) : 1;
	const double sigma = in > 2 ? std::max<double>(0.0, mxGetScalar(input[2])) : 0;
	const double epsilon = in > 3 ? std::max<double>(0.0, mxGetScalar(input[3])) : 1e-6;
	const int height = Image.height;
	const int width = Image.width;
	const int64_t HW = Image.layer_size;
	
	// number of adjastency elements in "entries" (some of them will be summed up)
	// due to any pixel can be involved in number of different windows (patches), effective window radius is twiced
	// we still calculate regression params separately for each patch 
	const int radius2 = radius * 2;
	const int diameter2 = radius2 * 2 + 1;
	const int window2 = diameter2 * diameter2;	
	int64_t const tlen = HW * window2;

	//std::unique_ptr<double[]> entries(new double[tlen]);
	double * const entries = new double[tlen];
	

	#pragma omp parallel for
	for (int64_t t = 0; t < tlen; t++)
	{
		entries[t] = 0.0;
	}
		
	switch (radius)
	{
	case 2: compute_adjustency<2>(Image, entries, tlen, sigma, epsilon); break;
	case 3: compute_adjustency<3>(Image, entries, tlen, sigma, epsilon); break;
	case 4: compute_adjustency<4>(Image, entries, tlen, sigma, epsilon); break;
	case 5: compute_adjustency<5>(Image, entries, tlen, sigma, epsilon); break;
	default: compute_adjustency<1>(Image, entries, tlen, sigma, epsilon); break;
	}	

	output[0] = mxCreateSparse(HW, HW, tlen, mxComplexity::mxREAL);
	double* const vals = mxGetPr(output[0]);
	mwIndex * const row_inds = mxGetIr(output[0]);
	mwIndex * const col_inds = mxGetJc(output[0]);

	int64_t sparse_index = 0;
	for (int64_t i = 0; i < HW; i++)
	{
		col_inds[i] = sparse_index;
		const int x = static_cast<int>(i / height);
		const int y = static_cast<int>(i % height);

		for (int j = 0; j < window2; j++)
		{
			const int dx = j / diameter2 - radius2;
			const int dy = j % diameter2 - radius2;
			const int xj = x + dx;
			const int yj = y + dy;
			
			if (x < 0 || y < 0 || x >= width || y >= height)
			{
				continue;
			}

			const int64_t indexj = (xj)*int64_t(height) + yj;

			if (entries[i*window2 + j] != 0)
			{
				vals[sparse_index] = entries[i*window2 + j];
				row_inds[sparse_index] = indexj;
				sparse_index++;				
			}
		}
	}
	col_inds[HW] = sparse_index;

	//entries.release();
	delete[] entries;
}
