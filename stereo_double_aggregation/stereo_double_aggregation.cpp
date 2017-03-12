/** stereo_double_aggregation
* @author Sergey Smirnov
* @date 2015-09-11
*/
#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}
#endif // __cplusplus
//#include "../common/common.h"
#include "../common/meximage2.h"
#include <algorithm>
#include <memory>
#ifndef _NDEBUG
#include <omp.h>
#endif

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")
#pragma warning(disable:4244)
#pragma warning(disable:4018)

using namespace mymex;

float getWeight2(const MexImage<float> &Color1, const MexImage<float> &Color2, const int x1, const int y1, const int x2, const int y2, const int disp, const float * const weights_lookup)
{
	const int width = Color1.width;
	float diff1 = 0;
	if (!isnan(Color1(x1, y1)) && !isnan(Color1(x2, y2)))
	{
		for (int c = 0; c < Color1.layers; c++)
		{
			diff1 += (abs(Color1(x1, y1, c) - Color1(x2, y2, c)));
		}
	}
	else
	{
		diff1 = FLT_MAX;
	}


	float diff2 = 0;
	const int x1r = x1 + disp < 0 ? 0 : x1 + disp >= width ? width - 1 : x1 + disp;
	const int x2r = x2 + disp < 0 ? 0 : x2 + disp >= width ? width - 1 : x2 + disp;
	if (!isnan(Color2(x1r, y1)) && !isnan(Color2(x2r, y2)))
	{
		for (int c = 0; c < Color1.layers; c++)
		{
			diff2 += (abs(Color2(x1, y1, c) - Color2(x2, y2, c)));
		}
	}
	else
	{
		diff2 = FLT_MAX;
	}


	return diff1 < diff2 ? weights_lookup[unsigned(mymex::round(diff1))] : weights_lookup[unsigned(mymex::round(diff2))];

}

void recursive_bilateral(MexImage<float> &Signal, MexImage<float> &Temporal, MexImage<float> &Weights, MexImage<float> &Weights2, const MexImage<float> &Color1, const MexImage<float> &Color2, const int disp, float const * const weights_lookup, float const sigma_spatial)
{
	const int width = Signal.width;
	const int height = Signal.height;
	const long HW = width*height;

	//const float alpha = exp(-sqrt(2.0) / (sigma_spatial*std::min(height, width)));
	const float alpha = sigma_spatial;

	Weights.setval(0.f);
	Temporal.setval(0.f);


	// horisontal passes
#pragma omp parallel for
	for (int y = 0; y<height; y++)
	{
		float t1 = Signal(0, y);
		float t2 = Signal(width - 1, y);
		float w1 = 1.f;
		float w2 = 1.f;

		Temporal(0, y) = t1;
		Temporal(width - 1, y) = t2;
		Weights(0, y) = w1;
		Weights(width - 1, y) = w2;

		for (int x1 = 1; x1<width; x1++)
		{
#pragma omp parallel sections
			{
#pragma omp section
				{
					const float weight1 = getWeight2(Color1, Color2, x1, y, x1 - 1, y, disp, weights_lookup);
					t1 = Signal(x1, y) + t1 * alpha*weight1;
					w1 = (1 + w1 * alpha*weight1);
#pragma omp atomic
					Temporal(x1, y) += t1;
#pragma omp atomic
					Weights(x1, y) += w1;
				}
#pragma omp section
				{
					const int x2 = width - x1 - 1;
					const float weight2 = getWeight2(Color1, Color2, x2, y, x2 + 1, y, disp, weights_lookup);
					t2 = Signal(x2, y) + t2 * alpha*weight2;
					w2 = (1 + w2 * alpha*weight2);
#pragma omp atomic
					Temporal(x2, y) += t2;
#pragma omp atomic
					Weights(x2, y) += w2;
				}
			}
		}

#pragma omp parallel for
		for (int x = 0; x<width; x++)
		{
			Weights(x, y) -= 1;
			Temporal(x, y) -= Signal(x, y);
		}
	}

	Signal.setval(0.f);
	Weights2.setval(0.f);

	//vertical passes		
#pragma omp parallel for
	for (int x = 0; x<width; x++)
	{
		float t1 = Temporal(x, 0);
		float t2 = Temporal(x, height - 1);
		float w1 = Weights(x, 0);
		float w2 = Weights(x, height - 1);

		Signal(x, 0) = t1;
		Signal(x, height - 1) = t2;
		Weights2(x, 0) = w1;
		Weights2(x, height - 1) = w2;

		for (int y1 = 1; y1<height; y1++)
		{
#pragma omp parallel sections
			{
#pragma omp section
				{
					const float weight1 = getWeight2(Color1, Color2, x, y1, x, y1 - 1, disp, weights_lookup);
					t1 = Temporal(x, y1) + t1 * alpha*weight1;
					w1 = (Weights(x, y1) + w1 * alpha*weight1);
#pragma omp atomic
					Signal(x, y1) += t1;
#pragma omp atomic
					Weights2(x, y1) += w1;
				}
#pragma omp section
				{
					const int y2 = height - y1 - 1;
					const float weight2 = getWeight2(Color1, Color2, x, y2, x, y2 + 1, disp, weights_lookup);
					t2 = Temporal(x, y2) + t2 * alpha*weight2;
					w2 = (Weights(x, y2) + w2 * alpha*weight2);
#pragma omp atomic
					Signal(x, y2) += t2;
#pragma omp atomic
					Weights2(x, y2) += w2;
				}
			}
		}

#pragma omp parallel for
		for (int y = 0; y<height; y++)
		{
			Weights2(x, y) -= Weights(x, y);
			Signal(x, y) -= Temporal(x, y);
		}
	}

	// final normalization
#pragma omp parallel for
	for (long i = 0; i<HW; i++)
	{
		const int x = i / height;
		const int y = i % height;
		Signal(x,y) /= Weights2(x,y);
	}
}


void mexFunction(int nout, mxArray* output[], int in, const mxArray* input[])
{
#ifndef _NDEBUG
	omp_set_num_threads(std::max<int>(4, omp_get_max_threads()));
	omp_set_dynamic(std::max<int>(2, omp_get_max_threads() / 2));
#endif

	if (in < 4 || in > 7 || nout != 2 || mxGetClassID(input[0]) != mxSINGLE_CLASS || mxGetClassID(input[1]) != mxSINGLE_CLASS)
	{
		mexErrMsgTxt("USAGE: [DispL, ConfL] = compute_cost(Left, Right, mindisp, maxdisp <,cost_threshold, sigma_color, sigma_spatial>);");
	}
	MexImage<float> Left(input[0]);
	MexImage<float> Right(input[1]);
	const int mindisp = static_cast<int>(mxGetScalar(input[2]));
	const int maxdisp = static_cast<int>(mxGetScalar(input[3]));
	const float cost_thr = (in > 4) ? static_cast<float>(mxGetScalar(input[4])) : 1000.f;
	const float sigma_color = (in > 5) ? static_cast<float>(mxGetScalar(input[5])) : 0.5f;
	const float sigma_spatial = (in > 6) ? static_cast<float>(mxGetScalar(input[6])) : 0.5f;
	
	if (maxdisp <= mindisp)
	{
		mexErrMsgTxt("ERROR: 'maxdisp' must be larger than 'mindisp'!");
	}

	const int width = mxGetWidth(input[0]);
	const int height = mxGetHeight(input[0]);
	const int colors = mxGetLayers(input[0]);
	const long HW = height * width;

	if (height != mxGetHeight(input[1]) || width != mxGetWidth(input[1]) || colors != mxGetLayers(input[1]))
	{
		mexErrMsgTxt("ERROR: Sizes of parameters 'left', 'right' must be the same.");
	}

	if (colors < 1 || colors > 9)
	{
		mexErrMsgTxt("Too many colors in your images.");
	}

	const int layers = std::max(maxdisp - mindisp + 1, 1);
	const mwSize size1[] = { (size_t)height, (size_t)width, 1 };

	std::unique_ptr<float[]> weights_lookup(new float[256 * colors]);
	#pragma omp parallel for
	for (int i = 0; i < 256 * colors; i++)
	{
		weights_lookup[i] = exp(-float(i) / (colors * 255 * sigma_color));
	}

	//Matlab-allocated variables
	output[0] = mxCreateNumericArray(3, size1, mxSINGLE_CLASS, mxREAL);	
	output[1] = mxCreateNumericArray(3, size1, mxSINGLE_CLASS, mxREAL);
	MexImage<float> DispL(output[0]);	
	MexImage<float> ConfL(output[1]);
	MexImage<float> BestCost(width, height);
	MexImage<float> SecondBestCost(width, height);
	BestCost.setval(FLT_MAX);	
	SecondBestCost.setval(FLT_MAX);

	#pragma omp parallel
	{
		MexImage<float> Cost(width, height);
		MexImage<float> Temporal(width, height);
		MexImage<float> Weights(width, height);
		MexImage<float> Weights2(width, height);

		#pragma omp for
		for (int l = 0; l < layers; l++)
		{
			Cost.setval(0.f);
			const int disp = l + mindisp;

			#pragma omp parallel for
			for (long i = 0; i < HW; i++)
			{
				const int x = i / height;
				const int y = i % height;
				const int xr = x - disp < 0 ? 0 : (x - disp >= width ? width - 1 : x - disp);
				for (int c = 0; c < colors; c++)
				{
					Cost(x, y) += abs(Left(x, y, c) - Right(xr, y, c));
				}
				Cost(x, y) /= colors;
				Cost(x, y) = Cost(x, y) > cost_thr ? cost_thr : Cost(x, y);
			}

			recursive_bilateral(Cost, Temporal, Weights, Weights2, Left, Right, disp, weights_lookup.get(), sigma_spatial);
			
			#pragma omp parallel for
			for (long i = 0; i < HW; i++)
			{
				const int x = i / height;
				const int y = i % height;
				
				if (Cost(x,y) < BestCost(x,y))
				{
					SecondBestCost(x, y) = BestCost(x, y);
					BestCost(x, y) = Cost(x, y);
					DispL(x, y) = disp;
				}
				else if (Cost(x, y) < SecondBestCost(x, y))
				{
					SecondBestCost(x, y) = Cost(x, y);
				}
			}

		}
		
	}

	#pragma omp parallel for
	for (long i = 0; i < HW; i++)
	{
		const int x = i / height;
		const int y = i % height;
		ConfL(x, y) = (SecondBestCost(x, y) - BestCost(x, y)) / SecondBestCost(x, y);
	}

}