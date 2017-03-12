/** recursive_bilateral
*	@file recursive_bilateral.cpp
*	@date 26.01.2014
*	@author Sergey Smirnov <sergey.smirnov@tut.fi>
*/

#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}
#endif // __cplusplus

#include "../common/meximage.h"
//#include "../common/defines.h"
#include <vector>
#define _USE_MATH_DEFINES
#include <cmath>
#include <algorithm>
#include <memory>
#ifndef _DEBUG
#include <omp.h>
#endif

//#define M_PI       3.14159265358979323846
#define WEIGHTS_ORIGINAL

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")

typedef unsigned char uint8;
using namespace mymex;

float getWeight(const MexImage<float> &Image, const int x1, const int y1, const int x2, const int y2, const float * const weights_lookup);

template<typename T>
void recursive_bilateral_new(const MexImage<T> &Signal, const MexImage<float> &Filtered, const MexImage<float> &Image, const MexImage<float> &Weights2, const float sigma_color, const float sigma_spatial = 1.0)
{
	const int width = Signal.width;
	const int height = Signal.height;
	const int layers = Signal.layers;
	const int colors = Image.layers;

	MexImage<float> Temporal(width, height, layers);
	MexImage<float> Weights(width, height);
	//MexImage<float> Weights2(width, height);

	Weights.setval(0.f);
	Weights2.setval(0.f);
	Temporal.setval(0.f);	
	Filtered.setval(0.f);	
	//WeightsTemporal.setval(0.f);

	std::unique_ptr<float[]> weights_lookup(new float[256 * colors]);	

	for (int i = 0; i < 256 * colors; i++)
	{
		weights_lookup[i] = float(exp(-double(i) / (255 * colors * sigma_color)));
	}

	const long HW = Signal.layer_size;

	// horisontal passes
	#pragma omp parallel
	{		
		std::unique_ptr<float[]> temporal1(new float[Signal.layers]);
		std::unique_ptr<float[]> temporal2(new float[Signal.layers]);
		
		float temporal_weight1, temporal_weight2;
#ifdef UPDATED
		std::unique_ptr<float[]> temporal_color1(new float[colors]);
		std::unique_ptr<float[]> temporal_color2(new float[colors]);
#endif
		#pragma omp for
		for(int y=0; y<height; y++)
		{
			for(int l=0; l<layers; l++)
			{
				temporal1[l] = float(Signal(0,y,l));
				temporal2[l] = float(Signal(width - 1, y, l));
				Temporal(0,y,l) = temporal1[l];
				Temporal(width-1,y,l) = temporal2[l];
			}
			temporal_weight1 = 1;
			temporal_weight2 = 1;
			Weights(0, y) = temporal_weight1;
			Weights(width-1, y) = temporal_weight2;
#ifdef UPDATED
			for(int c=0; c<colors; c++)
			{
				temporal_color1[c] = Image(0,y,c);
				temporal_color2[c] = Image(width - 1, y, c);
			}
#endif

			for(int x1=1; x1<width; x1++)
			{
				const int x2 = width - x1 - 1;
				//const float weight1 = 1.f;
				//const float weight2 = 1.f;
#ifdef UPDATED
				float diff1 = 0, diff2 = 0;
				for(int c=0; c<colors; c++)
				{
					diff1 += abs(Image(x1, y, c) - temporal_color1[c] / temporal_weight1);
					diff2 += abs(Image(x2, y, c) - temporal_color2[c] / temporal_weight2);
				}				
				diff1 = diff1 > 255 * colors ? 255 * colors : diff1;
				diff2 = diff2 > 255 * colors ? 255 * colors : diff2;
				const float weight1 = weights_lookup[(int)diff1];
				const float weight2 = weights_lookup[(int)diff2];
#else
				const float weight1 = getWeight(Image, x1, y, x1 - 1, y, weights_lookup.get());
				const float weight2 = getWeight(Image, x2, y, x2 + 1, y, weights_lookup.get());
#endif

				temporal_weight1 = (1 + temporal_weight1 * sigma_spatial*weight1);
				temporal_weight2 = (1 + temporal_weight2 * sigma_spatial*weight2);
				Weights(x1, y) += temporal_weight1;
				Weights(x2, y) += temporal_weight2;
#ifdef UPDATED
				for (int c = 0; c<colors; c++)
				{
					temporal_color1[c] = Image(x1, y, c) + temporal_weight1 * sigma_spatial*temporal_color1[c];
					temporal_color2[c] = Image(x2, y, c) + temporal_weight2 * sigma_spatial*temporal_color2[c];					
				}
#endif
				#pragma omp parallel for
				for(int l=0; l<layers; l++)
				{
					temporal1[l] = (float(Signal(x1, y, l)) + temporal1[l] * sigma_spatial*weight1);
					temporal2[l] = (float(Signal(x2, y, l)) + temporal2[l] * sigma_spatial*weight2);
					Temporal(x1,y,l) += temporal1[l];
					Temporal(x2,y,l) += temporal2[l];
				}
			}

			#pragma omp parallel for
			for(int x=0; x<width; x++)
			{				
				Weights(x, y) -= 1;
				for(int l=0; l<layers; l++)			
				{
					Temporal(x, y, l) -= float(Signal(x, y, l));
				}
			}
		}
	}	

	//// intermediate normalization
	//for(int x=0; x<width; x++)
	//{
	//	for(int y=0; y<height; y++)
	//	{
	//		for(int l=0; l<layers; l++)		
	//		{
	//			Temporal(x,y,l) /= Weights(x,y);
	//		}			
	//	}
	//}	

	//Weights.setval(0.f);
	
	//vertical passes
	#pragma omp parallel
	{
		
		std::unique_ptr<float[]> temporal1(new float[Signal.layers]);
		std::unique_ptr<float[]> temporal2(new float[Signal.layers]);
		
		float temporal_weight1, temporal_weight2;
#ifdef UPDATED
		std::unique_ptr<float[]> temporal_color1(new float[colors]);
		std::unique_ptr<float[]> temporal_color2(new float[colors]);
#endif		
		#pragma omp for
		for(int x=0; x<width; x++)
		{
			for(int l=0; l<layers; l++)
			{
				temporal1[l] = Temporal(x,0,l);
				temporal2[l] = Temporal(x,height-1,l);		
				Filtered(x,0,l) = temporal1[l];
				Filtered(x,height-1,l) = temporal2[l];
			}
			temporal_weight1 = Weights(x, 0);
			temporal_weight2 = Weights(x, height-1);
			Weights2(x, 0) = temporal_weight1;
			Weights2(x, height-1) = temporal_weight2;
#ifdef UPDATED
			for (int c = 0; c<colors; c++)
			{
				temporal_color1[c] = Image(x, 0, c);
				temporal_color2[c] = Image(x, height-1, c);
			}
#endif
			for(int y1=1; y1<height; y1++)
			{
				const int y2 = height - y1 - 1;
				//const float weight1 = 1.f;
				//const float weight2 = 1.f;
#ifdef UPDATED
				float diff1 = 0, diff2 = 0;
				for (int c = 0; c<colors; c++)
				{
					diff1 += abs(Image(x, y1, c) - temporal_color1[c] / temporal_weight1);
					diff2 += abs(Image(x, y2, c) - temporal_color2[c] / temporal_weight2);
				}
				diff1 = diff1 > 255 * colors ? 255 * colors : diff1;
				diff2 = diff2 > 255 * colors ? 255 * colors : diff2;
				const float weight1 = weights_lookup[(int)diff1];
				const float weight2 = weights_lookup[(int)diff2];
#else
				const float weight1 = getWeight(Image, x, y1, x, y1-1, weights_lookup.get());
				const float weight2 = getWeight(Image, x, y2, x, y2 + 1, weights_lookup.get());
#endif				

				temporal_weight1 = (Weights(x, y1) + temporal_weight1 * sigma_spatial*weight1);
				temporal_weight2 = (Weights(x, y2) + temporal_weight2 * sigma_spatial*weight2);
				Weights2(x, y1) += temporal_weight1;
				Weights2(x, y2) += temporal_weight2;
#ifdef UPDATED
				for (int c = 0; c<colors; c++)
				{
					temporal_color1[c] = Image(x, y1, c) + temporal_weight1 * sigma_spatial*temporal_color1[c];
					temporal_color2[c] = Image(x, y2, c) + temporal_weight2 * sigma_spatial*temporal_color2[c];
				}
#endif			
				#pragma omp parallel for
				for(int l=0; l<layers; l++)
				{
					temporal1[l] = (Temporal(x,y1,l) + temporal1[l] * sigma_spatial*weight1);
					temporal2[l] = (Temporal(x,y2,l) + temporal2[l] * sigma_spatial*weight2);				
					Filtered(x,y1,l) += temporal1[l];
					Filtered(x,y2,l) += temporal2[l];
				}	
			}

			#pragma omp parallel for
			for(int y=0; y<height; y++)
			{				
				Weights2(x, y) -= Weights(x, y);
				for(int l=0; l<layers; l++)			
				{
					Filtered(x,y,l) -= Temporal(x,y,l);
				}
			}
		}
	}

	// final normalization
	#pragma omp parallel for
	for(long i=0; i<HW; i++)
	{
		const int x = i / height;
		const int y = i % height;
		for(int l=0; l<layers; l++)		
		{
			Filtered(x,y,l) /= Weights2(x,y);
		}			
	}
}

//
////template<typename T>
//void recursive_bilateral(MexImage<float> &Signal, MexImage<float> &Filtered, MexImage<float> &Image, const double sigma_color, const double sigma_spatial = 1.0)
//{	
//	const int width = Signal.width;
//	const int height = Signal.height;
//	const long HW = Signal.layer_size;
//	const int layers = Signal.layers;
//	const int colors = Image.layers;
//
//	std::auto_ptr<float> weights_lookup_ptr(new float[256 * colors]);
//	float * const weights_lookup = weights_lookup_ptr.get();
//
//	for (int i = 0; i < 256 * colors; i++)
//	{
//		weights_lookup[i] = float(exp(-double(i) / (colors * sigma_color)));
//	}
//
//	MexImage<float> Aggregated(width, height, layers);
//	//MexImage<float> Weights(width, height);
//	//Weights.setval(0.f);
//	float alpha = exp(-sqrt(2.0) / (sigma_spatial*height));
//	float inv_alpha = 1 - alpha;
//#pragma omp parallel 
//	{
//		std::auto_ptr<float> accumulated_ptr(new float[layers]);
//		float * const accumulated = accumulated_ptr.get();
//
//#pragma omp for
//		for (int x = 0; x < width; x++)
//		{
//			for (int l = 0; l < layers; l++)
//			{
//				accumulated[l] = Signal(x, 0, l);
//				Aggregated(x, 0, l) = accumulated[l];
//			}
//
//			for (int y = 1; y < height; y++)
//			{
//				const float weight = getWeight(Image, x, y - 1, x, y, weights_lookup);
//
//				for (int l = 0; l < layers; l++)
//				{
//					accumulated[l] = accumulated[l] * weight*alpha + Signal(x, y, l) * inv_alpha;
//					Aggregated(x, y, l) = accumulated[l];
//				}
//			}
//
//			for (int l = 0; l < layers; l++)
//			{
//				accumulated[l] = 0.5 * (Signal(x, height - 1, l) + Aggregated(x, height - 1, l));
//				Aggregated(x, height - 1, l) = accumulated[l];
//			}
//
//			for (int y = height - 2; y >= 0; y--)
//			{
//				const float weight = getWeight(Image, x, y + 1, x, y, weights_lookup);
//
//				for (int l = 0; l < layers; l++)
//				{
//					accumulated[l] = accumulated[l] * weight*alpha + Signal(x, y, l) * inv_alpha;
//					Aggregated(x, y, l) = 0.5 * (Aggregated(x, y, l) + accumulated[l]);
//				}
//			}
//		}
//	}
//
//	alpha = exp(-sqrt(2.0) / (sigma_spatial*width));
//	inv_alpha = 1 - alpha;
//
//#pragma omp parallel 
//	{
//		std::auto_ptr<float> accumulated_ptr(new float[layers]);
//		float * const accumulated = accumulated_ptr.get();
//
//#pragma omp for
//		for (int y = 0; y < height; y++)
//		{
//			//float accumulated_weight = Weights(0, y);
//			for (int l = 0; l < layers; l++)
//			{
//				accumulated[l] = Aggregated(0, y, l);
//				Filtered(0, y, l) = accumulated[l];
//			}
//
//			for (int x = 1; x < width; x++)
//			{
//				const float weight = getWeight(Image, x - 1, y, x, y, weights_lookup);
//
//				for (int l = 0; l < layers; l++)
//				{
//					accumulated[l] = accumulated[l] * weight*alpha + Aggregated(x, y, l) * inv_alpha;
//					Filtered(x, y, l) = accumulated[l];
//				}
//			}
//
//			//accumulated_weight = 1.f;
//			//accumulated_weight = Weights(width - 1, y);
//			for (int l = 0; l < layers; l++)
//			{
//				accumulated[l] = 0.5 * (Aggregated(width - 1, y, l) + Filtered(width - 1, y, l));
//				Filtered(width - 1, y, l) = accumulated[l];
//			}
//
//			for (int x = width - 2; x >= 0; x--)
//			{
//				const float weight = getWeight(Image, x + 1, y, x, y, weights_lookup);
//
//				for (int l = 0; l < layers; l++)
//				{
//					accumulated[l] = accumulated[l] * weight*alpha + Aggregated(x, y, l) * inv_alpha;
//					Filtered(x, y, l) = 0.5 * (Filtered(x, y, l) + accumulated[l]);
//				}
//				//Weights(x, y) += accumulated_weight;
//				//accumulated_weight = accumulated_weight*weight + 1;
//			}
//		}
//	}
//
//
//}

void mexFunction(int nout, mxArray* output[], int in, const mxArray* input[])
{
#ifndef _DEBUG
	omp_set_num_threads(std::max<int>(4, omp_get_max_threads()));
	omp_set_dynamic(std::max<int>(2, omp_get_max_threads() / 2));
#endif

	if (in < 2 || in > 4 || nout < 1 || nout > 2 || mxGetClassID(input[1]) != mxSINGLE_CLASS)
	{
		mexErrMsgTxt("USAGE: [Filtered, <Weights>] = recursive_bilateral(Signal, single(Image), <sigma_color, sigma_spatial>);");
	}
		
	const float sigma_color = (in > 2) ? std::max<float>(0.f, std::min<float>(1.f, mxGetScalar(input[2]))): 1.0;
	const float sigma_spatial = (in > 3) ? std::max<float>(0.f, mxGetScalar(input[3])) : 1.0;
	
	MexImage<float> Image(input[1]);
		
	const size_t dims1[] = { (size_t)Image.height, (size_t)Image.width, 1 };
	std::unique_ptr<MexImage<float>> Weights;
	
	if (nout > 1)
	{
		output[1] = mxCreateNumericArray(3, dims1, mxSINGLE_CLASS, mxREAL);
		Weights = std::unique_ptr<MexImage<float>>(new MexImage<float>(output[1]));		
	}
	else
	{
		Weights = std::unique_ptr<MexImage<float>>(new MexImage<float>(Image.width, Image.height));
	}

	if (mxGetClassID(input[0]) == mxSINGLE_CLASS)
	{
		MexImage<float> Signal(input[0]);
		if (Signal.width != Image.width || Signal.height != Image.height)
		{
			mexErrMsgTxt("Signal and Image must have the same resolution");
		}
		const size_t dims[] = { (size_t)Signal.height, (size_t)Signal.width, (size_t)Signal.layers };
		output[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
		MexImage<float> Filtered(output[0]);

		recursive_bilateral_new<float>(Signal, Filtered, Image, *Weights.get(), sigma_color, sigma_spatial);
	}
	else if (mxGetClassID(input[0]) == mxUINT8_CLASS)
	{
		MexImage<unsigned char> Signal(input[0]);
		if (Signal.width != Image.width || Signal.height != Image.height)
		{
			mexErrMsgTxt("Signal and Image must have the same resolution");
		}
		const size_t dims[] = { (size_t)Signal.height, (size_t)Signal.width, (size_t)Signal.layers };
		output[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
		MexImage<float> Filtered(output[0]);

		recursive_bilateral_new<unsigned char>(Signal, Filtered, Image, *Weights.get(), sigma_color, sigma_spatial);
	}
	else
	{
		mexErrMsgTxt("Unsupported type of Signal");
	}
	
	
}



float getWeight(const MexImage<float> &Image, const int x1, const int y1, const int x2, const int y2, const float * const weights_lookup)
{
#ifdef WEIGHTS_ORIGINAL
	int max_diff = 0;
	for (int c = 0; c < Image.layers; c++)
	{
		int diff = int(abs(Image(x1, y1, c) - Image(x2, y2, c)));
		max_diff = diff > max_diff ? diff : max_diff;
	}
	return weights_lookup[max_diff];
#else
	int diff = 0;
	for (int c = 0; c < Image.layers; c++)
	{
		diff += int(abs(Image(x1, y1, c) - Image(x2, y2, c)));
		//int diff = int(abs(Image(x1, y1, c) - Image(x2, y2, c)));
		//max_diff = diff > max_diff ? diff : max_diff;
	}
	return exp(-float(diff) / (255 * sigma_color*Image.layers));
#endif
}

