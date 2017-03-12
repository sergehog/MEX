/** Cross-filtering of Non-local means (can be used for aggregation)
*	Implementation optimized with Integral Images
*	@file cross_nl_means.cpp
*	@date 3.04.2012
*	@author Sergey Smirnov <sergey.smirnov@tut.fi>
*/

#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}
#endif // __cplusplus

#include "../common/meximage.h"
//#include "../common/defines.h"
#include <cmath>
#include <algorithm>
#include <omp.h>

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")

typedef unsigned char uint8;
using namespace mymex;

//void calculateAverage(MexImage<float> &Image, MexImage<float> &Average, const int radius)
//{
//	const int width = Image.width;
//	const int height = Image.height;
//	const int HW = Image.layer_size;
//	const int colors = Image.layers;
//
//	#pragma omp parallel for
//	for(int x=0; x <= width-1; x++)
//	{			
//		for(int y=0; y <= height-1; y++)
//		{
//			long index = Image.Index(x,y);
//			for(int c=0; c<colors; c++)
//			{
//				Average.data[index + c*HW] = 0;
//			}
//
//			int pixels = 0;
//
//			for(int sx=std::max(0, x-radius); sx<=std::min(width-1, x+radius); sx++)
//			{
//				for(int sy=std::max(0, y-radius); sy<=std::min(height-1, y+radius); sy++)
//				{
//					long sindex = Average.Index(sx, sy);
//					
//					for(int c=0; c<colors; c++)
//					{
//						Average.data[index + c*HW] += Image[sindex + c*HW];
//					}
//					pixels ++;
//				}
//			}
//			if(pixels > 0)
//			{
//				for(int c=0; c<colors; c++)
//				{
//					Average.data[index + c*HW] /= pixels;
//				}
//			}
//		}
//	}
//}


void mexFunction (int nout, mxArray* output[], int in, const mxArray* input[])
{
	omp_set_num_threads(std::max<int>(4, omp_get_max_threads()));
	omp_set_dynamic(true);
	
	if(in < 4 || in > 6 || nout != 1 || mxGetClassID(input[0]) != mxSINGLE_CLASS || mxGetClassID(input[1]) != mxSINGLE_CLASS)
	{
		mexErrMsgTxt("USAGE: [Filtered] = cross_nl_means(single(Signal), single(Image), pixel_radius, search_radius [, sigma_distance, distance_threshold]);");
    }

	MexImage<float> Signal(input[0]);	
	MexImage<float> Image(input[1]);
	
	//const bool aggregate = true;

	const int height = Image.height;
	const int width = Image.width;
	const int colors = Image.layers;
	const int layers = Signal.layers;
	const long HW = Image.layer_size;
	const float nan = sqrt(-1.f);

	if(Signal.height != height || Signal.width != width)
	{
		mexErrMsgTxt("Edges and Signal must have the same width and height.");
	}
			
	const int radius = std::max(0, (int)mxGetScalar(input[2]));
	const int search_radius = (int)mxGetScalar(input[3]);
	const int search_diameter = (search_radius*2-1);
	const int search_window = search_diameter*search_diameter;
	
	const float sigma_distance = (in > 4) ? (float)mxGetScalar(input[4]) : 10.;
	const float distance_threshold = (in > 5) ? (float)mxGetScalar(input[5]) : 255;
	const float cost_threshold = 256.f;

	const size_t dims[] = {(size_t)height, (size_t)width, (size_t)layers};	
	const size_t dimsC[] = {(size_t)height, (size_t)width, (size_t)colors};	

	output[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL); 	
	//output[1] = mxCreateNumericArray(3, dimsC, mxSINGLE_CLASS, mxREAL); 	

	MexImage<float> Filtered(output[0]);
	//MexImage<float> Average(width, height, colors);
	MexImage<float> Weights(width, height, 1);

	float *weights_table = new float[256*colors];
	for(int i=0; i<255*colors; i++)
	{
		weights_table[i] = exp(-i/(colors*sigma_distance));
	}	
	
	//calculateAverage(Image, Average, radius);
	
	#pragma omp parallel for
	for(int i=0; i<HW; i++)
	{				
		//float value = Signal[i];
		//float weight;
		//if(_isnan(value))
		//{
		//	Weights.data[i] = 0.f;
		//	for(int c=0; c<layers; c++)
		//	{
		//		Filtered.data[i+c*HW] = 0.f;
		//	}
		//}
		//else
		//{
		//	Weights.data[i] = 1.f;
		//	for(int c=0; c<layers; c++)
		//	{
		//		Filtered.data[i+c*HW] = 0.f;
		//	}
		//}
		//float weight = 1.f;
		Weights.data[i] = 1.f;
		for(int c=0; c<layers; c++)
		{
			long cHW = c*HW;
			Filtered.data[i+cHW] = Signal[i+cHW];
		}
	}
	
	if(search_radius > 0)
	//#pragma omp parallel
	{
		MexImage<float> Cost(width, height);

		//#pragma omp for
		for(int i=0; i<search_window; i++)
		{
			int dxs = i/search_diameter - search_radius;
			int dys = i%search_diameter - search_radius;

			if(dxs==0 && dys == 0)
				continue;

			Cost.setval(cost_threshold);

			#pragma omp parallel for
			for(int index=0; index<HW; index++)
			{
				int y = index % height;
				int x = index / height;
				int xs = x + dxs;
				int ys = y + dys;
				if(xs < 0 || xs >= width || ys < 0 || ys >= height)
				{
					continue;
				}
				
				long indexs = Image.Index(xs, ys);
					
				float cost = 0;
				for(int c=0; c<colors; c++)
				{
					float diff = (Image[index + c*HW] - Image[indexs + c*HW]);
					cost += std::abs(diff);
				}
				Cost.data[index] = cost;
			}

			if(radius > 0)
			{
				Cost.IntegralImage(true);
			}

			#pragma omp parallel for
			for(int index=0; index<HW; index++)
			{
				int y = index % height;
				int x = index / height;
						
				int xs = x + dxs;
				int ys = y + dys;
				if(xs < 0 || xs >= width || ys < 0 || ys >= height)
				{
					continue;
				}
				//xs = xs < 0 ? 0 : ((xs >= width) ? width-1 : xs);
				//ys = ys < 0 ? 0 : ((ys >= height) ? height-1 : ys);
				
				long indexs = Image.Index(xs, ys);
				float value = (radius > 0) ? std::abs(Cost.getIntegralAverage(x, y, radius)) : Cost[index];

				if(value/colors < distance_threshold)
				{
					//float weight = exp(-value*value/sigma_distance);
					//float weight = exp(-value/sigma_distance);
					float weight = weights_table[mymex::round(value)];

					for(int c=0; c<layers; c++)
					{
						long cHW = c*HW;
						float input = Signal[indexs + cHW];
						//if(!_isnan(input))
						//{
						//#pragma omp critical
						//{
						Filtered.data[index + cHW] += input*weight;									
						//}
						//}							
					}
					//#pragma omp critical
					{
						Weights.data[index] += weight;
					}
					
				}
				
			}		
		}
	}

	delete[] weights_table;


	#pragma omp parallel for
	for(int i=0; i<HW; i++)
	{		
		float weight = Weights[i];
		if(weight > 0.0f)
		{
			for(int c=0; c<layers; c++)
			{
				Filtered.data[i+c*HW] /= weight;
			}	
		}
		else
		{
			for(int c=0; c<layers; c++)
			{
				Filtered.data[i+c*HW] = nan;
			}
		}
	}
}
