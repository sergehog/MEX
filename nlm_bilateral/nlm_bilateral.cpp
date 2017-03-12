/** 
*	@file nlm_bilateral.cpp
*	@date 18.11.2014
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
#include <memory>

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")

typedef unsigned char uint8;
using namespace mymex;

float getWeight(const MexImage<float> &Image, const int x1, const int y1, const int x2, const int y2, const float * const weights_lookup)
{
	int max_diff = 0;
	for (int c = 0; c < Image.layers; c++)
	{
		int diff = int(abs(Image(x1, y1, c) - Image(x2, y2, c)));
		max_diff = diff > max_diff ? diff : max_diff;
	}
	return weights_lookup[max_diff];
}

void recursive_bilateral(MexImage<float> &Signal, MexImage<float> &Temporal, MexImage<float> &Weights, MexImage<float> &Weights2, const MexImage<float> &Color, float const * const weights_lookup, float const sigma_spatial)
{	
	const int width = Signal.width;
	const int height = Signal.height;	
	const long HW = Signal.layer_size;	

	//const float alpha = exp(-sqrt(2.0) / (sigma_spatial*std::min(height, width)));
	const float alpha = sigma_spatial;

	Weights.setval(0.f);	
	Temporal.setval(0.f);	


	// horisontal passes
	#pragma omp parallel for
	for(int y=0; y<height; y++)
	{
		float t1 = Signal(0,y);
		float t2 = Signal(width-1,y);				
		float w1 = 1.f;
		float w2 = 1.f;			

		Temporal(0,y) = t1;
		Temporal(width-1,y) = t2;
		Weights(0, y) = w1;
		Weights(width-1, y) = w2;

		for(int x1=1; x1<width; x1++)
		{
			#pragma omp parallel sections
			{
				#pragma omp section
				{					
					const float weight1 = getWeight(Color, x1, y, x1-1,y,weights_lookup);
					t1 = Signal(x1,y) + t1 * alpha*weight1;
					w1 = (1 + w1 * alpha*weight1);
					#pragma omp atomic
					Temporal(x1,y) += t1;
					#pragma omp atomic
					Weights(x1, y) += w1;
				}
				#pragma omp section
				{
					const int x2 = width - x1 - 1;
					const float weight2 = getWeight(Color, x2, y, x2+1,y,weights_lookup);
					t2 = Signal(x2,y) + t2 * alpha*weight2;							
					w2 = (1 + w2 * alpha*weight2);			
					#pragma omp atomic
					Temporal(x2,y) += t2;			
					#pragma omp atomic
					Weights(x2, y) += w2;
				}
			}			
		}

		#pragma omp parallel for
		for(int x=0; x<width; x++)
		{				
			Weights(x, y) -= 1;
			Temporal(x,y) -= Signal(x,y);
		}
	}	
	
	Signal.setval(0.f);
	Weights2.setval(0.f);

	//vertical passes		
	#pragma omp parallel for
	for(int x=0; x<width; x++)
	{
		float t1 = Temporal(x,0);
		float t2 = Temporal(x, height-1);				
		float w1 = Weights(x, 0);
		float w2 = Weights(x, height-1);		

		Signal(x,0) = t1;
		Signal(x, height-1) = t2;
		Weights2(x,0) = w1;
		Weights2(x, height-1) = w2;

		for(int y1=1; y1<height; y1++)
		{
			#pragma omp parallel sections
			{
				#pragma omp section
				{	
					const float weight1 = getWeight(Color, x, y1, x,y1-1,weights_lookup);
					t1 = Temporal(x,y1) + t1 * alpha*weight1;
					w1 = (Weights(x, y1)  + w1 * alpha*weight1);
					#pragma omp atomic
					Signal(x,y1) += t1;
					#pragma omp atomic
					Weights2(x,y1) += w1;
				}
				#pragma omp section
				{
					const int y2 = height - y1 - 1;
					const float weight2 = getWeight(Color, x, y2, x,y2+1,weights_lookup);
					t2 = Temporal(x, y2) + t2 * alpha*weight2;							
					w2 = (Weights(x, y2)  + w2 * alpha*weight2);			
					#pragma omp atomic
					Signal(x, y2) += t2;			
					#pragma omp atomic
					Weights2(x, y2) += w2;
				}
			}
		}

		#pragma omp parallel for
		for(int y=0; y<height; y++)
		{				
			Weights2(x,y) -= Weights(x, y);
			Signal(x,y) -= Temporal(x,y);
		}
	}
	
	// final normalization
	#pragma omp parallel for
	for(long i=0; i<HW; i++)
	{
		Signal[i] /= Weights2[i];
	}
}


void mexFunction (int nout, mxArray* output[], int in, const mxArray* input[])
{
	omp_set_num_threads(std::max<int>(4, omp_get_max_threads()));
	omp_set_dynamic(omp_get_max_threads()/2);
	
	if(in != 5 || nout != 1 || mxGetClassID(input[0]) != mxSINGLE_CLASS || mxGetClassID(input[1]) != mxSINGLE_CLASS)
	{
		mexErrMsgTxt("USAGE: [Filtered] = nlm_bilateral(single(Signal), single(Image), sc, sd, search_radius);");
    }

	MexImage<float> Signal(input[0]);	
	MexImage<float> Image(input[1]);

	const int height = Image.height;
	const int width = Image.width;
	const int colors = Image.layers;
	const int layers = Signal.layers;
	const long HW = Image.layer_size;
	const float nan = sqrt(-1.f);

	if(Signal.height != height || Signal.width != width)
	{
		mexErrMsgTxt("Image and Signal must have the same width and height.");
	}
		
	//const float sigma_color = std::max(0.f, std::min(1.f, (float) mxGetScalar(input[2])));
	const float sigma_color = (float) mxGetScalar(input[2]);
	const float sigma_distance = std::max(0.f, std::min(1.f, (float) mxGetScalar(input[3])));
	const int search_radius = (int)mxGetScalar(input[4]);
	const int search_diameter = (search_radius*2+1);
	const int search_window = search_diameter*search_diameter;
	const float cost_threshold = 50.f;

	if(search_radius < 1)
	{
		mexErrMsgTxt("search_radius must be positive.");
	}

	const size_t dims[] = {(size_t)height, (size_t)width, (size_t)layers};	
	//const size_t dimsC[] = {(size_t)height, (size_t)width, (size_t)colors};	

	output[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL); 	

	MexImage<float> Filtered(output[0]);
	MexImage<float> Weights(width, height, 1);

	std::unique_ptr<float[]> weights_table = std::unique_ptr<float[]>(new float[256*colors]);

#pragma omp parallel sections
	{
		#pragma omp section
		{
			#pragma omp parallel for
			for(int i=0; i<255*colors; i++)
			{
				weights_table[i] = exp(-i/(colors*sigma_color));
			}	
		}		
		#pragma omp section
		{
			#pragma omp parallel for
			for(int i=0; i<HW*layers; i++)
			{
				Filtered[i] = Signal[i];
			}
		}
		#pragma omp section
		{
			#pragma omp parallel for
			for(int i=0; i<HW; i++)
			{				
				Weights[i] = 1.f;		
			}
		}
	}	
	#pragma omp parallel
	{
		MexImage<float> Cost(width, height);
		MexImage<float> Weights1(width, height);
		MexImage<float> Weights2(width, height);
		MexImage<float> Temporal(width, height);

		#pragma omp for
		for(int i=0; i<search_window/2; i++)
		{
			const int dxs = i/search_diameter - search_radius;
			const int dys = i%search_diameter - search_radius;

			//if(dxs==0 && dys == 0)
			//	continue;

			Cost.setval(cost_threshold);

			#pragma omp parallel for
			for(int index=0; index<HW; index++)
			{
				const int y = index % height;
				const int x = index / height;
				const int xs = x + dxs;
				const int ys = y + dys;

				if(xs < 0 || xs >= width || ys < 0 || ys >= height)
				{
					continue;
				}				
					
				float cost = 0;
				for(int c=0; c<colors; c++)
				{
					cost += std::abs(Image(x,y,c) - Image(xs,ys,c));
				}
				Cost(x,y) = cost/colors;
			}

			recursive_bilateral(Cost, Temporal, Weights1, Weights2, Image, weights_table.get(), sigma_distance);				
			//Cost.IntegralImage(true);

			#pragma omp parallel for
			for(int index=0; index<HW; index++)
			{
				const int x = index / height;
				const int y = index % height;
										
				const int xs = x + dxs;
				const int ys = y + dys;

				if(xs < 0 || xs >= width || ys < 0 || ys >= height)
				{
					continue;
				}
				
				//const long indexs = Image.Index(xs, ys);
				float value = Cost(x,y);//.getIntegralAverage(x, y, patch_radius);
				value = value < 0 ? 0 : value;

				//const float weight = weights_table[mymex::round(value)] * gaussian_table[i];
				const float weight = weights_table[mymex::round(value)];
				for(int c=0; c<layers; c++)
				{
					Filtered(x, y, c) += Signal(xs, ys, c) * weight;
					Filtered(xs, ys, c) += Signal(x, y, c) * weight;
				}

				Weights(xs, ys) += weight;
				Weights(x, y) += weight;				
			}								
		}
	}

	
	#pragma omp parallel for
	for(int i=0; i<HW; i++)
	{		
		const float weight = Weights[i];
		for(int c=0; c<layers; c++)
		{
			Filtered.data[i+c*HW] /= weight;
		}	
	}
}
