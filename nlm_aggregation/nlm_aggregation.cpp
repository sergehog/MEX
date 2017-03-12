/** 
*	@file nlm_aggregation.cpp
*	@date 13.06.2014
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


void mexFunction (int nout, mxArray* output[], int in, const mxArray* input[])
{
	omp_set_num_threads(std::max<int>(4, omp_get_max_threads()));
	omp_set_dynamic(omp_get_max_threads()/2);
	
	if(in < 4 || in > 6 || nout != 1 || mxGetClassID(input[0]) != mxSINGLE_CLASS || mxGetClassID(input[1]) != mxSINGLE_CLASS)
	{
		mexErrMsgTxt("USAGE: [Filtered] = nlm_aggregation(single(Signal), single(Image), patch_radius, search_radius [, sigma_distance, distance_threshold]);");
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
		

	const int patch_radius = (int)mxGetScalar(input[2]);
	const int patch_diameter = patch_radius*2+1;
	const int patch_window = patch_diameter*patch_diameter;

	const int search_radius = (int)mxGetScalar(input[3]);
	const int search_diameter = (search_radius*2+1);
	const int search_window = search_diameter*search_diameter;
	
	if(patch_radius < 1 || search_radius < 1)
	{
		mexErrMsgTxt("Both patch_radius and search_radius must be positive.");
	}

	const float sigma_distance = (in > 4) ? (float)mxGetScalar(input[4]) : 10.;
	const float distance_threshold = (in > 5) ? (float)mxGetScalar(input[5]) : 255;
	const float cost_threshold = 255.f;

	const size_t dims[] = {(size_t)height, (size_t)width, (size_t)layers};	
	const size_t dimsC[] = {(size_t)height, (size_t)width, (size_t)colors};	

	output[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL); 	

	MexImage<float> Filtered(output[0]);
	MexImage<float> Weights(width, height, 1);

	std::unique_ptr<float> weights_table_ptr = std::unique_ptr<float>(new float[256*colors]);
	std::unique_ptr<float> gaussian_table_ptr = std::unique_ptr<float>(new float[search_window]);

	float * const weights_table = weights_table_ptr.get();
	float * const gaussian_table = gaussian_table_ptr.get();


	#pragma omp parallel sections
	{
		#pragma omp section
		{
			#pragma omp parallel for
			for(int i=0; i<255*colors; i++)
			{
				weights_table[i] = exp(-i/(colors*sigma_distance));
			}	
		}
		#pragma omp section
		{			
			#pragma omp parallel for
			for(int i=0; i<search_window; i++)
			{		
				const int x = i/search_diameter - search_radius;
				const int y = i%search_diameter - search_radius;				 
				gaussian_table[i] = exp(-3*sqrt(float(x*x + y*y))/search_radius);
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

			Cost.IntegralImage(true);

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
				float value = Cost.getIntegralAverage(x, y, patch_radius);
				value = value < 0 ? 0 : value;

				if(value < distance_threshold)
				{
					//const float weight = weights_table[mymex::round(value)] * gaussian_table[i];
					const float weight = weights_table[mymex::round(value)];
					for(int j=0; j<patch_window; j++)
					{
						const int dxp = j/patch_diameter - patch_radius;
						const int dyp = j%patch_diameter - patch_radius;

						const int xp = x + dxp;
						const int yp = y + dyp;
						const int xsp = xs + dxp;
						const int ysp = ys + dyp;

						if(xsp < 0 || xsp >= width || ysp < 0 || ysp >= height)
						{
							continue;
						}
						if(xp < 0 || xp >= width || yp < 0 || yp >= height)
						{
							continue;
						}

						for(int c=0; c<layers; c++)
						{
							Filtered(xp, yp, c) += Signal(xsp, ysp, c) * weight;
							Filtered(xsp, ysp, c) += Signal(xp, yp, c) * weight;
						}

						Weights(xsp, ysp) += weight;
						Weights(xp, yp) += weight;
					}	
				}
				
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
