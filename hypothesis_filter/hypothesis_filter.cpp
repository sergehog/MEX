/** Fast Implementation of Super-Resolution Range-Image Filter
*	@file hypothesis_filter.cpp
*	@date 01.01.2008
*	@author Sergey Smirnov <sergey.smirnov@tut.fi>
*/

#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}
#endif // __cplusplus

#include "../meximage/meximage.h"
#include <cmath>
#include <limits>
#include <algorithm>
#include <omp.h>

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")

//#ifdef _MSC_VER
//	#define isnan _isnan
//#endif

//struct depth_limits
//{
//	float maxdepth;
//	float mindepth;
//
//public: 
//	depth_limits(float val) : maxdepth(val), mindepth(val) {};
//
//private:
//	depth_limits() {};
//};

using namespace mymex;

void prepare_color_weights(float * const color_weights, const float sigma_color, const int maxvalue, const int colors);
void prepare_spatial_weights(float * const spatial_weights, const float sigma_distance, const int radius);
//depth_limits read_depth(MexImage<float> &Depth, float* const depths, const int x, const int y, const int radius);

void mexFunction (int nout, mxArray* output[], int in, const mxArray* input[])
{
	omp_set_num_threads(std::max(2, omp_get_max_threads()));
	omp_set_dynamic(0);

	if(in < 2 || in > 5 || nout != 1 || mxGetClassID(input[0]) != mxSINGLE_CLASS || mxGetClassID(input[1]) != mxUINT8_CLASS)
	{
		mexErrMsgTxt("USAGE: [ResultDepth] = hypothesis_filter(single(Depth), uint8(Image), <sigma_color, sigma_distance, search_limit>);");
    }
	// Input depth + view
	MexImage<float> Depth(input[0]);
	MexImage<uint8> Image(input[1]);	

	// const params
	const int height = Depth.height;
	const int width = Depth.width;
	const int HW = width*height;
	const int colors = Image.layers;	

	if(height != Image.height || width != Image.width)
	{
		mexErrMsgTxt("Width and height of both input images must coincide."); 
	}

	// other input params
    const float sigma_color = (in > 2) ? (float)mxGetScalar(input[2]) : 20.f;
    const float sigma_distance = (in > 3) ? (float)mxGetScalar(input[3]) : 3.f;
	const float search_limit = (in > 4) ? (float)mxGetScalar(input[4]) : 2.5f;
	const int radius = (int)floor(sigma_distance*2.5); // plus/minus two and half sigmas region
    const int diameter = radius*2 + 1;
	const int window = diameter*diameter;	
	//const float nan = std::numeric_limits<float>::quiet_NaN();
	
	// Matlab-allocated output filtered depth
#ifdef _MSC_VER
	const size_t dims[] = {height, width, 1};
	output[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);	
#else
	const int dims[] = {height, width, 1};
	output[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);	
#endif

	MexImage<float> Filtered(output[0]);
	MexImage<bool> Valid(width, height);

	for(long i=0; i<HW; i++)
	{
		Valid.data[i] = !_isnan(Depth[i]);		
	}

	// create look-up tables
	float* const color_weights = new float[255*colors+1];
	float* const spatial_weights = new float[window];	
	
	// fill-in look-up tables
	#pragma omp parallel sections
	{
		#pragma omp section
		{
			prepare_color_weights(color_weights, sigma_color, 255, colors);    
		}
		#pragma omp section
		{
			prepare_spatial_weights(spatial_weights, sigma_distance, radius);
		}
	}
	
	// start parallelized processing
	#pragma omp parallel 
	{	
		int maxlayers = 1000; // that is not a constant, but large enough value to avoid re-allocations
		float* costs = new float[maxlayers];
		float* const weights = new float[window];		
		float* const depths = new float[window]; 		
		
		#pragma omp for
		for(int index=0; index<HW; index ++)
		{
			int x = index / height;
			int y = index % height;			

			float depth = Depth[index];
			float maxdepth = depth;
			float mindepth = depth;
			float summ = 0;

			for(int i=0; i<window; i++)
			{
				depths[i] = 0;
				weights[i] = 0.f;
			}

			// read depths and calculate bilateral weights
			for(int sx=x-radius, i=0; sx<=x+radius; sx++)
			{
				for(int sy=y-radius; sy<=y+radius; sy++, i++)
				{					
					if(sx < 0 || sx >= Depth.width || sy < 0 || sy >= Depth.height )
					{
						continue;
					}

					long sindex = Depth.Index(sx,sy);

					if(!Valid[sindex])
					{
						continue;
					}

					depths[i] = Depth[sindex];
					maxdepth =  (_isnan(maxdepth) || depths[i] > maxdepth) ? depths[i] : maxdepth;
					mindepth = (_isnan(mindepth) || depths[i] < mindepth) ? depths[i] : mindepth;
						
					int color_difference = 0;
					for(int c=0; c<colors; c++)
					{							
						color_difference += abs(int(Image[index + c*HW]) - Image[sindex + c*HW]);
					}

					weights[i] = color_weights[color_difference] * spatial_weights[i];
					summ += weights[i];
					
				}
			}

			// skip processing if depth range is too small
			if(maxdepth - mindepth < 0.5 && Valid[index])
			{
				Filtered.data[index] = Depth[index];
				continue;
			}			

			for(int i=0; i<window; i++)
			{
				weights[i] /= summ;
			}

			maxdepth = ceil(maxdepth+1);
			mindepth = floor(mindepth-1);
			
			int layers = int(maxdepth - mindepth);

			// ensure that allocated 'costs' large enough :) 
			if(layers >= maxlayers)
			{
				delete[] costs;
				maxlayers = layers*2;
				costs = new float[maxlayers];
			}

			float best_cost = search_limit;
			int best_layer = 0;
			float best_d = 0;

			// go thru layers 
			for(int layer=0; layer <= layers; layer ++)
			{
				float d = mindepth + layer;				
				
				float value = 0;
				for(int i=0; i<window; i++)
				{
					float diff = d-depths[i];
					diff *= diff;
					diff = diff>search_limit ? search_limit : diff;
					value += diff*weights[i];
				}
				
				costs[layer] = value; 

				if(value <= best_cost)
				{
					best_cost = value;
					best_d = d;
					best_layer = layer;
				}
			}

			float cost_down = (best_layer > 0) ? costs[best_layer-1] : search_limit*2;
			float cost_up = (best_layer < layers) ? costs[best_layer+1] : search_limit*2; 
			float filtered  = best_d - (cost_up-cost_down)/(2*(cost_up+cost_down-2*best_cost));
			Filtered.data[index] = filtered;
			//Filtered.data[index] = best_d;
		}
		
		delete[] costs, weights, depths;
	}

	delete[] spatial_weights;
	delete[] color_weights;
}

//! Color normalization made in the look-up table
void prepare_color_weights(float * const color_weights, const float sigma_color, const int maxvalue, const int colors)
{
	#pragma omp parallel for
    for(int i=0; i<maxvalue*colors; i++)
    {
        color_weights[i] = exp(-((float)i)/(colors*sigma_color));
    }
}

//! 
void prepare_spatial_weights(float * const spatial_weights, const float sigma_distance, const int radius)
{
    // pre-calculation of distance weights
	//#pragma omp parallel for
	for(int i=-radius, index = 0; i<=radius; i++)
	{
		for(int j=-radius; j<=radius; j++, index++)
		{
			spatial_weights[index] = exp(-sqrt((float)i*i+j*j)/sigma_distance);
		}
	}
}
