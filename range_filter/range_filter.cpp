//range_filter
/** fast_range_filter
*	@file fast_range_filter.cpp
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
//#include <vector>
//#define _USE_MATH_DEFINES
//#include <cmath>
//#include <algorithm>
#include <memory>

#ifndef _DEBUG
#include <omp.h>
#endif

//#define M_PI       3.14159265358979323846

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")

typedef unsigned char uint8;
using namespace mymex;


template<typename IT>
float getWeight(MexImage<IT> &Image, const int x1, const int y1, const int x2, const int y2, const float * const weights_lookup)
{	
	float weight = 0;
	for(int c=0; c<Image.layers; c++)
	{
		weight += abs(float(Image(x1,y1,c))-float(Image(x2,y2,c)));
	}

	
	return weights_lookup[(int)round(weight)];
	//return exp(-weight/(10*Image.layers));
}

template<typename IT>
void calculate_weights(MexImage<IT> &Image, MexImage<float> &Weights, const int radius, const double sigma_distance, IT maxvalue)
{
	const int width = Image.width;
	const int height = Image.height;
	const int colors = Image.layers;
	const long HW = Image.layer_size;	
	const int diameter = radius*2+1;
	const int window = diameter*diameter;

	std::unique_ptr<float> weights_lookup_ptr = std::unique_ptr<float>(new float[static_cast<int>(colors*maxvalue)]);
	float* const weights_lookup = weights_lookup_ptr.get();
	
	// prepare lookup table
	#pragma omp parallel for
	for(int i=0; i<int(colors*maxvalue); i++)
	{
		weights_lookup[i] = (float) exp(-i/(colors*sigma_distance));
	}

	for(int w=0; w<window/2; w++)
	{
		const int dx = w/diameter - radius;
		const int dy = w%diameter - radius;
		
		#pragma omp parallel for
		for(long i=0; i<HW; i++)
		{
			const int x = i / height;
			const int y = i % height;

			const int xx = x + dx;
			const int yy = y + dy;

			if(xx < 0 || xx >= width || yy<0 || yy>= height)
			{
				continue;
			}

			const float weight = getWeight<IT>(Image, x, y, xx, yy, weights_lookup);

			Weights(x,y,w) = weight;
		}
	}
}

template<typename IT>
void single_range_filter(MexImage<float> &Signal, MexImage<float> &Filtered, MexImage<IT> &Image, const int radius, const double sigma_distance, IT maxvalue)
{
	const int width = Signal.width;
	const int height = Signal.height;
	const long HW = Signal.layer_size;
	const int layers = Signal.layers;	
	const int diameter = radius*2+1;
	const int window = diameter*diameter;
	const int colors = Image.layers;
	
	float* const weights_lookup = new float[colors*maxvalue];	
	MexImage<float> AccWeights(width, height);
	AccWeights.setval(1);

	// prepare lookup table
	#pragma omp parallel for
	for(int i=0; i<int(colors*maxvalue); i++)
	{
		weights_lookup[i] = (float) exp(-i/(colors*sigma_distance));
	}			


	#pragma omp parallel for
	for(long i=0; i<HW*layers; i++)
	{
		Filtered[i] = Signal[i];
	}

	
	for(int w=0; w<window/2; w++)
	{
		const int dx = w/diameter - radius;
		const int dy = w%diameter - radius;
		
		//if(dx==0 && dy==0) // condition in the main cycle (w<search_window/2) already excludes such possibility
		//{
		//	continue;
		//}

		#pragma omp parallel for
		for(long i=0; i<HW; i++)
		{
			const int x = i / height;
			const int y = i % height;

			const int xx = x + dx;
			const int yy = y + dy;

			if(xx < 0 || xx >= width || yy<0 || yy>= height)
			{
				continue;
			}

			const float weight = getWeight<IT>(Image, x, y, xx, yy, weights_lookup);
			#pragma omp atomic
			AccWeights(x,y) += weight;

			#pragma omp atomic
			AccWeights(xx,yy) += weight;
			
			for(int l=0; l<layers; l++)
			{				
				#pragma omp atomic
				Filtered(x,y,l) += Signal(xx,yy,l)*weight;

				#pragma omp atomic
				Filtered(xx,yy,l) += Signal(x,y,l)*weight;				
			}
		}
	}
	
	#pragma omp parallel for
	for(long i=0; i<HW; i++)
	{
		//if(AccWeights[i] > 0)
		for(int l=0; l<layers; l++)
		{
			Filtered[i+l*HW] /= (AccWeights[i]);
		}
	}
	delete[] weights_lookup;
}

void fast_range_filter(MexImage<float> &Signal, MexImage<float> &Filtered, MexImage<float> &Weights, const int radius)
{
	const int width = Signal.width;
	const int height = Signal.height;
	const long HW = Signal.layer_size;
	const int layers = Signal.layers;	
	const int diameter = radius*2+1;
	const int window = diameter*diameter;
	
	MexImage<float> AccWeights(width, height);

	#pragma omp parallel sections
	{
		#pragma omp section
		{			
			AccWeights.setval(1);
		}

		#pragma omp section
		{
			#pragma omp parallel for
			for(long i=0; i<HW*layers; i++)
			{
				Filtered[i] = Signal[i];
			}
		}
	}
	
	for(int w=0; w<window/2; w++)
	{
		const int dx = w/diameter - radius;
		const int dy = w%diameter - radius;
		
		//if(dx==0 && dy==0) // condition in the main cycle (w<search_window/2) already excludes such possibility
		//{
		//	continue;
		//}

		#pragma omp parallel for
		for(long i=0; i<HW; i++)
		{
			const int x = i / height;
			const int y = i % height;

			const int xx = x + dx;
			const int yy = y + dy;

			if(xx < 0 || xx >= width || yy<0 || yy>= height)
			{
				continue;
			}

			//const float weight = getWeight<IT>(Image, x, y, xx, yy, weights_lookup);
			const float weight = Weights(x,y,w);
			AccWeights(x,y) += weight;
			AccWeights(xx,yy) += weight;

			for(int l=0; l<layers; l++)
			{
				Filtered(x,y,l) += Signal(xx,yy,l)*weight;
				Filtered(xx,yy,l) += Signal(x,y,l)*weight;
			}
		}
	}
	
	#pragma omp parallel for
	for(long i=0; i<HW; i++)
	{
		for(int l=0; l<layers; l++)
		{
			Filtered[i+l*HW] /= (AccWeights[i]);
		}
	}
}

//#define EXPERIMENTAL_WEIGHT

void mexFunction (int nout, mxArray* output[], int in, const mxArray* input[])
{
#ifndef _DEBUG
	omp_set_num_threads(std::max<int>(4, omp_get_max_threads()));
	omp_set_dynamic(std::max<int>(2, omp_get_max_threads()/2));
#endif
	
	if(in < 3 || in > 5 || nout != 1 || mxGetClassID(input[0]) != mxSINGLE_CLASS)
	{
		mexErrMsgTxt("USAGE: [Filtered] = fast_range_filter(single(Signal), Image, radius, <sigma_difference, repeat>);");
    }
	 
	const int radius = std::max(1, (int)mxGetScalar(input[2])); 
	const double sigma_distance = (in > 3) ? mxGetScalar(input[3]) : 1.0; 	
	const unsigned repeat = (in > 4) ? (unsigned)mxGetScalar(input[4]) : 0;
	const int diameter = radius*2+1;
	const int window = diameter*diameter;

	MexImage<float> Signal(input[0]);
	const size_t dims[] = {(size_t)Signal.height, (size_t)Signal.width, (size_t)Signal.layers};
	output[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL); 	
	MexImage<float> Filtered(output[0]);

	if(mxGetClassID(input[1]) == mxUINT8_CLASS)
	{
		MexImage<unsigned char> Image(input[1]);
				
		if(repeat > 0)
		{
			MexImage<float> Weights(Signal.width, Signal.height, window/2);
			calculate_weights<unsigned char>(Image, Weights, radius, sigma_distance, 255);
			fast_range_filter(Signal, Filtered, Weights, radius);
			for(int r=0; r<=repeat/2; r++)
			{
				fast_range_filter(Filtered, Signal, Weights, radius);
				fast_range_filter(Signal, Filtered, Weights, radius);
			}
		}
		else
		{
			single_range_filter<unsigned char>(Signal, Filtered, Image, radius, sigma_distance, 255);
		}
	}
	else if(mxGetClassID(input[1]) == mxUINT16_CLASS)
	{
		MexImage<unsigned short> Image(input[1]);
		
		if(repeat > 0)
		{
			MexImage<float> Weights(Signal.width, Signal.height, window/2);
			calculate_weights<unsigned short>(Image, Weights, radius, sigma_distance, 65535);
			fast_range_filter(Signal, Filtered, Weights, radius);
			for(int r=0; r<=repeat/2; r++)
			{
				fast_range_filter(Filtered, Signal, Weights, radius);
				fast_range_filter(Signal, Filtered, Weights, radius);
			}
		}
		else
		{
			single_range_filter<unsigned short>(Signal, Filtered, Image, radius, sigma_distance, 65535);
		}
	}
	else if(mxGetClassID(input[1]) == mxSINGLE_CLASS)
	{
		MexImage<float> Image(input[1]);
		
		float minv = Image[0];
		float maxv = Image[0];
		for(long i=1; i<Image.layer_size*Image.layers; i++)
		{
			minv = (Image[i] < minv) ? Image[i] : minv;
			maxv = (Image[i] > minv) ? Image[i] : maxv;
		}
		
		if(repeat > 0)
		{
			MexImage<float> Weights(Signal.width, Signal.height, window/2);
			calculate_weights<float>(Image, Weights, radius, sigma_distance, maxv-minv);
			fast_range_filter(Signal, Filtered, Weights, radius);
			for(int r=0; r<=repeat/2; r++)
			{
				fast_range_filter(Filtered, Signal, Weights, radius);
				fast_range_filter(Signal, Filtered, Weights, radius);
			}
		}
		else
		{
			single_range_filter<float>(Signal, Filtered, Image, radius, sigma_distance, 255);
		}
	}	
	else
	{
		mexErrMsgTxt("Unsupported Image datatype"); 
	}
}