/** MEX implementation of Bilateral filter
*	@file bilateral_filter.cpp
*	@date 01.01.2008
*	@author Sergey Smirnov
*/

#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}
#endif // __cplusplus

#include "../common/meximage.h"
#include <cmath>
#include <algorithm>
#ifndef _DEBUG
#include <omp.h>
#endif

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")

#define MAXDIFF 1024

using namespace mymex;

// pre-calculation of distance weights
template<typename TS>
void init_distance(TS* const distance, const double sigma_distance, const int radius)
{
	//#pragma omp parallel for
	for(int i=-radius, index = 0; i<=radius; i++)
	{
		for(int j=-radius; j<=radius; j++, index++)
		{
			distance[index] = (TS)exp(-sqrt((double)i*i+j*j)/sigma_distance);
		}
	}
}

// pre-calculation of color weights
template<typename TS>
void init_colors(TS * const distance, const double sigma_color, const int colors, const int maxdiff)
{
	for(int i=0; i<maxdiff*colors; i++)
	{
		distance[i] = (TS)exp(-double(i)/(colors*sigma_color));
	}
}


template<typename TS, typename TI>
void bilateral_filter(MexImage<TS> &Signal, MexImage<TI> &Image, MexImage<TS> &Filtered, const double sigma_color, const float sigma_distance, const int radius) 
{
	const int width = Image.width;
	const int height = Image.height;
	const int colors = Image.layers;
	const int layers = Signal.layers;
	const long HW = Image.layer_size;
	const int diameter = radius*2+1;
	const int window = diameter*diameter;

	// pre-calculation of look-up tables
	TS *color_table = new TS[MAXDIFF*colors];
	TS *distance_table = new TS[window];
	
	init_colors<TS>(color_table, sigma_color, colors, MAXDIFF);
	init_distance<TS>(distance_table, sigma_distance, radius);

	//main filtering cycle	
	#pragma omp parallel
	{
		TS * const weights = new TS[window];		

		#pragma omp for
		for(long index=0; index<HW; index++)
		{
			const int y = index % height;
			const int x = index / height;

			
			//int offset = INDEX(x,y,height);
			TS weights = 1.f;
				
#pragma omp parallel for reduction(+:weights)
			for (int i = 0; i < window; i++)
			{
				const int dy = i % diameter - radius;
				const int dx = i / diameter - radius;

				const int xx = x + dx;
				const int yy = y + dy;

				if ((dx == 0 && dy == 0) || xx < 0 || xx >= width || yy < 0 || yy >= height)
				{
					continue;
				}

				TS weight = 0;
				for (int c = 0; c < colors; c++)
				{
					//float diff = Image(x,y,c)-Image(xx,yy,c);
					//weight += diff*diff;
					weight += abs(TS(Image(x, y, c)) - TS(Image(xx, yy, c)));
				}
				weight = color_table[int(mymex::round(weight))] * distance_table[i];
				weights += weight;

				for (int l = 0; l < layers; l++)
				{
					Filtered(x, y, l) += Signal(xx, yy, l) * weight;
				}
			}
			
			for(int l=0; l<layers; l++)
			{				
				Filtered(x,y,l) /= weights;
			}
		}

		delete[] weights;
	}

	delete[] color_table;
	delete[] distance_table;


}

void mexFunction (int nout, mxArray* output[], int in, const mxArray* input[])
{
#ifndef _DEBUG
	omp_set_num_threads(std::max(2,omp_get_max_threads()));
	omp_set_dynamic(std::max(1,omp_get_max_threads())/2);
#endif

	if(in < 2 || in > 5 || nout != 1)
	{
		mexErrMsgTxt("USAGE: [Result] = bilateral_filter(Signal, Edges <, sigma_color, sigma_distance, radius>);"); 
    } 
	
	const double sigma_color = (in > 2) ? (float)mxGetScalar(input[2]) : 2;	
	const double sigma_distance = (in > 3) ? (float)mxGetScalar(input[3]) : 2;	
	const int radius = (in > 4) ? (unsigned)mxGetScalar(input[4]) : int(sigma_distance*2);	


	if(mxGetClassID(input[0]) == mxSINGLE_CLASS)
	{
		MexImage<float> Signal(input[0]);
		const size_t dims[] = {(size_t)Signal.height, (size_t)Signal.width, Signal.layers};
		output[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);		
		MexImage<float> Filtered(output[0]);	
		for(long i=0; i<Signal.layer_size*Signal.layers; i++)
		{
			Filtered[i] = Signal[i];
		}

		if(mxGetClassID(input[1]) == mxUINT8_CLASS)
		{
			MexImage<unsigned char> Image(input[1]);
			if(Signal.height != Image.height|| Signal.width != Image.width)
			{
				mexErrMsgTxt("Width and Height of Input Images must coincide"); 
			}
			bilateral_filter<float, unsigned char>(Signal, Image, Filtered, sigma_color, sigma_distance, radius);
		}
		else if(mxGetClassID(input[1]) == mxUINT16_CLASS)
		{
			MexImage<unsigned short> Image(input[1]);
			if(Signal.height != Image.height|| Signal.width != Image.width)
			{
				mexErrMsgTxt("Width and Height of Input Images must coincide"); 
			}
			bilateral_filter<float, unsigned short>(Signal, Image, Filtered, sigma_color, sigma_distance, radius);
		}
		else if(mxGetClassID(input[1]) == mxSINGLE_CLASS)
		{
			MexImage<float> Image(input[1]);
			if(Signal.height != Image.height|| Signal.width != Image.width)
			{
				mexErrMsgTxt("Width and Height of Input Images must coincide"); 
			}
			bilateral_filter<float, float>(Signal, Image, Filtered, sigma_color, sigma_distance, radius);
		}
		else
		{
			mexErrMsgTxt("Unsupported configuratuion of Image/Signal datatypes"); 
		}
	}
	else if(mxGetClassID(input[0]) == mxDOUBLE_CLASS)
	{
		MexImage<double> Signal(input[0]);
		const size_t dims[] = {(size_t)Signal.height, (size_t)Signal.width, Signal.layers};
		output[0] = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);		
		MexImage<double> Filtered(output[0]);	
		for(long i=0; i<Signal.layer_size*Signal.layers; i++)
		{
			Filtered[i] = Signal[i];
		}
		if(mxGetClassID(input[1]) == mxUINT8_CLASS)
		{
			MexImage<unsigned char> Image(input[1]);
			if(Signal.height != Image.height|| Signal.width != Image.width)
			{
				mexErrMsgTxt("Width and Height of Input Images must coincide"); 
			}
			bilateral_filter<double, unsigned char>(Signal, Image, Filtered, sigma_color, sigma_distance, radius);
		}
		else if(mxGetClassID(input[1]) == mxUINT16_CLASS)
		{
			MexImage<unsigned short> Image(input[1]);
			if(Signal.height != Image.height|| Signal.width != Image.width)
			{
				mexErrMsgTxt("Width and Height of Input Images must coincide"); 
			}
			bilateral_filter<double, unsigned short>(Signal, Image, Filtered, sigma_color, sigma_distance, radius);
		}
		else if(mxGetClassID(input[1]) == mxDOUBLE_CLASS)
		{
			MexImage<double> Image(input[1]);
			if(Signal.height != Image.height|| Signal.width != Image.width)
			{
				mexErrMsgTxt("Width and Height of Input Images must coincide"); 
			}
			bilateral_filter<double, double>(Signal, Image, Filtered, sigma_color, sigma_distance, radius);
		}
		else
		{
			mexErrMsgTxt("Unsupported configuratuion of Image/Signal datatypes"); 
		}

	}
	else
	{
		mexErrMsgTxt("Unsupported configuratuion of Image/Signal datatypes"); 
	}
	
}