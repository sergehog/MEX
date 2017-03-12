/** fast_average
*	@file fast_average.cpp
*	@date 3.08.2012
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

//typedef unsigned char uint8;
using namespace mymex;


void mexFunction (int nout, mxArray* output[], int in, const mxArray* input[])
{
	omp_set_num_threads(std::max(2, omp_get_max_threads()));
	omp_set_dynamic(std::max(1, omp_get_max_threads()/2));
	
	if(in < 1 || in > 2 || nout != 1 || mxGetClassID(input[0])!=mxSINGLE_CLASS)
	{
		mexErrMsgTxt("USAGE: [Averaged] = fast_average(single(Signal_with_NaNs), radius);");
    }

	MexImage<float> Signal(input[0]);	
	
	const int height = Signal.height;
	const int width = Signal.width;	
	const int layers = Signal.layers;
	const long HW = Signal.layer_size;
	const float nan = sqrt(-1.f);
			
	const int radius = (in > 1) ? std::max(0, (int)mxGetScalar(input[1])) : 3;

	const size_t dims[] = {(size_t)height, (size_t)width, (size_t)layers};	

	output[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL); 	

	MexImage<float> Filtered(output[0]);
	MexImage<float> Integral(width, height);
	MexImage<float> Pixels(width, height);

	for(int c=0; c<layers; c++)
	{
		const long cHW = c*HW;
		float minvalue = Signal[cHW];
		float maxvalue = Signal[cHW];

		#pragma omp parallel for
		for(long i=0; i<HW; i++)
		{
			minvalue = Signal[i + cHW] < minvalue ? Signal[i + cHW] : minvalue;
			maxvalue = Signal[i + cHW] > maxvalue ? Signal[i + cHW] : maxvalue;
			Integral[i] = _isnan(Signal[i+cHW]) ? 0.f : Signal[i+cHW];
			Pixels[i] = _isnan(Signal[i+cHW]) ? 0.f : 1.f;
		}
		#pragma omp parallel sections
		{
			#pragma omp section
			{
				Integral.IntegralImage(true);
			}
			#pragma omp section
			{
				Pixels.IntegralImage(true);
			}
		}		
		
		#pragma omp parallel for
		for (long i = 0; i < HW; i++)
		{
			const int y = i % height;
			const int x = i / height;
			
			const float value = Integral.getIntegralSum(x, y, radius);
			const float pixels = Pixels.getIntegralSum(x, y, radius);
			if(pixels > 0.f)
			{
				Filtered[i + cHW] = std::min(maxvalue, std::max(minvalue, value / pixels));
			}
			else
			{
				Filtered[i + cHW] = nan;
			}
		}
	}
	
}