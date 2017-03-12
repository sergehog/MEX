/** cross_bilateral
*	@file cross_bilateral.cpp
*	@date 2.12.2013
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


void mexFunction (int nout, mxArray* output[], int in, const mxArray* input[])
{
	omp_set_num_threads(std::max<int>(4, omp_get_max_threads()));
	omp_set_dynamic(true);
	
	if(in < 2 || in > 4 || nout != 1 || mxGetClassID(input[0]) != mxSINGLE_CLASS || mxGetClassID(input[1]) != mxSINGLE_CLASS)
	{
		mexErrMsgTxt("USAGE: [Filtered] = cross_bilateral(single(Signal), single(Image), [sigma_color, sigma_distance]);");
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
				
	const float sigma_color = (in > 3) ? (float)mxGetScalar(input[3]) : 10.f;
	const float sigma_distance = (in > 4) ? (float)mxGetScalar(input[4]) : 10.f;	

	const size_t dims[] = {(size_t)height, (size_t)width, (size_t)layers};	
	//const size_t dimsC[] = {(size_t)height, (size_t)width, (size_t)colors};	

	output[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL); 	
	//output[1] = mxCreateNumericArray(3, dimsC, mxSINGLE_CLASS, mxREAL); 	

	MexImage<float> Filtered(output[0]);
	//MexImage<float> Average(width, height, colors);

	MexImage<float> Weights(width, height, 1);
	Weights.setval(0.f);

	float *weights_table = new float[256*colors];
	for(int i=0; i<255*colors; i++)
	{
		weights_table[i] = exp(-i/(colors*sigma_distance));
	}	
	
	for(int x=0; x<width; x++)
	{
		for()
		{}
	}
}
