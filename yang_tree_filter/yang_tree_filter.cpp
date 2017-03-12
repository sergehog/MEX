//q_yang_filter
/** 
*	@file q_yang_filter.cpp
*	@date 10.06.2013
*	@author Sergey Smirnov <sergey.smirnov@tut.fi>
*/

#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}
#endif // __cplusplus

#include "../common/meximage.h"
#include "qx_tree_filter.h"
//#include "../common/defines.h"
#include <vector>
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
	
	if(in < 3 || in > 4 || nout != 1 || mxGetClassID(input[0]) != mxDOUBLE_CLASS || mxGetClassID(input[1]) != mxUINT8_CLASS)
	{
		mexErrMsgTxt("USAGE: [Filtered] = q_yang_filter(double(Signal), uint8(Image), sigma);");
    }

	MexImage<double> Signal(input[0]);	
	MexImage<uint8> Image(input[1]);
	
	const int height = Image.height;
	const int width = Image.width;
	const int colors = Image.layers;
	const int signal_layers = Signal.layers;
	const long HW = Image.layer_size;
	//const float nan = sqrt(-1.f);
	//const int r = 1;

	const float sigma = (in > 2) ? (float)mxGetScalar(input[2]) : 0.1;

	if(Signal.height != height || Signal.width != width)
	{
		mexErrMsgTxt("Edges and Signal must have the same width and height.");
	}
			
	unsigned char* texture = new unsigned  char[HW*colors];
	double * cost = new double[HW*signal_layers];
	double * cost_out = new double[HW*signal_layers];

	for(int x=0; x<width; x++)
	{
		for(int y=0; y<height; y++)
		{
			long i = Image.Index(x,y);
			long t = y*width + x;
			for(int c=0; c<colors; c++)
			{
				texture[t*colors + c] = Image[i+c*HW];
			}

			for(int c=0; c<signal_layers; c++)
			{
				cost[t*signal_layers + c] = Signal[i+c*HW];
			}
		}
	}	
	
	qx_tree_filter m_tf;
	//m_tf.init(height,width,colors,sigma,4);
	m_tf.init(height,width,colors,sigma,8);
	m_tf.build_tree(texture);
	m_tf.filter(cost, cost_out, signal_layers);
	m_tf.clean();
	delete[] texture;
	delete[] cost;

	const size_t dims[] = {(size_t)height, (size_t)width, (size_t)signal_layers};		
	output[0] = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL); 	
	MexImage<double> Filtered(output[0]);		
	
	for(int x=0; x<width; x++)
	{
		for(int y=0; y<height; y++)
		{
			long i = Image.Index(x,y);
			long t = y*width + x;			

			for(int c=0; c<signal_layers; c++)
			{
				Filtered.data[i+c*HW] = cost_out[t*signal_layers + c];
			}
		}
	}	
	
	delete[] cost_out;
}