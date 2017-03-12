/** Inverts inpainted gradient for a depth maps
*	@file inverse_gradient.cpp
*	@date 06.09.2013
*	@author Sergey Smirnov <sergey.smirnov@tut.fi>
*/

#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}
#endif // __cplusplus

#include "../common/meximage.h"
#include <float.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <omp.h>

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")

using namespace mymex;
using namespace std;

#define isnan _isnan
struct contour
{	
	unsigned char c_left : 1;
	unsigned char c_right : 1;
	unsigned char c_up : 1;
	unsigned char c_down : 1;
	
	contour()
	{
		c_left = c_right = c_up = c_down = 0;
	}

	contour(int value)
	{		
		c_left = value & 1;
		c_right = value & 2;
		c_up = value & 4;
		c_down = value & 8;
	}
	
	operator int()
	{
		return c_left + c_right*2 + c_up*4 + (int)c_down*8;
	}
};

void mexFunction (int nout, mxArray* output[], int in, const mxArray* input[])
{	
	omp_set_num_threads(std::max(4, omp_get_num_threads())/2);
	omp_set_dynamic(0);

	if(in < 2 || in > 4 || nout != 1 || mxGetClassID(input[0]) != mxSINGLE_CLASS || mxGetClassID(input[1]) != mxSINGLE_CLASS || mxGetClassID(input[2]) != mxSINGLE_CLASS )
	{
		mexPrintf("USAGE: [Depth] = inverse_gradient(Dx, Dy, DepthIn);\n");
		mexPrintf("Dx - SINGLE-valued without NaN values \n");
		mexPrintf("Dy - SINGLE-valued without NaN values \n");
		mexPrintf("DepthIn - SINGLE-valued image of same size, with NaN \n");		
		
		mexErrMsgTxt("Wrong input parameters!");
    }

	MexImage<float> Dx(input[0]);
	MexImage<float> Dy(input[1]);
	MexImage<float> DepthIn(input[2]);
	
	const int height = DepthIn.height;
	const int width = DepthIn.width;	
	const long HW = DepthIn.layer_size;	
	const float nan = sqrt(-1.f);	
			
	if(Dx.width != width || Dx.height != height || Dy.width != width || Dy.height != height)
	{
		mexErrMsgTxt("Input images are not of the same size!");
	}

	
	size_t dims[] = {(size_t)height, (size_t)width, 1};
	output[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
	
	MexImage<float> Depth(output[0]);
	MexImage<contour> Contour(width, height);
	
	for(long i=0; i<HW; i++)
	{
		Depth.data[i] = DepthIn[i];
	}

	int iter = 0;
	int contour_found = 1;
	while(contour_found > 0 && iter < 1000)
	{
		iter ++;
		contour_found = 0;
		Contour.setval(0);

		#pragma omp parallel for reduction(+:contour_found)
		for(long i=0; i<HW; i++)
		{
			if(!isnan(Depth[i]))
				continue;

			int x = i / height;
			int y = i % height;
								
			// check if it's contour 
			contour value;
			value.c_left = (x > 0) ? !isnan(Depth[Depth.Index(x-1, y)]) : false;		
			value.c_right = (x < width-1) ? !isnan(Depth[Depth.Index(x+1, y)]) : false;
			value.c_up = (y > 0) ? !isnan(Depth[Depth.Index(x, y-1)]) : false;		
			value.c_down = (y < height-1) ? !isnan(Depth[Depth.Index(x, y+1)]) : false;
		
			Contour.data[i] = value;
			contour_found ++;
		}


		if(contour_found < 1)
			break;

		#pragma omp parallel for
		for(long i=0; i<HW; i++)
		{
			if(!int(Contour[i]))
				continue;

			int x = i / height;
			int y = i % height;
					
			int norm = 0;
			float value = 0;
			if(Contour[i].c_left)
			{
				value += (Depth[Depth.Index(x-1, y)] + Dx[i]/2);
				norm ++;
			}
			if(Contour[i].c_right)
			{
				value += Depth[Depth.Index(x+1, y)] - Dx[i]/2;
				norm ++;
			}
			if(Contour[i].c_up)
			{
				value += Depth[Depth.Index(x, y-1)] + Dy[i]/2;
				norm ++;
			}
			if(Contour[i].c_down)
			{
				value += Depth[Depth.Index(x, y+1)] - Dy[i]/2;
				norm ++;
			}			
			Depth.data[i] = value/norm;
		}		
	}
}