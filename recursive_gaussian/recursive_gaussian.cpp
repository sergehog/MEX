/** 
* @file recursive_gaussian.cpp
* @author Sergey Smirnov <sergey.smirnov@tut.fi>
* @date 29.10.2014
* @copyright 3D Media Group / Tampere University of Technology
*/
 


#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}
#endif // __cplusplus

#include "../common/meximage.h"
//#include "../common/defines.h"
#include <vector>
#define _USE_MATH_DEFINES
#include <cmath>
#include <algorithm>
#include <memory>
#ifndef _DEBUG
#include <omp.h>
#endif

//#define M_PI       3.14159265358979323846

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")

//typedef unsigned char uint8;
using namespace mymex;


void mexFunction(int nout, mxArray* output[], int in, const mxArray* input[])
{
#ifndef _DEBUG
	omp_set_num_threads(std::max<int>(4, omp_get_max_threads()));
	omp_set_dynamic(std::max<int>(2, omp_get_max_threads() / 2));
#endif

	//if (in != 2 || nout < 1 || nout > 2 || mxGetClassID(input[0]) != mxSINGLE_CLASS)
	if (in != 2 || nout != 1 || mxGetClassID(input[0]) != mxSINGLE_CLASS)
	{
		//mexErrMsgTxt("USAGE: [Filtered, <Normalization>] = recursive_gaussian(single(Signal), alpha);");
		mexErrMsgTxt("USAGE: [Filtered] = recursive_gaussian(single(Signal), alpha);");
	}
		
	const float a = (in > 1) ? std::max<float>(0.f, std::min<float>(1.f, mxGetScalar(input[1]))) : 0.8;	
	MexImage<float> Signal(input[0]);

	const int width = Signal.width;
	const int height = Signal.height;
	const int layers = Signal.layers;
	const long HW = Signal.layer_size;

	const size_t dims[] = { (size_t)height, (size_t)width, (size_t)layers };
	output[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
	MexImage<float> Filtered(output[0]);
	MexImage<float> Temporal(width, height, layers);
	//if (nout > 1)
	//{
	//	const size_t dims2[] = { (size_t)height, (size_t)width, 1 };
	//	output[1] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
	//	MexImage<float> Normalization(output[1]);
	//}
	
	Temporal.setval(0.f);
	#pragma omp parallel 
	{
		std::unique_ptr<float[]> y_1(new float[layers]);
		std::unique_ptr<float[]> y_2(new float[layers]);

		std::unique_ptr<float[]> ry_1(new float[layers]);
		std::unique_ptr<float[]> ry_2(new float[layers]);
		
		#pragma omp for
		for(int x=0; x<width; x++)
		{
			for(int l=0; l<layers; l++)
			{				
				y_2[l] = Signal(x, 0, l);
				y_1[l] = Signal(x, 1, l) + a * y_2[l];				
				Temporal(x, 0, l) += y_2[l];
				Temporal(x, 1, l) += y_1[l];

				ry_2[l] = Signal(x, height-1, l);
				ry_1[l] = Signal(x, height-2, l) + a * ry_2[l];				
				Temporal(x, height-1, l) += ry_2[l];
				Temporal(x, height-2, l) += ry_1[l];
			}

			for(int y=2; y<height; y++)
			{
				for(int l=0; l<layers; l++)
				{
					float y_0 = Signal(x, y, l) +  a*y_1[l];
					Temporal(x, y, l) += y_0;
					y_2[l] = y_1[l];
					y_1[l] = y_0;

					float ry_0 = Signal(x, height-y-1, l) +  a*ry_1[l];
					Temporal(x, height-y-1, l) += ry_0;
					ry_2[l] = ry_1[l];
					ry_1[l] = ry_0;
				}
			}
		}
	}
	
	#pragma omp parallel for
	for(long i=0; i<HW*layers; i++)
	{
		Temporal[i] -= Signal[i];
	}

	#pragma omp parallel 
	{
		std::unique_ptr<float[]> y_1 (new float[layers]);
		std::unique_ptr<float[]> y_2 (new float[layers]);

		std::unique_ptr<float[]> ry_1 (new float[layers]);
		std::unique_ptr<float[]> ry_2 (new float[layers]);
		
		#pragma omp for
		for(int y=0; y<height; y++)		
		{
			for(int l=0; l<layers; l++)
			{				
				y_2[l] = Temporal(0, y, l);
				y_1[l] = Temporal(1, y, l) + a * y_2[l];				
				Filtered(0, y, l) += y_2[l];
				Filtered(1, y, l) += y_1[l];

				ry_2[l] = Temporal(width-1, y, l);
				ry_1[l] = Temporal(width-2, y, l) + a * ry_2[l];				
				Filtered(width-1, y, l) += ry_2[l];
				Filtered(width-2, y, l) += ry_1[l];
			}

			for(int x=2; x<width; x++)
			{
				for(int l=0; l<layers; l++)
				{
					float y_0 = Temporal(x, y, l) +  a*y_1[l];
					Filtered(x, y, l) += y_0;
					y_2[l] = y_1[l];
					y_1[l] = y_0;

					float ry_0 = Temporal(width-x-1, y, l) +  a*ry_1[l];
					Filtered(width-x-1, y, l) += ry_0;
					ry_2[l] = ry_1[l];
					ry_1[l] = ry_0;
				}
			}
		}
	}

	#pragma omp parallel for
	for(long i=0; i<HW*layers; i++)
	{
		Filtered[i] -= Temporal[i];
	}	

}