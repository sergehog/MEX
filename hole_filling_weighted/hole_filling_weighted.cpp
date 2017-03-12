/** hole_filling_weighted
*	@file hole_filling_weighted.cpp
*	@date 28.03.2013
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
//typedef size_t matlab_size;
#define isnan _isnan

void mexFunction (int nout, mxArray* output[], int in, const mxArray* input[])
{	
	omp_set_num_threads(std::max(4, omp_get_num_threads()/2));
	omp_set_dynamic(false);

	if(in < 2 || in > 6 || nout != 2 || mxGetClassID(input[0]) != mxSINGLE_CLASS || mxGetClassID(input[1]) != mxSINGLE_CLASS )
	{
		mexPrintf("USAGE: [Signal, Weights_out] = hole_filling_weighted(Signal_with_NaNs, Weights, <radius, direction, maxdisp>);\n");
		mexPrintf("Image - SINGLE-valued (possibly color) image with NaN values, threated as holes \n");
		mexPrintf("Weights - SINGLE-valued 1-layer image with NaN values, threated as holes \n");
		mexPrintf("radius - radius of processing window (default: 2)\n");
		mexPrintf("direction - 0 (default): no direction, 1: left-to-right, -1: right-to-left \n");		
		mexPrintf("maxdisp - if direction != 0, process last N roows in opposite direction (def: 0) \n");		
		mexErrMsgTxt("Wrong input parameters!");
    }

	MexImage<float> Signal(input[0]);
	MexImage<float> Weights(input[1]);
	const int height = Signal.height;
	const int width = Signal.width;
	const int layers = Signal.layers;
	const long HW = Signal.layer_size;	
	const float nan = sqrt(-1.f);	
		
	if(Weights.width != width || Weights.height != height || Weights.layers != 1)
	{
		mexErrMsgTxt("Wrong input parameters!");
	}

	const int radius = (in > 2) ? (int)mxGetScalar(input[2]) : 3;	
	const int direction = (in > 3) ? (int)mxGetScalar(input[3]) : 0; 	
	const int offset = (in > 4) ? std::abs((int)mxGetScalar(input[4])) : 0;	
	
	matlab_size dimsI[] = {(matlab_size)height, (matlab_size)width, (matlab_size)layers};
	matlab_size dims[] = {(matlab_size)height, (matlab_size)width, 1};

	output[0] = mxCreateNumericArray(3, dimsI, mxSINGLE_CLASS, mxREAL);
	output[1] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);

	MexImage<float> SignalOut(output[0]);
	MexImage<float> WeightsOut(output[1]);


	MexImage<float> IntegralSignal(width, height, layers);
	MexImage<float> IntegralWeights(width, height, 1);
	MexImage<bool> Valid(width, height, 1);
	
	
	#pragma omp parallel for
	for(long i=0; i<HW; i++)
	{
		WeightsOut.data[i] = isnan(Signal[i]) || isnan(Weights[i]) || Weights[i] < 0 ? 0.f : (Weights[i] > 1 ? 1.f : Weights[i]);
	}

	unsigned long holes = 0;

	#pragma omp parallel for reduction(+:holes)
	for(long i=0; i<HW; i++)
	{
		if(isnan(Signal[i]))
		{
			for(int c=0; c<layers; c++)
			{
				SignalOut.data[i+c*HW] = 0;
			}
			holes ++;
			Valid.data[i] = false;
		}
		else
		{
			for(int c=0; c<layers; c++)
			{
				SignalOut.data[i+c*HW] = Signal[i+c*HW]*WeightsOut[i];
			}
			Valid.data[i] = true;
		}
	}

	unsigned long updated = 1;
	while(holes > 0 && updated > 0)
	{
		updated = 0;

		#pragma omp parallel for
		for(long i=0; i<HW*layers; i++)
		{
			IntegralSignal.data[i] = SignalOut[i];
		}

		#pragma omp parallel for
		for(long i=0; i<HW; i++)
		{
			IntegralWeights.data[i] = WeightsOut[i];
		}

		IntegralSignal.IntegralImage(true);
		IntegralWeights.IntegralImage(true);

		#pragma omp parallel for reduction(-:holes)
		for(long i=0; i<HW; i++)
		{
			if(!Valid[i])
			{
				const int x = i / heig.ht;
				const int y = i % height;

				float weight = IntegralWeights.getIntegralSum(x,y,radius);

				if(weight > 0)
				{
					for(int c=0; c<layers; c++)
					{
						SignalOut.data[i + c*HW] = IntegralSignal.getIntegralSum(x,y,radius,c)/weight;
					}
					
					Valid.data[i] = true;
					//WeightsOut.data[i] = IntegralWeights.getIntegralAverage(x,y,radius);
					WeightsOut.data[i] = 1.f;
					updated ++;
					holes --;
				}
				/*else
				{
					WeightsOut.data[i] = 0.f;
				}*/
			}
		}
	}

	#pragma omp parallel for 
	for(long i=0; i<HW; i++)
	{
		if(!isnan(Signal[i]))
		{
			for(int c=0; c<layers; c++)
			{
				SignalOut.data[i + c*HW] = Signal[i + c*HW];
			}
		}
	}

	//#pragma omp parallel for 
	//	for(long i=0; i<HW; i++)
	//	{
	//		if(!isnan(Signal[i]))
	//		{
	//			continue;
	//		}

	//		if(!Valid[i] && WeightsOut[i] > 0.001)
	//		{
	//			for(int c=0; c<layers; c++)
	//			{
	//				SignalOut.data[i + c*HW] /= WeightsOut[i];
	//			}
	//			
	//			holes --;
	//			updated ++;
	//		}
	//		else
	//		{
	//			for(int c=0; c<layers; c++)
	//			{
	//				SignalOut.data[i + c*HW] = nan;
	//			}
	//		}
	//	}
	
	
}
