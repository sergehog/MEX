/** guided_image_filter
*	@file guided_image_filter.cpp
*	@date 23.04.2015
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
#define colors 3 

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")

//typedef unsigned char uint8;
using namespace mymex;

void calculateVarIxx(MexImage<float> &Guide, MexImage<float> &IntegralGuide, MexImage<float> & varTmp, MexImage<float> &varI, const int radius, const int a, const int b, const int varIndex)
{
	const int height = Guide.height;
	const int width = Guide.width;
	const long HW = Guide.layer_size;

	#pragma omp parallel for
	for (long i = 0; i < HW; i++)
	{
		varTmp[i] = Guide[i + a*HW] * Guide[i + b*HW];
	}
	varTmp.IntegralImage(true);
	#pragma omp parallel for
	for (long i = 0; i < HW; i++)
	{
		const int x = i / height;
		const int y = i % height;
		float meanIs[colors];
		float varAvg[1];
		IntegralGuide.getIntegralAverage(x, y, radius, meanIs);
		varTmp.getIntegralAverage(x, y, radius, varAvg);
		varI[varIndex] = varAvg[0] * varAvg[0] - meanIs[a] * meanIs[b];
	}
}

void mexFunction(int nout, mxArray* output[], int in, const mxArray* input[])
{
	omp_set_num_threads(std::max(2, omp_get_max_threads()));
	omp_set_dynamic(std::max(1, omp_get_max_threads() / 2));

	if (in != 4 || nout != 1 || mxGetClassID(input[0]) != mxSINGLE_CLASS || mxGetClassID(input[1]) != mxSINGLE_CLASS)
	{
		mexErrMsgTxt("USAGE: [Filtered] = guided_image_filter(single(Signal), single(Guide), radius, epsilon);");
	}

	MexImage<float> Signal(input[0]);
	MexImage<float> Guide(input[1]);

	const int height = Signal.height;
	const int width = Signal.width;
	const int layers = Signal.layers;
	//const int colors = Guide.layers;
	const long HW = Signal.layer_size;
	const float nan = sqrt(-1.f);
	if (Guide.width != width || Guide.height != height || Guide.layers != colors)
	{
		mexErrMsgTxt("Input images must have same resolution, and Guide must have 3 color channels!");
	}

	const int radius = (in > 2) ? std::max(0, (int)mxGetScalar(input[2])) : 3;
	const float epsilon = (in > 3) ? (float)mxGetScalar(input[3]) : 0.001;

	const size_t dims[] = { (size_t)height, (size_t)width, (size_t)layers };

	output[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);

	MexImage<float> Filtered(output[0]);
	MexImage<float> IntegralGuide(width, height, colors);
	MexImage<float> IntegralSignal(width, height, layers);
	MexImage<float> varI(width, height, colors * 2);
	MexImage<float> varTmp(width, height);
	Filtered.setval(0.f);
	IntegralGuide.IntegralFrom(Guide);
	IntegralSignal.IntegralFrom(Signal);
		
	calculateVarIxx(Guide, IntegralGuide, varTmp, varI, radius, 0, 0, 0); //varIrr
	calculateVarIxx(Guide, IntegralGuide, varTmp, varI, radius, 0, 1, 1); //varIrg
	calculateVarIxx(Guide, IntegralGuide, varTmp, varI, radius, 0, 2, 2); //varIrb
	calculateVarIxx(Guide, IntegralGuide, varTmp, varI, radius, 1, 1, 3); //varIgg
	calculateVarIxx(Guide, IntegralGuide, varTmp, varI, radius, 1, 2, 4); //varIgb
	calculateVarIxx(Guide, IntegralGuide, varTmp, varI, radius, 2, 2, 5); //varIbb


	#pragma omp parallel 
	{
		MexImage<float> meanP(width, height);
		MexImage<float> corrIP(width, height, colors);
		MexImage<float> covIP(Signal, colors);

		#pragma omp for
		for (int c = 0; c < layers; c++)
		{
			MexImage<float> A(Signal, c);

			meanP.IntegralFrom(A);
			#pragma parallel for
			for (long i = 0; i < HW; i++)
			{
				corrIP[i] = A[i] * Guide[i];
				corrIP[i + HW] = A[i] * Guide[i + HW];
				corrIP[i + HW * 2] = A[i] * Guide[i + HW * 2];
			}
			corrIP.IntegralImage(true);
			#pragma parallel for
			for (long i = 0; i < HW; i++)
			{
				const int x = i / height;
				const int y = i % height;
				covIP
			}
			
		}
	}
	
	for (int c = 0; c<layers; c++)
	{
		const long cHW = c*HW;
		float minvalue = Signal[cHW];
		float maxvalue = Signal[cHW];


		for (long i = 0; i<HW; i++)
		{
			minvalue = Signal[i + cHW] < minvalue ? Signal[i + cHW] : minvalue;
			maxvalue = Signal[i + cHW] > maxvalue ? Signal[i + cHW] : maxvalue;
			Integral[i] = _isnan(Signal[i + cHW]) ? 0.f : Signal[i + cHW];
			Pixels[i] = _isnan(Signal[i + cHW]) ? 0.f : 1.f;
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
		for (long i = 0; i<HW; i++)
		{
			int y = i % height;
			int x = i / height;

			float value = Integral.getIntegralSum(x, y, radius);
			float pixels = Pixels.getIntegralSum(x, y, radius);
			if (pixels > 0.f)
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
