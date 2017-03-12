/** 
*	@file wuz_ordering.cpp
*	@date 07.08.2014
*	@author Sergey Smirnov
*/

#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}
#endif // __cplusplus
#include "../common/meximage.h"
#include <utility>
#include <algorithm>
using namespace mymex;

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")


void wuz_ordering(MexImage<float> &Cost, MexImage<float> &DispL, MexImage<float> &DispH, int mindisp)
{
	const int width = Cost.width;
	const int height = Cost.height;
	const int layers = Cost.layers;
	const long HW = width*height;

	#pragma omp parallel for
	for (long i = 0; i<HW; i++)
	{
		const int x = i / height;
		const int y = i % height;

		int first_best = Cost(x,y,0) < Cost(x,y,1) ? 0 : 1;
		int second_best = 1 - first_best;
		float first_value = Cost(x, y, first_best);
		float second_value = Cost(x, y, second_best);

		for(int l=2; l<layers; l++)
		{
			if(Cost(x, y, l) < first_value)
			{
				if(abs(first_best - l) > 5)
				{
					second_value = first_value;
					second_best = first_best;
				}
				first_value = Cost(x, y, l);
				first_best = l;
			}
			else if(Cost(x, y, l) < second_value && abs(first_best - l) > 5)
			{
				second_value = Cost(x, y, l);
				second_best = l;
			}
		}

		if(abs(first_best-second_best <= 1))
		{
			DispL(x,y) = first_best + mindisp;
			DispH(x,y) = first_best + mindisp;
		}
		else
		{
			DispL(x,y) = std::min<int>(first_best, second_best) + mindisp;
			DispH(x,y) = std::max<int>(first_best, second_best) + mindisp;
		}

	}
}

void mexFunction (int nout, mxArray* output[], int in, const mxArray* input[])
{
	if (nout != 2 || in < 1 || in > 4 || mxGetClassID(input[0]) != mxSINGLE_CLASS)
	{
		mexErrMsgTxt("USAGE: [DispL, DispH] = wuz_ordering(single(Cost), <mindisp_to_be_added>);"); 
    } 

	MexImage<float> Cost(input[0]);
	const mwSize depthdims[] = {Cost.height, Cost.width, 1};
	const int mindisp = (in > 1) ? (int) mxGetScalar(input[1]) : 0;

	output[0] = mxCreateNumericArray(3, depthdims, mxSINGLE_CLASS, mxREAL);
	output[1] = mxCreateNumericArray(3, depthdims, mxSINGLE_CLASS, mxREAL);
	
	MexImage<float> DispL(output[0]);
	MexImage<float> DispH(output[1]);

	wuz_ordering(Cost, DispL, DispH, mindisp);

}