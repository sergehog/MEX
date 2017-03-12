/** fast binocular matching 
*	@file stereo_box.cpp
*	@date 10.03.2010
*	@author Sergey Smirnov <sergey.smirnov@tut.fi>
*/

#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}
#endif // __cplusplus

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")


#include "../common/defines.h"
#include "../common/common.h"
#include "../common/matching.h"
#include <math.h>
#include <algorithm>
#include <utility>
#include "omp.h"

#define RADIUS 3
#define MAXRADIUS 20
#define COST_THRESHOLD 100

void mexFunction (int nout, mxArray* output[], int in, const mxArray* input[])
{
	omp_set_dynamic(0);
	omp_set_num_threads(std::max(2, omp_get_num_threads()));

	if(in < 4 || in > 6 || nout != 4 || mxGetClassID(input[0])!=mxUINT8_CLASS || mxGetClassID(input[1])!=mxUINT8_CLASS)
	{
		mexErrMsgTxt("USAGE: [DispLR, DispRL, ConfLR, ConfRL] = stereo_box(uint8(Left), uint8(Right), mindisp, maxdisp[, radius, cost_threshold]);");
    }

	const unsigned dims = (unsigned)mxGetNumberOfDimensions(input[0]);
	const int height = (int)(mxGetDimensions(input[0]))[0];
	const int width = (int)(mxGetDimensions(input[0]))[1];
	const int HW = width*height;
	const int HW2 = HW*2;
	UINT8 *Left = (UINT8*)mxGetData(input[0]);
	UINT8 *Right = (UINT8*)mxGetData(input[1]);

	if((mxGetDimensions(input[1]))[0] != height || (mxGetDimensions(input[1]))[1] != width)
	{
		mexErrMsgTxt("All the images must be the same size!");
	}

	const int mindisp = static_cast<int>(mxGetScalar(input[2]));
	const int maxdisp = static_cast<int>(mxGetScalar(input[3]));
	const int layers = maxdisp - mindisp + 1;

	if(maxdisp <= mindisp)
	{
		mexErrMsgTxt("maxdisp must be larger than mindisp");
	}

	const int radius = (in > 4) ? MIN((int)mxGetScalar(input[4]),MAXRADIUS) : RADIUS;
	static float cost_threshold = (in > 5) ? (float)mxGetScalar(input[5]) : COST_THRESHOLD;

    int diameter = DIAMETER(radius);
    int window = WINDOW(radius);
	int window2 = window*2;

	//const unsigned costdims[] = {height, width, maxdisp};
	const mwSize depthdims[] = {height, width, 1};

	output[0] = mxCreateNumericArray(3, depthdims, mxSINGLE_CLASS, mxREAL); 
	output[1] = mxCreateNumericArray(3, depthdims, mxSINGLE_CLASS, mxREAL); //
	output[2] = mxCreateNumericArray(3, depthdims, mxSINGLE_CLASS, mxREAL); //
	output[3] = mxCreateNumericArray(3, depthdims, mxSINGLE_CLASS, mxREAL); //

	float* DispLR = (float*) mxGetData(output[0]);
	float* DispRL = (float*) mxGetData(output[1]);
	float* ConfLR = (float*) mxGetData(output[2]);
	float* ConfRL = (float*) mxGetData(output[3]);

	#pragma omp parallel
	{
		float *CostLR = new float[layers]; //[MAXDISP];
		float *CostRL = new float[layers]; //[MAXDISP];

		#pragma omp for
		for(int y=0; y<height; y++)
		{
			for(int x=0; x<width; x++)
			{
				int index = INDEX(x,y,height);

				for(int d=0; d<layers; d++)
				{
					int disp = d + mindisp;
					CostLR[d] = calculateSAD_replicated(Left, Right, x, y, x-disp, y, radius, width, height, cost_threshold);
					CostRL[d] = calculateSAD_replicated(Right, Left, x, y, x+disp, y, radius, width, height, cost_threshold);
				}

				std::pair<float, float> dl = winner_takes_all(CostLR, layers, cost_threshold);
				std::pair<float, float> dr = winner_takes_all(CostRL, layers, cost_threshold);
				//pair<float, float> cc = winner_takes_all(CostC, maxdisp);

				DispLR[index] = dl.first + mindisp;
				DispRL[index] = dr.first + mindisp;

				ConfLR[index] = dl.second;
				ConfRL[index] = dr.second;
			}

		}
		

		delete CostLR;
		delete CostRL;
	}

}
