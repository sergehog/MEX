/** Computes 3D Cost Volume for 2 (rectified) images. Uses SAD dissimilarity metric
* @author Sergey Smirnov
*/
#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}
#endif // __cplusplus
#include "../common/common.h"
#include <math.h>

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")
#pragma warning(disable:4244)
#pragma warning(disable:4018)

#define MAXCost 200.f


void mexFunction (int nout, mxArray* output[], int in, const mxArray* input[])
{
	if(in < 3 || in > 5 || nout < 1 || nout > 2 || mxGetClassID(input[0])!=mxUINT8_CLASS || mxGetClassID(input[1])!=mxUINT8_CLASS)
	{
		mexErrMsgTxt("USAGE: [CostL, <CostR>] = compute_cost(UINT8(Left), UINT8(Right), maxdisp, [mindisp = 0, cost_threshold]);"); 
    } 
	const unsigned dims = (unsigned)mxGetNumberOfDimensions(input[0]);
	const int height = (unsigned)(mxGetDimensions(input[0]))[0];
	const int width = (unsigned)(mxGetDimensions(input[0]))[1];
	const int HW = width*height;
    const int HW2 = HW*2;

    if(height!=(mxGetDimensions(input[1]))[0] || width!=(mxGetDimensions(input[1]))[1])
	{
		mexErrMsgTxt("ERROR: Sizes of parameters 'left', 'right' must be the same."); 
	}	
	if(mxGetClassID(input[0])!=mxUINT8_CLASS || mxGetClassID(input[1])!=mxUINT8_CLASS)
	{
		mexErrMsgTxt("ERROR: Only UINT8 type is allowed for input parameters: 'left', 'right'."); 
	}

	const UINT8 *Left = (UINT8*)mxGetData(input[0]);
	const UINT8 *Right = (UINT8*)mxGetData(input[1]);
	const int maxdisp = static_cast<int>(mxGetScalar(input[2]));	
	const int mindisp = (in > 3) ?  static_cast<int>(mxGetScalar(input[3])) : 0;	
	const float cost_threshold = (in > 4) ?  static_cast<float>(mxGetScalar(input[4])) : MAXCost;	
	const int dispLayers = static_cast<unsigned>(MAX(maxdisp - mindisp, 1));
	const unsigned depthcost[] = {(unsigned)height, (unsigned)width, (unsigned)dispLayers+1};

	//Matlab-allocated variables
	output[0] = mxCreateNumericArray(3, depthcost, mxSINGLE_CLASS, mxREAL);
	float *CostL = (SINGLE*) mxGetData(output[0]);
    float *CostR = NULL;
    if(nout > 1)
    {
        output[1] = mxCreateNumericArray(3, depthcost, mxSINGLE_CLASS, mxREAL);	
        CostR = (float*) mxGetData(output[1]);
    }
	
	if(nout > 1)
    {
		for(int i=0; i<HW*(dispLayers+1); i++)
		{
			CostL[i] = cost_threshold;
			CostR[i] = cost_threshold;
		}	
    }
	else
	{
		for(int i=0; i<HW*(dispLayers+1); i++)
		{
			CostL[i] = cost_threshold;
		}
	}


    for(int x=-maxdisp; x<width+maxdisp; x++)
	{
        for(int y=0; y<height; y++)
        {
			for(int d=0; d<=dispLayers; d++)
			{
				int disp = d + mindisp;
				int xl = (x < 0) ? 0 : (x >= width ? width-1 : x);
				int xr = (x < 0 ? 0 : x) - disp;
				xr = (xr < 0) ? 0 : (xr >= width ? width-1 : xr);
                int indexL = INDEX(xl,y,height);
                int indexR = INDEX(xr,y,height);

				float value = 0;
				for(int i=0; i<3; i++)
                {
                    float difference = Left[indexL+HW*i]-Right[indexR+HW*i];
                    difference *= difference;
                    value += difference;
                }

                value = MIN(sqrt(value)/3, cost_threshold);
				if(x == xl)
				{
					CostL[indexL + HW*d] = value;
				}
                if(x-disp == xr && nout > 1)
                {
                    CostR[indexR + HW*d] = value;
                }
            }
        }
    }	

}

