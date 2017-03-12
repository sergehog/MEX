#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}

#endif // __cplusplus

#include "../common/common.h"
#include <cmath>

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")
#pragma warning(disable:4244)
#pragma warning(disable:4018)


#define OCCLUDED 0
#define NOTVISIBLE 0
#define NOTSMOOTH 0
#define BADMATCH 0
#define GOODMATCH 1 

//void ltr_check_float(float*, float *, UINT8*, UINT8*, double, unsigned, unsigned);
//void ltr_check_integer(UINT8*, UINT8*, UINT8*, UINT8*, double, unsigned, unsigned);

UINT8 maxdisp = 1;
float maxdispf = 1;

void mexFunction (int nout, mxArray* output[], int in, const mxArray* input[])
{
	if(in != 3 || nout != 2)// || mxGetClassID(input[0])!=mxUINT8_CLASS || mxGetClassID(input[1])!=mxUINT8_CLASS)
	{
		mexErrMsgTxt("Left-to-Right Consistency Check \n USAGE: [OcclL, OcclR] = ltr_check(DispL, DispR, threshold);"); 
    } 
	//const unsigned dims = (unsigned)mxGetNumberOfDimensions(input[0]);
	const int height = (unsigned)(mxGetDimensions(input[0]))[0];
	const int width = (unsigned)(mxGetDimensions(input[0]))[1];
	const int HW = height*width;

	if(height!=(mxGetDimensions(input[1]))[0] || width!=(mxGetDimensions(input[1]))[1])
	{
		mexErrMsgTxt("ERROR: Sizes of parameters 'DispL' and 'DispR' must be the same."); 
	}

	if(mxGetClassID(input[0]) != mxGetClassID(input[1]))
	{
		mexErrMsgTxt("ERROR: Sizes of parameters 'DispL' and 'DispR' must be of the same type."); 
	}

	const mwSize depthdims[] = {height, width, 1};	
	//mexPrintf("Here is all OK. 1\r\n");

	output[0] = mxCreateNumericArray(3, depthdims, mxUINT8_CLASS, mxREAL);
	output[1] = mxCreateNumericArray(3, depthdims, mxUINT8_CLASS, mxREAL);

	UINT8 *OcclL = (UINT8*) mxGetData(output[0]);
	UINT8 *OcclR = (UINT8*) mxGetData(output[1]);

	float *DispL = NULL, *DispR = NULL;

	if(mxGetClassID(input[0]) == mxSINGLE_CLASS)
    {
        DispL = (float*)mxGetData(input[0]);
	    DispR = (float*)mxGetData(input[1]);
		//mexPrintf("floats\r\n");
    }
    else if(mxGetClassID(input[0]) == mxUINT8_CLASS)
    {
        UINT8 *DispLi = (UINT8*)mxGetData(input[0]);
	    UINT8 *DispRi = (UINT8*)mxGetData(input[1]);
		DispL = new float[HW];
		DispR = new float[HW];
        for(int i=0; i<HW; i++)
        {
			DispL[i] = (float)DispLi[i];
			DispR[i] = (float)DispRi[i];			
        }
		//mexPrintf("ints\r\n");
    }
    else
    {
        mexErrMsgTxt("ERROR: 'DispL' and 'DispR' must be both of SINGLE type or of UINT8 type."); 
    }

    const float threshold = (float)mxGetScalar(input[2]);


	for(int x=0, index=0; x<width; x++)
	{
		for(int y=0; y<height; y++, index++)
		{
			float dispL = DispL[index];
			float dispR = DispR[index];
			int xR = ROUND(x-dispL, int);
			int xL = ROUND(x+dispR, int);
			if(xR<0 || xR >= width)
			{
				OcclL[index] = 0;
			}
			else
			{
				int indexR = INDEX(xR, y, height);
				bool occlL = (std::abs(DispR[indexR] - dispL) <= threshold);
				OcclL[index] = (UINT8)occlL;
			}
			if(xL<0 || xL >= width)
			{
				OcclR[index] = 0;
			}
			else
			{
				int indexL = INDEX(xL, y, height);
				bool occlR = (std::abs(DispL[indexL] - dispR) <= threshold);
				OcclR[index] = (UINT8)occlR;
			}
		}
	}
	//mexPrintf("got it!\r\n");


	if(mxGetClassID(input[0]) == mxUINT8_CLASS)
	{
		delete[] DispL;
		delete[] DispR;
	}
}


