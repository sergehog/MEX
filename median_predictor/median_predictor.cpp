#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}
#endif // __cplusplus
#include <math.h>
#include "../common/common.h"
#include <vector>
#include <algorithm>

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")

double median(double* image, bool *occlusions, unsigned index, unsigned height);
void leftpredict(double *out, double *image, bool *occlusions, unsigned index, unsigned height, unsigned HW);

void mexFunction (int nout, mxArray* output[], int in, const mxArray* input[])
{
	if(in < 2)
	{
		mexErrMsgTxt("USAGE: [I, occls_upd] = median_predictor(double(Image), logical(occls))"); 
    } 

	unsigned dims = mxGetNumberOfDimensions(input[0]);
	unsigned height = (mxGetDimensions(input[0]))[0];
	unsigned width = (mxGetDimensions(input[0]))[1];
	unsigned colors = dims > 2 ? 3 : 1;
	unsigned dims3d[] = {height, width, colors};
	unsigned dims2d[] = {height, width, 1};
	bool dispfloat = false;
	unsigned HW = width*height;

	output[0] = mxCreateNumericArray(3, dims3d, mxDOUBLE_CLASS, mxREAL);
	output[1] = mxCreateNumericArray(3, dims2d, mxLOGICAL_CLASS, mxREAL);

	double* image = (double*)mxGetData(input[0]);
	bool *occls = (bool*)mxGetData(input[1]);
	double* out = (double*)mxGetData(output[0]);
	bool *occlusions = (bool*)mxGetData(output[1]);

	for(int i=0; i<HW; i++)
	{
		occlusions[i] = occls[i];
	}

	for(int i=width-2; i>=0; i--)
	{
		leftpredict(out, image, occlusions, INDEX(i,0,height), height, HW);
		leftpredict(out, image, occlusions, INDEX(i,1,height), height, HW);
		leftpredict(out, image, occlusions, INDEX(i,height-1,height), height, HW);
		leftpredict(out, image, occlusions, INDEX(i,height-2,height), height, HW);
	}

	for(int j=2; j<height-2; j++)
	{
		bool started = false;
		for(int i=width-2; i>=0; i--)
		{
			unsigned index = INDEX(i,j,height);
			
			if(!occlusions[index])
			{
				if(!started)
					continue;
				occlusions[index] = 1;
				out[index] = median_right(out, occlusions, index, height);
				out[index + HW] = median_right(out + HW, occlusions, index, height);
				out[index + 2*HW] = median_right(out + 2*HW, occlusions, index, height);
				
			} else {
				out[index] = image[index];
				out[index + HW] = image[index + HW];
				out[index + 2*HW] = image[index + 2*HW];
			}

			started = true;
		}
	}

}

void leftpredict(double *out, double *image, bool *occlusions, unsigned index, unsigned height, unsigned HW)
{
	if(!occlusions[index])
	{
		out[index] = out[index+height];
		out[index+HW] = out[index+HW+height];
		out[index+2*HW] = out[index+2*HW+height];
		occlusions[index] = 1;
	} 
	else
	{
		out[index] = image[index];
		out[index+HW] = image[index+HW];
		out[index+2*HW] = image[index+2*HW];
	}
}


double median_right(double* image, bool *occlusions, unsigned index, unsigned height)//double a1, double a2, double a3, double a4, double a5, bool b1, bool b2, bool b3, bool b4, bool b5)
{
	std::vector<double> vals;
	vals.push_back(image[index-2]);
	vals.push_back(image[index-1]);
	vals.push_back(image[index+height-1]);
	vals.push_back(image[index+height]);
	
		
	if(occlusions[index+height+1])
	{
		vals.push_back(image[index+height+1]);
	}
	if(occlusions[index+1])
	{
		vals.push_back(image[index+1]);
	}
	if(occlusions[index+2])
	{
		vals.push_back(image[index+2]);
	}

	if(vals.size() == 7)
	{
		sort(vals.begin(), vals.end());
		return vals.at(3);
	}
	else if(vals.size() == 6)
		return median6<double>(vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]);
	else if(vals.size() == 5)
		return median5<double>(vals[0], vals[1], vals[2], vals[3], vals[4]);
	else if(vals.size()==4)
		return median5<double>(vals[0], vals[1], vals[2], vals[3]);
	else
		return median3<double>(vals[0], vals[1], vals[2]);
}

