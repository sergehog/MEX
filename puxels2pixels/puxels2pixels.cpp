/** puxels2pixels
* @author Sergey Smirnov
* @date 08.04.2016
*/


#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}
#endif // __cplusplus

#include "../common/meximage.h"
#include <algorithm>
#include <memory>
#ifndef _DEBUG
#include <omp.h>
#endif

#define isnan _isnan
#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")

using namespace mymex;


void mexFunction(int nout, mxArray* output[], int in, const mxArray* input[])
{
#ifndef _DEBUG
	omp_set_num_threads(std::max(4, omp_get_max_threads()));
	omp_set_dynamic(std::max(2, omp_get_max_threads() / 2));
#endif

	if (in < 4 || in > 5 || nout != 1 || mxGetClassID(input[2]) != mxSINGLE_CLASS || mxGetClassID(input[1]) != mxSINGLE_CLASS || mxGetClassID(input[3]) != mxSINGLE_CLASS)
	{
		mexErrMsgTxt("USAGE: [] = puxels2pixels(S, single(Image1), C1 <,[h w]>);");
	}

	const float z = static_cast<float>(mxGetScalar(input[0]));
	const float * const cd = (float*)mxGetData(input[1]);
	const float * const c1 = (float*)mxGetData(input[3]);
	glm::mat4x3 Cd = glm::mat4x3(cd[0], cd[1], cd[2], cd[3], cd[4], cd[5], cd[6], cd[7], cd[8], cd[9], cd[10], cd[11]);
	glm::mat4x3 C1 = glm::mat4x3(c1[0], c1[1], c1[2], c1[3], c1[4], c1[5], c1[6], c1[7], c1[8], c1[9], c1[10], c1[11]);
	//glm::mat4 Cinv = glm::inverse(glm::mat4(C1));
	glm::mat4 Cdi = glm::inverse(glm::mat4(Cd));
	MexImage<float> Image(input[2]);
	int height = Image.height;
	int width = Image.width;
	const int colors = Image.layers;
	const float nan = sqrt(-1.f);
	if (in == 5)
	{
		if (mxGetClassID(input[4]) != mxDOUBLE_CLASS)
		{
			mexErrMsgTxt("[h w] must be array of type DOUBLE.");
		}
		const double * const hw = (double*)mxGetData(input[4]);
		height = int(hw[0]);
		width = int(hw[1]);
	}

	const mwSize dimC[] = { (unsigned)height, (unsigned)width, (unsigned)colors };
	//Matlab-allocated variables
	output[0] = mxCreateNumericArray(3, dimC, mxSINGLE_CLASS, mxREAL);
	MexImage<float> Projected(output[0]);
	Projected.setval(nan);
	projectImage_CLAMP_TO_EDGE(Projected, Cdi, Image, C1, z);
	//projectImage(Projected, Cdi, Image, C1, z);
}