/** 
* @file mesh_stereo.cpp
* @author Sergey Smirnov <sergey.smirnov@tut.fi>
* @date 10.3.2015
* @copyright 3D Media Group / Tampere University of Technology
*/

#define GLM_FORCE_CXX11  
#include <glm/glm.hpp>
//#include <glm/gtc/matrix_transform.hpp>

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
#pragma warning(disable:4244)
#pragma warning(disable:4018)

using namespace mymex;

void mexFunction(int nout, mxArray* output[], int in, const mxArray* input[])
{
#ifndef _DEBUG
	omp_set_num_threads(std::max(4, omp_get_max_threads()));
	omp_set_dynamic(std::max(2, omp_get_max_threads() / 2));
#endif

	if (in != 2 || nout != 2 || mxGetClassID(input[0]) != mxSINGLE_CLASS)
	{
		mexPrintf("Depth Estimation for a reference camera\n");
		mexErrMsgTxt("USAGE: [Disp, Vertexes, Triangles] = mesh_stereo(single(Cost), tolerable_error);");
	}

	MexImage<float>Cost(input[0]);	
	const float threshold = static_cast<float>(mxGetScalar(input[1]));

	const int width = Cost.width;
	const int height = Cost.height;
	const mwSize dims[] = { (size_t)height, (size_t)width, 1 };
	
	//Matlab-allocated variables	
	output[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
	MexImage<float> Disp(output[0]);

	


}