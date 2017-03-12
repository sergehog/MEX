/** Takes two Z maps with their camera matrixes and checks if all pixels corresponds to each outher
* @file left2right.cpp
* @author Sergey Smirnov <sergey.smirnov@tut.fi>
* @date 22.10.2014
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
#define colors 3
#define cost_thr 20.f

template<typename T>
float getDepth(T val, const float minZ, const float maxZ)
{
	return (float)val;
}

template<>
float getDepth<uint8_t>(uint8_t val, const float minZ, const float maxZ)
{
	return 1.f / (float(val) / 255.f + (1 / minZ - 1 / maxZ) + 1 / maxZ);
}

template<typename T>
void checkValidity(MexImage<bool> &Valid, glm::mat4 Cinv, glm::mat4x3 Ct, const MexImage<T> &Zref, const MexImage<T> &Zt, const float thr, const float minZ, const float maxZ)
{
	const int width = Zref.width;
	const int height = Zref.height;
	const long HW = width * height;
	const int t_width = Zt.width;
	const int t_height = Zt.height;

	#pragma omp parallel for
	for(long i=0; i<HW; i++)
	{
		const int u = i / height;
		const int v = i % height;
		//float const z = Zref[i];
		float const z = getDepth<T>(Zref[i], minZ, maxZ);
		if (isnan(z))
		{
			Valid[i] = false;
			continue;
		}
		const glm::vec3 uvzt = Ct * (Cinv * glm::vec4(u*z, v*z, z, 1));
		const float ui = uvzt.x / uvzt.z;
		const float vi = uvzt.y / uvzt.z;

		const int ul = std::floor(ui);
		const int vl = std::floor(vi);

		if(ul < 0 || vl < 0 || ul >= t_width-1 || vl >= t_height-1)
		{
			Valid[i] = false;
			continue;
		}

		const float dx = ui-ul;
		const float dy = vi-vl;
		float const za = getDepth<T>(Zt(ul, vl), minZ, maxZ);
		float const zb = getDepth<T>(Zt(ul+1, vl), minZ, maxZ);
		float const zc = getDepth<T>(Zt(ul, vl+1), minZ, maxZ);
		float const zd = getDepth<T>(Zt(ul+1, vl+1), minZ, maxZ);

		const float zt = za*(1-dx)*(1-dy) + zb*dx*(1-dy) + zc*(1-dx)*dy + zd*dx*dy;
		
		const float dt = (255 * (minZ / zt) * ((maxZ - zt) / (maxZ - minZ)));
		const float di = (255 * (minZ / uvzt.z) * ((maxZ - uvzt.z) / (maxZ - minZ)));
		Valid[i] = abs(dt - di) < thr;
		//Valid[i] = abs(zt-uvzt.z) < thr;
		
	}	
}


void mexFunction (int nout, mxArray* output[], int in, const mxArray* input[])
{
	#ifndef _DEBUG
	omp_set_num_threads(std::max(4,omp_get_max_threads())); 
	omp_set_dynamic(std::max(2, omp_get_max_threads() / 2));
	#endif
	
	mxClassID type1 = mxUNKNOWN_CLASS, type2 = mxUNKNOWN_CLASS;
	if (in > 3)
	{
		type1 = mxGetClassID(input[1]);
		type2 = mxGetClassID(input[3]);
	}

	if (in < 7 || nout != 2 || !((type1 == mxSINGLE_CLASS && type2 == mxSINGLE_CLASS) || (type1 == mxUINT8_CLASS && type2 == mxUINT8_CLASS)))
	{
		mexPrintf("Left-to-Right Correspondance Check for unrectified cameras (depth from plane-sweeping).\n");
		mexErrMsgTxt("USAGE: [Valid1, Valid2] = left2right(single(Z1) or uint8(Disp), single(C1), Z2, C2, threshold, minZ, maxZ);");
	}

	const float threshold = static_cast<float>(mxGetScalar(input[4]));

	const float * const c1 = (float*)mxGetData(input[1]);
	glm::mat4x3 C1 = glm::mat4x3(c1[0], c1[1], c1[2], c1[3], c1[4], c1[5], c1[6], c1[7], c1[8], c1[9], c1[10], c1[11]);
	const float * const c2 = (float*)mxGetData(input[3]);
	glm::mat4x3 C2 = glm::mat4x3(c2[0], c2[1], c2[2], c2[3], c2[4], c2[5], c2[6], c2[7], c2[8], c2[9], c2[10], c2[11]);

	const glm::mat4 C1inv = glm::inverse(glm::mat4(C1));
	const glm::mat4 C2inv = glm::inverse(glm::mat4(C2));

	const int width1 = mymex::mxGetWidth(input[0]);
	const int height1 = mymex::mxGetHeight(input[0]);
	const int width2 = mymex::mxGetWidth(input[2]);
	const int height2 = mymex::mxGetHeight(input[2]);

	const mwSize dims1[] = { (size_t)height1, (size_t)width1, 1 };
	const mwSize dims2[] = { (size_t)height2, (size_t)width2, 1 };

	//Matlab-allocated variables	
	output[0] = mxCreateNumericArray(3, dims1, mxLOGICAL_CLASS, mxREAL);
	output[1] = mxCreateNumericArray(3, dims2, mxLOGICAL_CLASS, mxREAL);

	MexImage<bool> Valid1(output[0]);
	MexImage<bool> Valid2(output[1]);
	const float minZ = static_cast<float>(mxGetScalar(input[5]));
	const float maxZ = static_cast<float>(mxGetScalar(input[6]));

	if (mxGetClassID(input[0]) == mxSINGLE_CLASS && mxGetClassID(input[2]) == mxSINGLE_CLASS)
	{
		MexImage<float> Z1(input[0]);
		MexImage<float> Z2(input[2]);
		float const nan = sqrt(-1.f);
#pragma omp parallel sections
		{
#pragma omp section
		{
			checkValidity<float>(Valid1, C1inv, C2, Z1, Z2, threshold, minZ, maxZ);
		}

#pragma omp section
		{
			checkValidity<float>(Valid2, C2inv, C1, Z2, Z1, threshold, minZ, maxZ);
		}
		}
	}
	else if (mxGetClassID(input[0]) == mxUINT8_CLASS && mxGetClassID(input[2]) == mxUINT8_CLASS && in == 7)
	{
		MexImage<uint8_t> D1(input[0]);
		MexImage<uint8_t> D2(input[2]);
#pragma omp parallel sections
		{
#pragma omp section
		{
			checkValidity<uint8_t>(Valid1, C1inv, C2, D1, D2, threshold, minZ, maxZ);
		}

#pragma omp section
		{
			checkValidity<uint8_t>(Valid2, C2inv, C1, D2, D1, threshold, minZ, maxZ);
		}
		}
	}
	else
	{
		mexPrintf("Depth maps must be transmitted as raw depth (SINGLE-valued) or as Generalized Disparity Map/Inverse Depth (UINT8).\n In the latter case you should also specify minZ and maxZ.");
	}
	
	
	

	


		
		

}