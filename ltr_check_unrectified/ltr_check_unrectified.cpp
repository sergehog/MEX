/** ltr_check_unrectified
* @author Sergey Smirnov
*/
#define GLM_MESSAGES 
//#define GLM_FORCE_CXX98  
#define GLM_FORCE_CXX11  
//#define GLM_FORCE_SSE2 
#include <glm/glm.hpp>
//#include <glm/gtc/matrix_transform.hpp>


#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}

#endif // __cplusplus
//#include "../common/common.h"
#include "../common/meximage.h"
#include <algorithm>
#ifndef _NDEBUG
#include <omp.h>
#endif

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")
#pragma warning(disable:4244)
#pragma warning(disable:4018)

using namespace mymex;

void calculateOcclusions(MexImage<float> &Depth1, MexImage<float> &Depth2, const glm::mat4x3 C1, const glm::mat4x3 C2, const float minZ, const float maxZ, const int layers, const float threshold, MexImage<bool> &Occl12)
{
	const int height1 = Depth1.height;
	const int width1 = Depth1.width;
	const long HW1 = width1*height1;
    
	const int height2 = Depth2.height;
	const int width2 = Depth2.width;
	const long HW2 = width2*height2;
	
	const glm::mat4 C1inv = glm::inverse(glm::mat4(C1));

	#pragma omp parallel for
	for(long i=0; i<HW1; i++)
	{
		Occl12.data[i] = 0;
	}

	#pragma omp parallel for
	for(long i=0; i<HW1; i++)
	{
		const int u1 = i / height1;
		const int v1 = i % height1;
		//const float z1 = 1.f/((Depth1[i]/layers)*(1.f/minZ - 1.f/maxZ) + 1.f/maxZ);
		const float z1 = Depth1[i];
		const glm::vec3 uvz2 = C2 * C1inv * glm::vec4(u1*z1, v1*z1, z1, 1);
		const float u2 = uvz2.x / uvz2.z;
		const float v2 = uvz2.y / uvz2.z;
		const int ul = std::floor(u2);
		const int vl = std::floor(v2);

		if(ul<0 || ul >= width2-1 || vl < 0 || vl >= height2-1)
		{
			continue;
		}

		const long indexA = Depth2.Index(ul, vl);
		const long indexB = Depth2.Index(ul+1, vl);
		const long indexC = Depth2.Index(ul, vl+1);
		const long indexD = Depth2.Index(ul+1, vl+1);
		const float dx = u2-ul;
		const float dy = v2-vl;

		float interpolated = Depth2[indexA] * (1 - dx)*(1 - dy) + Depth2[indexB] * dx*(1 - dy) + Depth2[indexC] * (1 - dx)*dy + Depth2[indexD] * dx*dy;
		//float interpolated = Depth2[indexA]*dx*dy +  Depth2[indexB]*(1-dx)*dy + Depth2[indexC]*dx*(1-dy) + Depth2[indexD]*(1-dx)*(1-dy);

		if(abs(Depth1[i]-interpolated) < threshold)
		{
			Occl12.data[i] = 1;
		}
	}
}

void mexFunction (int nout, mxArray* output[], int in, const mxArray* input[])
{
	#ifndef _NDEBUG
	omp_set_num_threads(std::max(2, omp_get_max_threads())); 
	omp_set_dynamic(omp_get_max_threads()-1);
	#endif

	if(in < 6 || in > 8 || nout != 2 || mxGetClassID(input[0])!=mxSINGLE_CLASS || mxGetClassID(input[1])!=mxSINGLE_CLASS || mxGetClassID(input[2])!=mxSINGLE_CLASS || mxGetClassID(input[3])!=mxSINGLE_CLASS)
	{
		mexErrMsgTxt("USAGE: [Occl12, Occcl21] = ltr_check_unrectified(single(Z1), single(Z2), C1, C2, minZ, maxZ, [layers, threshold=1]);"); 
    } 
	MexImage<float> Depth1(input[0]);
	MexImage<float> Depth2(input[1]);

	const int height1 = Depth1.height;
	const int width1 = Depth1.width;
	const long HW1 = width1*height1;
    
	const int height2 = Depth2.height;
	const int width2 = Depth2.width;
	const long HW2 = width2*height2;
	
	if(Depth1.layers != 1 || Depth2.layers != 1)
	{
		mexErrMsgTxt("Depth maps must be single-color images."); 
	}

	if((mxGetDimensions(input[2]))[0] != 3 || (mxGetDimensions(input[2]))[1] != 4)
	{
		mexErrMsgTxt("C1 matrix must be 4x3"); 
	}

	if((mxGetDimensions(input[3]))[0] != 3 || (mxGetDimensions(input[3]))[1] != 4)
	{
		mexErrMsgTxt("C2 matrix must be 4x3"); 
	}
	
	const float * const c1 = (float*)mxGetData(input[2]);
	const float * const c2 = (float*)mxGetData(input[3]);

	const glm::mat4x3 C1 = glm::mat4x3(c1[0],c1[1],c1[2],c1[3],c1[4],c1[5],c1[6],c1[7],c1[8],c1[9],c1[10],c1[11]);
	const glm::mat4x3 C2 = glm::mat4x3(c2[0],c2[1],c2[2],c2[3],c2[4],c2[5],c2[6],c2[7],c2[8],c2[9],c2[10],c2[11]);

	//const glm::mat4 C1inv = glm::inverse(glm::mat4(C1));	
	//const glm::mat4 C2inv = glm::inverse(glm::mat4(C2));	

	const float minZ = static_cast<float>(mxGetScalar(input[4]));	
	const float maxZ = static_cast<float>(mxGetScalar(input[5]));	
	const int layers = (in > 6) ?  static_cast<int>(mxGetScalar(input[6])) : 256;	
	const float threshold = (in > 7) ?  static_cast<float>(mxGetScalar(input[7])) : 1.f;	
	const float nan = sqrt(-1.f);
	const mwSize depthcost1[] = {(unsigned)height1, (unsigned)width1, 1};
	const mwSize depthcost2[] = {(unsigned)height2, (unsigned)width2, 1};
	
	//Matlab-allocated variables
	output[0] = mxCreateNumericArray(3, depthcost1, mxLOGICAL_CLASS, mxREAL);
	output[1] = mxCreateNumericArray(3, depthcost2, mxLOGICAL_CLASS, mxREAL);
	MexImage<bool> Occl12(output[0]);
	MexImage<bool> Occl21(output[1]);

	#pragma omp parallel sections
	{
		#pragma omp section
		{
			calculateOcclusions(Depth1, Depth2, C1, C2, minZ, maxZ, layers, threshold, Occl12);
		}
		#pragma omp section
		{
			calculateOcclusions(Depth2, Depth1, C2, C1, minZ, maxZ, layers, threshold, Occl21);
			
		}
	}	
}