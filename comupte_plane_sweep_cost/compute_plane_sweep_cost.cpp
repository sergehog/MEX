/** compute_plane_sweep_cost
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

void calculateCostVolume(MexImage<float> &Image1, MexImage<float> &Image2, const glm::mat4x3 C1, const glm::mat4x3 C2, const float minZ, const float maxZ, const int layers, const float cost_threshold, MexImage<float> &Cost1)
{
	const int height1 = Image1.height;
	const int width1 = Image1.width;
	const int colors = Image1.layers;
	const long HW1 = width1*height1;
    
	const int height2 = Image2.height;
	const int width2 = Image2.width;
	const long HW2 = width2*height2;
	
	const glm::mat4 C1inv = glm::inverse(glm::mat4(C1));

	#pragma omp parallel for
	for(long i=0; i<HW1*layers; i++)
	{
		Cost1.data[i] = cost_threshold;
	}

	#pragma omp parallel for
	for(int d=0; d<layers; d++)
	{		
		const float z =  1.f/((float(d)/layers)*(1.f/minZ - 1.f/maxZ) + 1.f/maxZ);
		//float z = maxZ - d * (maxZ - minZ) / layers;
		//const float z = minZ + d * (maxZ - minZ) / layers;

		for(int u1=0; u1<width1; u1++)
		{
			for(int v1=0; v1<height1; v1++)
			{
				long index1 = Image1.Index(u1, v1);
				const glm::vec3 uvz2 = C2 * (C1inv * glm::vec4(u1*z, v1*z, z, 1));				
				const float u2 = uvz2.x / uvz2.z;
				const float v2 = uvz2.y / uvz2.z;
				const int ul = std::floor(u2);
				const int vl = std::floor(v2);

				if(ul<0 || ul >= width2-1 || vl < 0 || vl >= height2-1)
				{
					continue;
				}
				
				const long indexA = Image2.Index(ul, vl);
				const long indexB = Image2.Index(ul+1, vl);
				const long indexC = Image2.Index(ul, vl+1);
				const long indexD = Image2.Index(ul+1, vl+1);
				const float dx = u2-ul;
				const float dy = v2-vl;

				float diff = 0;
				#pragma omp parallel for reduction(+:diff)
				for(int c=0; c<colors; c++)
				{
					float interpolated = Image2[indexA + c*HW2]*(1-dx)*(1-dy) + Image2[indexB + c*HW2] * dx*(1-dy) + Image2[indexC + c*HW2] * (1-dx)*dy + Image2[indexD + c*HW2] * dx*dy;
					diff += abs(interpolated - Image1[index1 + c*HW1]);
				}
				diff /= colors;
				Cost1.data[index1 + d*HW1] = (diff > cost_threshold) ? cost_threshold : diff; //(!_isnan(cost_threshold)) && (diff > cost_threshold) ? cost_threshold : diff;
			}
		}
	}
}

void mexFunction (int nout, mxArray* output[], int in, const mxArray* input[])
{
	#ifndef _NDEBUG
	omp_set_num_threads(std::max(4, omp_get_max_threads())); 
	omp_set_dynamic(std::max(2,omp_get_max_threads()/2));
	#endif

	if(in < 6 || in > 8 || nout != 2 || mxGetClassID(input[0])!=mxSINGLE_CLASS || mxGetClassID(input[1])!=mxSINGLE_CLASS || mxGetClassID(input[2])!=mxSINGLE_CLASS || mxGetClassID(input[3])!=mxSINGLE_CLASS)
	{
		mexErrMsgTxt("USAGE: [Cost12, Cost21] = compute_plane_sweep_cost(single(Image1), single(Image2), C1, C2, minZ, maxZ, [layers, cost_threshold]);"); 
    } 
	MexImage<float> Image1(input[0]);
	MexImage<float> Image2(input[1]);

	const int height1 = Image1.height;
	const int width1 = Image1.width;
	const int colors = Image1.layers;
	const long HW1 = width1*height1;
    
	const int height2 = Image2.height;
	const int width2 = Image2.width;
	const long HW2 = width2*height2;

	if(colors != Image2.layers)
	{
		mexErrMsgTxt("ERROR: at least number of colors must coincide o_O."); 
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
	const glm::mat4 C1inv = glm::inverse(glm::mat4(C1));	

	const float minZ = static_cast<float>(mxGetScalar(input[4]));	
	const float maxZ = static_cast<float>(mxGetScalar(input[5]));	
	const int layers = (in > 6) ?  static_cast<int>(mxGetScalar(input[6])) : 256;	
	const float cost_threshold = (in > 7) ?  static_cast<float>(mxGetScalar(input[7])) : 255.f;	
	const float nan = sqrt(-1.f);
	const mwSize depthcost1[] = {(unsigned)height1, (unsigned)width1, (unsigned)layers};
	const mwSize depthcost2[] = {(unsigned)height2, (unsigned)width2, (unsigned)layers};
	
	//Matlab-allocated variables
	output[0] = mxCreateNumericArray(3, depthcost1, mxSINGLE_CLASS, mxREAL);
	output[1] = mxCreateNumericArray(3, depthcost2, mxSINGLE_CLASS, mxREAL);
	MexImage<float> Cost1(output[0]);	
	MexImage<float> Cost2(output[1]);	

	#pragma omp parallel sections
	{
		#pragma omp section
		{
			calculateCostVolume(Image1, Image2, C1, C2, minZ, maxZ, layers, cost_threshold, Cost1);
		}
		#pragma omp section
		{
			calculateCostVolume(Image2, Image1, C2, C1, minZ, maxZ, layers, cost_threshold, Cost2);
		}
	}
}