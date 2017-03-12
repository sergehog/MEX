/** plane_projection
* @author Sergey Smirnov
* @date 23.04.2014
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


#define LIM(A,B) ((A) < 0 ? 0 : ((A) > (B)-1 ? (B)-1 : (A)))

void projectImage_CLAMP_TO_EDGE(MexImage<float> &Desired, const glm::mat4 Cdinv, MexImage<float> &Image, const glm::mat4x3 Ci, const float z)
{
	const int d_width = Desired.width;
	const int d_height = Desired.height;
	const int i_width = Image.width;
	const int i_height = Image.height;
	const int colors = Desired.layers;
	//const long HW = width * height;

	for(int ud=0; ud<d_width; ud++)
	{
		for(int vd=0; vd<d_height; vd++)
		{
			long indexd = Image.Index(ud, vd);
			const glm::vec3 uvzi = Ci * (Cdinv * glm::vec4(ud*z, vd*z, z, 1));				

			const float ui = uvzi.x / uvzi.z;
			const float vi = uvzi.y / uvzi.z;
			const int ul = std::floor(ui);
			const int vl = std::floor(vi);
						
			const float dx = ui-ul;
			const float dy = vi-vl;

			//if(ul<0 || ul >= i_width-1 || vl < 0 || vl >= i_height-1)
			//{
			//	continue;
			//}
				
			//#pragma loop(hint_parallel(colors))
			for(int c=0; c<colors; c++)
			{
				Desired(ud, vd, c) = Image(LIM(ul,i_width), LIM(vl,i_height), c)*(1-dx)*(1-dy) 
					+ Image(LIM(ul+1,i_width), LIM(vl,i_height), c)*dx*(1-dy) 
					+ Image(LIM(ul,i_width),LIM(vl+1,i_height),c)*(1-dx)*dy 
					+ Image(LIM(ul+1,i_width),LIM(vl+1,i_height),c)*dx*dy;													
			}			
		}
	}
}


void projectImage(MexImage<float> &Desired, const glm::mat4 Cdi, MexImage<float> &Image, const glm::mat4x3 C1, const float z)
{
	const int d_width = Desired.width;
	const int d_height = Desired.height;
	const int i_width = Image.width;
	const int i_height = Image.height;
	const int colors = Desired.layers;

	const long dHW = d_width * d_height;

	for (int ud = 0; ud<d_width; ud++)
	{
		for (int vd = 0; vd<d_height; vd++)
		{
			long indexd = Image.Index(ud, vd);
			const glm::vec3 uvzi = C1 * (Cdi * glm::vec4(ud*z, vd*z, z, 1));
			//const glm::vec3 uvzi = Cd * (Cinv * glm::vec4(ud*z, vd*z, z, 1));

			const float ui = uvzi.x / uvzi.z;
			const float vi = uvzi.y / uvzi.z;
			const int ul = std::floor(ui);
			const int vl = std::floor(vi);

			if (ul<0 || ul >= i_width - 1 || vl < 0 || vl >= i_height - 1)
			{
				continue;
			}

			const float dx = ui - ul;
			const float dy = vi - vl;

			for (int c = 0; c<colors; c++)
			{
				Desired(ud, vd, c) = Image(ul, vl, c)*(1 - dx)*(1 - dy) + Image(ul + 1, vl, c)*dx*(1 - dy) + Image(ul, vl + 1, c)*(1 - dx)*dy + Image(ul + 1, vl + 1, c)*dx*dy;
			}
		}
	}
}

void mexFunction(int nout, mxArray* output[], int in, const mxArray* input[])
{
#ifndef _DEBUG
	omp_set_num_threads(std::max(4, omp_get_max_threads()));
	omp_set_dynamic(std::max(2, omp_get_max_threads() / 2));
#endif
	
	if (in < 4 || in > 5 || nout != 1 || mxGetClassID(input[2]) != mxSINGLE_CLASS || mxGetClassID(input[1]) != mxSINGLE_CLASS || mxGetClassID(input[3]) != mxSINGLE_CLASS)
	{
		mexErrMsgTxt("USAGE: [Desired] = plane_projection(z, C_desired, single(Image1), C1 <,[h w]>);");
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
	if(in == 5)
	{
		if(mxGetClassID(input[4]) != mxDOUBLE_CLASS)
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