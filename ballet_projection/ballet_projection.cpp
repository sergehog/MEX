/** Specialized version of plane_projection, specifically for "Ballet" and "Breakdancers" datasets
* It's not working :(
* @file ballet_projection
* @author Sergey Smirnov
* @date 21.08.2014
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


void projectImage(MexImage<float> &Desired, const glm::mat4x3 Cd, MexImage<float> &Image, const glm::mat4x3 C1, const float z)
{
	const int d_width = Desired.width;
	const int d_height = Desired.height;
	const int i_width = Image.width;
	const int i_height = Image.height;
	const int colors = Desired.layers;

	const long dHW = d_width * d_height;

	//for (int ud = 0; ud<d_width; ud++)
	for (int u = 0; u<i_width; u++)
	{
		//for (int vdb = 0; vdb<d_height; vdb++)
		for (int vb = 0; vb<i_height; vb++)
		{			
			//const long indexd = Image.Index(u, vb);
			
			// image (0,0) is bottom lefthand corner
			double v = (double) i_height - vb - 1.0;

			double c0 = z*C1[0][2] + C1[0][3];
			double c1 = z*C1[1][2] + C1[1][3];
			double c2 = z*C1[2][2] + C1[2][3];

			double y = u*(c1*C1[2][0] - C1[1][0]*c2) + v*(c2*C1[0][0] - C1[2][0]*c0) + C1[1][0]*c0 - c1*C1[0][0];
			y /= v*(C1[2][0]*C1[0][1] - C1[2][1]*C1[0][0]) + u*(C1[1][0]*C1[2][1] - C1[1][1]*C1[2][0]) + C1[0][0]*C1[1][1] - C1[1][0]*C1[0][1];
		
			double x = (y)*(C1[0][1] - C1[2][1]*u) + c0 - c2*u;
			x /= C1[2][0]*u - C1[0][0];

			////////////////////////////////
			double ud = Cd[0][0]*x + Cd[0][1]*y + Cd[0][2]*z + Cd[0][3];
			double vd = Cd[1][0]*x + Cd[1][1]*y + Cd[1][2]*z + Cd[1][3];
			double w = Cd[2][0]*x + Cd[2][1]*y + Cd[2][2]*z + Cd[2][3];

			ud /= w;
			vd /= w;

			// image (0,0) is bottom lefthand corner
			vd = (double) d_height - vd - 1.0;

			for (int c = 0; c<colors; c++)
			{
				Desired(int(ud), int(vd), c) = Image(u, vb, c);
			}

			//const int vd = i_height - vdb - 1;

			//const glm::vec3 uvzi = Cd * (Cinv * glm::vec4(ud*z, vd*z, z, 1));

			//const float ui = uvzi.x / uvzi.z;
			//const float vi = uvzi.y / uvzi.z;
			//
			//const float vib = i_height - vi - 1;
			//const int ul = std::floor(ui);
			//const int vl = std::floor(vib);

			//if (ul<0 || ul >= i_width - 1 || vl < 0 || vl >= i_height - 1)
			//{
			//	continue;
			//}

			//const float dx = ui - ul;
			//const float dy = vib - vl;

			
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
	projectImage(Projected, Cd, Image, C1, z);
}