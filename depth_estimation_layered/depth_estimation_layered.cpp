//depth_estimation_layered
/** computes Cost Volume at the reference camera position, using two or more unrectified cameras
* @file depth_estimation.cpp
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

#include <cfloat>
#include <algorithm>
#include <memory>
#ifndef _DEBUG
#include <omp.h>
#endif
#include "../common/meximage.h"

#define isnan _isnan
#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")
#pragma warning(disable:4244)
#pragma warning(disable:4018)

using namespace mymex;
//#define colors 9
//#define cost_thr 10.f

//#define MAXDIFF 255*3+1


#define LIM(A,B) ((A) < 0 ? 0 : ((A) > (B)-1 ? (B)-1 : (A)))

void projectImage(MexImage<float> &Desired, const glm::mat4 Cdinv, MexImage<float> &Image, const glm::mat4x3 Ci, const float z)
{
	const int d_width = Desired.width;
	const int d_height = Desired.height;
	const int i_width = Image.width;
	const int i_height = Image.height;
	const int colors = Image.layers;
	//const long HW = width * height;

	for (int ud = 0; ud<d_width; ud++)
	{
		for (int vd = 0; vd<d_height; vd++)
		{
			long indexd = Image.Index(ud, vd);
			const glm::vec3 uvzi = Ci * (Cdinv * glm::vec4(ud*z, vd*z, z, 1));

			const float ui = uvzi.x / uvzi.z;
			const float vi = uvzi.y / uvzi.z;
			const int ul = std::floor(ui);
			const int vl = std::floor(vi);

			const float dx = ui - ul;
			const float dy = vi - vl;

			//if(ul<0 || ul >= i_width-1 || vl < 0 || vl >= i_height-1)
			//{
			//	continue;
			//}

			//#pragma loop(hint_parallel(colors))
#pragma omp parallel for
			for (int c = 0; c<colors; c++)
			{
				Desired(ud, vd, c) = Image(LIM(ul, i_width), LIM(vl, i_height), c)*(1 - dx)*(1 - dy)
					+ Image(LIM(ul + 1, i_width), LIM(vl, i_height), c)*dx*(1 - dy)
					+ Image(LIM(ul, i_width), LIM(vl + 1, i_height), c)*(1 - dx)*dy
					+ Image(LIM(ul + 1, i_width), LIM(vl + 1, i_height), c)*dx*dy;
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

	const int _offset = 4; // 4 obligatory params goes before variable number of cameras

	if (in < _offset + 4 || (in - _offset - 4) % 2 != 0 || nout != 1 || mxGetClassID(input[_offset]) != mxSINGLE_CLASS || mxGetClassID(input[_offset+1]) != mxSINGLE_CLASS)
	{
		mexPrintf("Depth Estimation for a reference camera\n");
		mexErrMsgTxt("USAGE: [Cost] = depth_estimation_layered(minZ, maxZ, layers, Trimap, single(ImageRef), single(CRef), Image1, C1, <,..., ImageN, CN>);");
	}

	const int cameras = (in - _offset) / 2;

	if (cameras < 2)
	{
		mexErrMsgTxt("At least two camera frames (and matrixes) must be provided!");
	}

	if ((in - _offset) % 2)
	{
		mexErrMsgTxt("Number of Camera Matrixes should correspond to number of Camera Images!");
	}

	const float minZ = static_cast<float>(mxGetScalar(input[0]));
	const float maxZ = static_cast<float>(mxGetScalar(input[1]));
	const int layers = static_cast<int>(mxGetScalar(input[2]));

	const int height = mxGetDimensions(input[_offset])[0];
	const int width = mxGetDimensions(input[_offset])[1];
	const int colors = mymex::mxGetLayers(input[_offset]);
	const long HW = width*height;

	// validate input parameters
	for (int n = 0; n<cameras; n++)
	{
		if (mxGetClassID(input[_offset + n * 2]) != mxSINGLE_CLASS)
		{
			mexPrintf("Camera Image %d must be of a SINGLE-type", n + 1);
			mexErrMsgTxt("\n");
		}

		if ((mxGetDimensions(input[_offset + n * 2]))[0] != height || (mxGetDimensions(input[_offset + n * 2]))[1] != width || mymex::mxGetLayers(input[_offset + n * 2]) != colors)
		{
			mexErrMsgTxt("All Camera Images must have the same resolution and three colors!");
		}

		if (mxGetClassID(input[_offset + n * 2 + 1]) != mxSINGLE_CLASS)
		{
			mexPrintf("Camera Matrix %d must be of a SINGLE-type", n + 1);
			mexErrMsgTxt("\n");
		}

		if ((mxGetDimensions(input[_offset + n * 2 + 1]))[0] != 3 || (mxGetDimensions(input[_offset + n * 2 + 1]))[1] != 4)
		{
			mexPrintf("Camera matrix %d must be of size 3x4", n + 1);
			mexErrMsgTxt("\n");
		}
	}
	
	// data-structure with variable number of input images
	std::unique_ptr<MexImage<float>*[]> Images(new MexImage<float>*[cameras]);
	std::unique_ptr<glm::mat4x3[]> C(new glm::mat4x3[cameras]);


#pragma omp parallel for
	for (int n = 0; n<cameras; n++)
	{
		Images[n] = new MexImage<float>(input[_offset + n * 2]);
		const float * const c1 = (float*)mxGetData(input[_offset + n * 2 + 1]);
		C[n] = glm::mat4x3(c1[0], c1[1], c1[2], c1[3], c1[4], c1[5], c1[6], c1[7], c1[8], c1[9], c1[10], c1[11]);
	}

	const glm::mat4 Cdinv = glm::inverse(glm::mat4(C[0]));
	const float nan = sqrt(-1.f);
	const mwSize dims[] = { (unsigned)height, (unsigned)width, layers };

	// Matlab-allocated output param	
	output[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);

	MexImage<float> Cost(output[0]);
	Cost.setval(FLT_MAX);


	// for each depth-layer in the space
#pragma omp parallel 
	{
		MexImage<float> ColorTmpl(width, height, colors);
		ColorTmpl.setval(0);

#pragma omp for		
		for (int d = 0; d < layers; d++)
		{
#ifdef LINEAR_DEPTH_QUANTIZATION
			const float z = minZ + d*(maxZ - minZ) / (layers - 1);
#else 
			const float z = 1.f / ((float(d) / layers)*(1.f / minZ - 1.f / maxZ) + 1.f / maxZ);
#endif			

			MexImage<float> &ImageRef = *Images[0];

			for (int n = 1; n < cameras; n++)
			{
				MexImage<float> &ImageTempl = *Images[n];
				projectImage(ColorTmpl, Cdinv, ImageTempl, C[n], z);

#pragma omp parallel for
				for (long i = 0; i < HW; i++)
				{
					float diff = 0.f;
					for (int c = 0; c < colors; c++)
					{
						diff += abs(ImageRef[i + c*HW] - ColorTmpl[i + c*HW]);
					}
					//diff /= colors;
					//diff = diff > cost_thr ? cost_thr : diff;

					Cost[i + d*HW] = std::min(diff, Cost[i + d*HW]);
				}
			}
		}
	}

	for (int n = 0; n<cameras; n++)
	{
		delete Images[n];
	}

}