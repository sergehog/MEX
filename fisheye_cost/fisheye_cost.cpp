/** computes Cost Volume at the reference camera position, using two or more unrectified fisheye-lens cameras
* @file fisheye_cost.cpp
* @author Sergey Smirnov <sergey.smirnov@tut.fi>
* @date 23.03.2015
* @copyright 3D Media Group / Tampere University of Technology
*/

#define GLM_FORCE_CXX11  
#include <glm/glm.hpp>
#include <glm/ext.hpp>

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
#include "../common/fisheye.h"

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

//extern const double AngleToHeightNormalized[];
//extern const double HeightToAngle[];
const double diameter = 1305.0;
//extern double sunex(double phi);
//extern double reverseSunex(double r);
const float leftMargin = 0;
const float topMargin = 0;

glm::vec3 rotq(glm::vec4 q, glm::vec3 vec)
{
	glm::vec3 q3 = glm::vec3(q.x, q.y, q.z);
	glm::vec3 a = glm::cross(vec, q3);
	return vec + 2.f * glm::cross(a + q.w * vec, q3);
}

void projectFisheyeImage(MexImage<float> &Desired0, const glm::vec4 quat0, const glm::vec2 center0, const glm::vec3 position0, MexImage<float> &Image1, const glm::vec4 quat1, const glm::vec2 center1, const glm::vec3 position1, const float z)
{	
	const int width = Desired0.width;
	const int height = Desired0.height;
	const int colors = Desired0.layers;
	const long HW = width * height;
	const float nan = sqrt(-1);
	Desired0.setval(nan);
	//glm::quat Qinv = glm::inverse(quat0);
	#pragma omp parallel for
	for (long i = 0; i < HW; i++)
	{
		int x0 = i / height;
		int y0 = i % height;

		float x0norm = x0 - (center0.x - diameter * 0.5 - leftMargin);
		float y0norm = y0 - (center0.y - diameter * 0.5 - topMargin);
		x0norm = 2 * x0norm / diameter - 1;
		y0norm = 2 * y0norm / diameter - 1;

		float r0 = sqrt(x0norm*x0norm + y0norm*y0norm);
		if (isnan(r0) || r0 > 1.0)
		{
			continue;
		}
		float theta = atan2(y0norm, x0norm);
		float phi = (float)reverseSunex((double)r0);
		//glm::vec3 P = z * cos(phi) * glm::vec3(sin(phi) * cos(theta), sin(phi) * sin(theta), cos(phi));
		glm::vec3 P = z * glm::vec3(sin(phi) * cos(theta), sin(phi) * sin(theta), cos(phi));
		
		// undo reference camera rotation
		P = rotq(quat0, P);
		
		// translate point using camera shifts
		P = P - (position1 - position0);
		//P = P - (position0 - position1);

		// rotate to the secondary camera
		P = rotq(quat1, P);

		float phi1 = atan2(sqrt(P.x*P.x + P.y*P.y), P.z);
		if (isnan(phi1) || abs(phi1) > fov)
		{
			continue;
		}
		float r1 = (float)sunex(phi1);
		float theta1 = atan2(P.y, P.x);

		float x1 = r1 * cos(theta1);
		float y1 = r1 * sin(theta1);

		x1 = diameter*(x1 + 1) / 2;
		y1 = diameter*(y1 + 1) / 2;

		x1 = x1 + (center1.x - diameter * 0.5 - leftMargin);
		y1 = y1 + (center1.y - diameter * 0.5 - topMargin);

		const int x1f = std::floor(x1);
		const int y1f = std::floor(y1);
		//for (int c = 0; c < colors; c++)
		//{
		//	Desired0(x0, y0, c) = Image1(x1f, y1f,c);
		//}
		const float dx = x1 - x1f;
		const float dy = y1 - y1f;

		#pragma omp parallel for
		for (int c = 0; c<colors; c++)
		{
			Desired0(x0, y0, c) = Image1(LIM(x1f, width), LIM(y1f, height), c)*(1 - dx)*(1 - dy)
				+ Image1(LIM(x1f + 1, width), LIM(y1f, height), c)*dx*(1 - dy)
				+ Image1(LIM(x1f, width), LIM(y1f + 1, height), c)*(1 - dx)*dy
				+ Image1(LIM(x1f + 1, width), LIM(y1f + 1, height), c)*dx*dy;
		}		
	}
	
}


void mexFunction(int nout, mxArray* output[], int in, const mxArray* input[])
{
#ifndef _DEBUG
	omp_set_num_threads(std::max(4, omp_get_max_threads()));
	omp_set_dynamic(std::max(2, omp_get_max_threads() / 2));
#endif

	const int _offset = 3; // 5 obligatory params goes before variable number of cameras
	#define PN 3
	if (in < _offset + PN * 2 || nout != 1 || mxGetClassID(input[3]) != mxSINGLE_CLASS || mxGetClassID(input[4]) != mxSINGLE_CLASS || mxGetClassID(input[5]) != mxSINGLE_CLASS)
	{
		mexPrintf("Depth Estimation for a reference camera\n");
		mexErrMsgTxt("USAGE: [Image0] = fisheye_cost(minZ, maxZ, layers, sinlge(Image0), single(center0), single(quat&pos0), Image1, center1, quat&pos1);");
	}

	const int cameras = (in - _offset) / PN;

	if (cameras < 2)
	{
		mexErrMsgTxt("At least two camera frames (with 'center' and 'quat&pos' parameters) must be provided!");
	}

	if ((in - _offset) % PN)
	{
		mexErrMsgTxt("Each input camera must be defined by 3 SINGLE-valued parametes: ImageX, centerX, quat&posX!");
	}

	const float minZ = static_cast<float>(mxGetScalar(input[0]));
	const float maxZ = static_cast<float>(mxGetScalar(input[1]));
	const int layers = static_cast<int>(mxGetScalar(input[2]));

	const int height = mxGetDimensions(input[_offset])[0];
	const int width = mxGetDimensions(input[_offset])[1];
	const int colors = mymex::mxGetLayers(input[_offset]);
	const long HW = width*height;
	const int r = 2;
	//! pre-check the input data
	for (int n = 0; n<cameras; n++)
	{
		if (mxGetClassID(input[_offset + n * PN]) != mxSINGLE_CLASS)
		{
			mexPrintf("Camera Image %d must be of a SINGLE-type", n + 1);
			mexErrMsgTxt("\n");
		}

		if ((mxGetDimensions(input[_offset + n * PN]))[0] != height || (mxGetDimensions(input[_offset + n * PN]))[1] != width || mymex::mxGetLayers(input[_offset + n * PN]) != colors)
		{
			mexErrMsgTxt("All Camera Images must have the same resolution and three colors!");
		}

		if ((mxGetClassID(input[_offset + n * PN + 1]) != mxSINGLE_CLASS))
		{
			mexPrintf("Principal point position (center%d) must be array[2] of a SINGLE-type", n);
			mexErrMsgTxt("\n");
		}

		const size_t cN = (mxGetDimensions(input[_offset + n * PN + 1]))[0];
		const size_t cM = (mxGetDimensions(input[_offset + n * PN + 1]))[1];
		if (cN * cM != 2)
		{
			mexPrintf("Principal point position (center%d) must be array[2] of a SINGLE-type", n);
			mexErrMsgTxt("\n");
		}


		if ((mxGetClassID(input[_offset + n * PN + 2]) != mxSINGLE_CLASS))
		{
			mexPrintf("Quaterion 'quat&pos%d' must be array[4+3] of a SINGLE-type", n);
			mexErrMsgTxt("\n");
		}

		const size_t qN = (mxGetDimensions(input[_offset + n * PN + 2]))[0];
		const size_t qM = (mxGetDimensions(input[_offset + n * PN + 2]))[1];

		if (qN * qM != 7)
		{
			mexPrintf("Quaterion 'quat&pos%d' must be array[4+3] of a SINGLE-type", n);
			mexErrMsgTxt("\n");
		}
	}

	std::unique_ptr<MexImage<float>*[]> Images(new MexImage<float>*[cameras]);
	std::unique_ptr<glm::mat4x3[]> C(new glm::mat4x3[cameras]);
	std::unique_ptr<glm::vec4[]> Quats(new glm::vec4[cameras]);
	//std::unique_ptr<center[]> Centers(new center[cameras]);
	std::unique_ptr<glm::vec2[]> Centers(new glm::vec2[cameras]);
	std::unique_ptr<glm::vec3[]> Positions(new glm::vec3[cameras]);

	
	
#pragma omp parallel for
	for (int n = 0; n<cameras; n++)
	{
		Images[n] = new MexImage<float>(input[_offset + n * PN]);
		const float * const centerX = (float*)mxGetData(input[_offset + n * PN + 1]);
		const float * const quatX = (float*)mxGetData(input[_offset + n * PN + 2]);
		Centers[n] = glm::vec2(centerX[0], centerX[1]);
		Quats[n] = glm::vec4(quatX[0], quatX[1], quatX[2], quatX[3]);
		Positions[n] = glm::vec3(quatX[4], quatX[5], quatX[6]);
	}

	//const glm::mat4 Cdinv = glm::inverse(glm::mat4(C[0]));
	const glm::vec4 invQuat = glm::normalize(glm::vec4(-Quats[0].x, -Quats[0].y, -Quats[0].z, Quats[0].w));

	const float nan = sqrt(-1.f);
	const mwSize dims[] = { (unsigned)height, (unsigned)width, layers };

	//Matlab-allocated variables	
	output[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);

	MexImage<float> Cost(output[0]);
	Cost.setval(FLT_MAX);


	// for each depth-layer in the space
#pragma omp parallel 
	{
		MexImage<float> ProjTmpl(width, height, colors);
		

#pragma omp for		
		for (int d = 0; d < layers; d++)
		{
//#ifdef LINEAR_DEPTH_QUANTIZATION
			//const float z = minZ + d*(maxZ - minZ) / (layers - 1);
//#else 
			const float z = 1.f / ((float(d) / (layers-1))*(1.f / minZ - 1.f / maxZ) + 1.f / maxZ);
//#endif			

			MexImage<float> &ImageRef = *Images[0];

			for (int n = 1; n < cameras; n++)
			{
				MexImage<float> &ImageTempl = *Images[n];
				//ProjTmpl.setval(nan);
				projectFisheyeImage(ProjTmpl, invQuat, Centers[0], Positions[0], ImageTempl, Quats[n], Centers[n], Positions[n], z);
				
				#pragma omp parallel for
				for (long i = 0; i < HW; i++)
				{
					const int x = i / height;
					const int y = i % height;

					if (isnan(ProjTmpl[i]) || isnan(ImageRef[i]))
					{
						Cost[i] = FLT_MAX;
						continue;
					}

					float min_diff = FLT_MAX;

					for (int x1 = std::max(x - r, 0); x1 <= std::min(x + r, width - 1); x1++)
					{
						for (int y1 = std::max(y - r, 0); y1 <= std::min(y + r, height - 1); y1++)
						{
							float diff = 0.f;
							for (int c = 0; c < colors; c++)
							{
								diff += abs(ImageRef(x,y,c) - ProjTmpl(x1,y1,c));
							}
							min_diff = diff < min_diff ? diff : min_diff;
						}

					}					
					//Cost[i + d*HW] = diff / 3;
					Cost[i + d*HW] = std::min(min_diff, Cost[i + d*HW]);
				}
			}
		}
	}

	for (int n = 0; n<cameras; n++)
	{
		delete Images[n];
	}

}

