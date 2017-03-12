/** 
* @file fisheye_project.cpp
* @author Sergey Smirnov <sergey.smirnov@tut.fi>
* @date 30.03.2015
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


	if (in < 6 || nout != 1 || mxGetClassID(input[1]) != mxSINGLE_CLASS || mxGetClassID(input[2]) != mxSINGLE_CLASS || mxGetClassID(input[3]) != mxSINGLE_CLASS || mxGetClassID(input[4]) != mxSINGLE_CLASS || mxGetClassID(input[5]) != mxSINGLE_CLASS)
	{
		mexPrintf("Depth Estimation for a reference camera\n");
		mexErrMsgTxt("USAGE: [Image0] = fisheye_project(Z0, single(center0), single(quat&pos0), Image1, center1, quat&pos1);");
	}

	const float Z = static_cast<float>(mxGetScalar(input[0]));
	MexImage<float> Image1(input[3]);
	const int height = Image1.height;
	const int width = Image1.width;
	const int colors = Image1.layers;
	const long HW = width*height;

	const float * const center0 = (float*)mxGetData(input[1]);
	const float * const quat0 = (float*)mxGetData(input[2]);
	const float * const center1 = (float*)mxGetData(input[4]);
	const float * const quat1 = (float*)mxGetData(input[5]);

	glm::vec2 Center0 = glm::vec2(center0[0], center0[1]);	
	glm::vec4 Quat0 = glm::vec4(quat0[0], quat0[1], quat0[2], quat0[3]);
	glm::vec3 Position0 = glm::vec3(quat0[4], quat0[5], quat0[6]);

	glm::vec2 Center1 = glm::vec2(center1[0], center1[1]);
	glm::vec4 Quat1 = glm::vec4(quat1[0], quat1[1], quat1[2], quat1[3]);
	glm::vec3 Position1 = glm::vec3(quat1[4], quat1[5], quat1[6]);


	const glm::vec4 invQuat = glm::normalize(glm::vec4(-Quat0.x, -Quat0.y, -Quat0.z, Quat0.w));

	const float nan = sqrt(-1.f);
	const mwSize dims[] = { (unsigned)height, (unsigned)width, colors };

	//Matlab-allocated variables	
	output[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);

	MexImage<float> Image0(output[0]);
	Image0.setval(nan);

	projectFisheyeImage(Image0, invQuat, Center0, Position0, Image1, Quat1, Center1, Position1, Z);
}
