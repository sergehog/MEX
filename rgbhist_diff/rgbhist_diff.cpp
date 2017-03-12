/** RGB histogram difference, taking into account equalization transform
*	@file rgbhist_diff.cpp
*	@date 28.01.2016
*	@author Sergey Smirnov <sergey.smirnov@tut.fi>
*/

#include "../common/meximage.h"
#include <float.h>
#include <cmath>
#include <atomic>
#include <algorithm>
#include <vector>
//#include <Eigen\Dense>

#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}
#endif // __cplusplus
#ifndef _DEBUG
#include <omp.h>
#endif

#ifdef WIN32
#define isnan _isnan
#endif

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")

#define IsNonZero(d) ((d)!=0.0)
#define colors 3

using namespace mymex;
//using namespace Eigen;

#define length size_t(256)*size_t(256)*size_t(256)

void mexFunction(int nout, mxArray* output[], int in, const mxArray* input[])
{
#ifndef _DEBUG
	omp_set_num_threads(std::max<int>(8, omp_get_max_threads()));
	omp_set_dynamic(std::max<int>(4, omp_get_max_threads() / 2));;
#endif	

	if (in != 3 || mxGetClassID(input[0]) != mxUINT8_CLASS || mxGetClassID(input[1]) != mxUINT8_CLASS || mxGetClassID(input[2]) != mxDOUBLE_CLASS)
	{
		mexErrMsgTxt("USAGE: [error] = rgbhist_diff(uint8(H1), uint8(H2), A);\n");
	}
	
	MexImage<uint8_t> H1(input[0]);
	MexImage<uint8_t> H2(input[1]);
	MexImage<double> A(input[2]);

	if (H1.width != 1 || H1.height != length || H2.width !=1 || H2.height != length)
	{
		mexErrMsgTxt("H1 and H2 must be full length 8-bits RGB histograms staked to a (256^3) uint8 vector \n");
	}

	if (A.width != 4 || A.height != 3)
	{
		mexErrMsgTxt("A must be a transformation matrix 3x4. \n");
	}
	
	double err = 0;
	for (int r2 = 0; r2 < 256; r2++)
	{
		for (int g2 = 0; g2 < 256; g2++)
		{
			for (int b2 = 0; b2 < 256; b2++)
			{
				const size_t index = 256 * 256 * r2 + 256 * g2 + b2;
				const double r1 = r2*A(0, 0) + g2*A(1, 0) + b2*A(2, 0) + A(3, 0);
				const double g1 = r2*A(0, 1) + g2*A(1, 1) + b2*A(2, 1) + A(3, 1);
				const double b1 = r2*A(0, 2) + g2*A(1, 2) + b2*A(2, 2) + A(3, 2);

				if (r1 < 0.0 || g1 < 0.0 || b1 < 0.0 || r1 >= 255.0 || g1 >= 255.0 || b1 >= 255.0)
				{
					err += H2[index];
					continue;
				}
				
				// try to figure out interpolated histogram value
				const int r1l = int(floor(r1));
				const int r1u = int(ceil(r1));
				const int g1l = int(floor(g1));
				const int g1u = int(ceil(g1));
				const int b1l = int(floor(b1));
				const int b1u = int(ceil(b1));
				
				const uint8_t a = H1[256 * 256 * r1l + 256 * g1l + b1l];
				const uint8_t b = H1[256 * 256 * r1u + 256 * g1l + b1l];

				const uint8_t c = H1[256 * 256 * r1l + 256 * g1u + b1l];
				const uint8_t d = H1[256 * 256 * r1u + 256 * g1u + b1l];

				const uint8_t e = H1[256 * 256 * r1l + 256 * g1l + b1u];
				const uint8_t f = H1[256 * 256 * r1u + 256 * g1l + b1u];

				const uint8_t g = H1[256 * 256 * r1l + 256 * g1u + b1u];
				const uint8_t h = H1[256 * 256 * r1u + 256 * g1u + b1u];

				double distance = 0; // L1 distance
				double weights = 0;
				double value = 0;

				distance = abs(r1 - r1l) + abs(g1 - g1l) + abs(b1 - b1l); 
				value += (distance < 1.0) ? a*(1.0 - distance) : 0;
				weights += (distance < 1.0) ? (1.0 - distance) : 0;

				distance = abs(r1 - r1u) + abs(g1 - g1l) + abs(b1 - b1l);
				value += (distance < 1.0) ? b*(1.0 - distance) : 0;
				weights += (distance < 1.0) ? (1.0 - distance) : 0;

				distance = abs(r1 - r1l) + abs(g1 - g1u) + abs(b1 - b1l);
				value += (distance < 1.0) ? c*(1.0 - distance) : 0;
				weights += (distance < 1.0) ? (1.0 - distance) : 0;

				distance = abs(r1 - r1u) + abs(g1 - g1l) + abs(b1 - b1l);
				value += (distance < 1.0) ? d*(1.0 - distance) : 0;
				weights += (distance < 1.0) ? (1.0 - distance) : 0;
				
				//
				distance = abs(r1 - r1l) + abs(g1 - g1l) + abs(b1 - b1u);
				value += (distance < 1.0) ? e*(1.0 - distance) : 0;
				weights += (distance < 1.0) ? (1.0 - distance) : 0;

				distance = abs(r1 - r1u) + abs(g1 - g1l) + abs(b1 - b1u);
				value += (distance < 1.0) ? f*(1.0 - distance) : 0;
				weights += (distance < 1.0) ? (1.0 - distance) : 0;

				distance = abs(r1 - r1l) + abs(g1 - g1u) + abs(b1 - b1u);
				value += (distance < 1.0) ? g*(1.0 - distance) : 0;
				weights += (distance < 1.0) ? (1.0 - distance) : 0;

				distance = abs(r1 - r1u) + abs(g1 - g1l) + abs(b1 - b1u);
				value += (distance < 1.0) ? h*(1.0 - distance) : 0;
				weights += (distance < 1.0) ? (1.0 - distance) : 0;
				
				// difference between H2 histogram and interpolated H1 value
				err += abs(H2[index] - (value / weights));
			}
		}
	}	

	output[0] = mxCreateDoubleScalar(err);	

}
