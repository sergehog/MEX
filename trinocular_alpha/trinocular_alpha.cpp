/** trinocular_alpha
* @file trinocular_alpha.cpp
* @author Sergey Smirnov <sergey.smirnov@tut.fi>
* @date 2.09.2015
* @copyright 3D Media Group / Tampere University of Technology
*/


#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}
#endif // __cplusplus

#include "../common/meximage.h"
//#include "../common/defines.h"
#include <vector>
#define _USE_MATH_DEFINES
#include <cmath>
#include <algorithm>
#include <memory>
#include <array>
#ifndef _DEBUG
#include <omp.h>
#endif

//#define M_PI       3.14159265358979323846

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")

//typedef unsigned char uint8;
using namespace mymex;

void fastGaussian(const MexImage<float> &Cost, const MexImage<float> &Temporal, const float sigma)
{
	const int width = Cost.width;
	const int height = Cost.height;

	// has to be initialized with zeros
	Temporal.setval(0.f);

	// Horisontal (left-to-right & right-to-left) pass
	#pragma omp parallel for
	for (int y = 0; y<height; y++)
	{
		// left-to-right accumulation buffer
		std::unique_ptr<float[]> temporal1 = std::unique_ptr<float[]>(new float[width]);

		// right-to-left accumulation buffer
		std::unique_ptr<float[]> temporal2 = std::unique_ptr<float[]>(new float[width]);

		temporal1[0] = Cost(0, y);
		temporal2[width - 1] = Cost(width - 1, y);
		for (int x1 = 1; x1<width; x1++)
		{
			const int x2 = width - x1 - 1;
			temporal1[x1] = (Cost(x1, y) + temporal1[x1 - 1] * sigma);
			temporal2[x2] = (Cost(x2, y) + temporal2[x2 + 1] * sigma);
		}
		for (int x = 0; x<width; x++)
		{
			Temporal(x, y) = (temporal1[x] + temporal2[x]) - Cost(x, y);
		}
	}

	// Vertical (up-to-down & down-to-up) pass
	#pragma omp parallel for
	for (int x = 0; x<width; x++)
	{
		std::unique_ptr<float[]> temporal1 = std::unique_ptr<float[]>(new float[height]);
		std::unique_ptr<float[]> temporal2 = std::unique_ptr<float[]>(new float[height]);
		temporal1[0] = Temporal(x, 0);
		temporal2[height - 1] = Temporal(x, height - 1);
		for (int y1 = 1; y1<height; y1++)
		{
			const int y2 = height - y1 - 1;
			temporal1[y1] = (Temporal(x, y1) + temporal1[y1 - 1] * sigma);
			temporal2[y2] = (Temporal(x, y2) + temporal2[y2 + 1] * sigma);
		}
		for (int y = 0; y<height; y++)
		{
			Cost(x, y) = (temporal1[y] + temporal2[y]) - Temporal(x, y);
		}
	}
}



void mexFunction(const int nout, mxArray* output[], const int in, const mxArray* input[])
{
#ifndef _DEBUG
	omp_set_num_threads(std::max<int>(4, omp_get_max_threads()));
	omp_set_dynamic(std::max<int>(2, omp_get_max_threads() / 2));
#endif

	if (in != 6  || nout != 4 || mxGetClassID(input[0]) != mxUINT8_CLASS || mxGetClassID(input[1]) != mxUINT8_CLASS || mxGetClassID(input[2]) != mxUINT8_CLASS)
	{
		mexErrMsgTxt("USAGE: [Alpha, DispF, DispB, ReconstructedC] = trinocular_alpha(uint8(Left), uint8(Central), uint8(Right), mindisp, maxdisp, sigma);");
	}

	MexImage<uint8_t> Left(input[0]);
	MexImage<uint8_t> Central(input[1]);
	MexImage<uint8_t> Right(input[2]);

	const int mindisp = (int)mxGetScalar(input[3]);
	const int maxdisp = (int)mxGetScalar(input[4]);
	const float sigma = in > 5 ? std::min<float>(1.f, std::max<float>(0.f, (float)mxGetScalar(input[5]))) : 0.8f;
	
	//const float a = in > 5 ? (float)mxGetScalar(input[5]) : 10.f;
	//const float b = in > 6 ? (float)mxGetScalar(input[6]) : 2.f;
	const int layers = maxdisp - mindisp + 1;
	const int width = Left.width;
	const int height = Left.height;
	const int colors = Left.layers;
	const long HW = Left.layer_size;
	const float nan = sqrt(-1.f);
	//const int diameter = radius * 2 + 1;
	//const int window = diameter * diameter;

	const size_t dims1[] = { (size_t)height, (size_t)width, 1 };
	const size_t dims11[] = { (size_t)height, (size_t)width, (size_t)layers + 1 };
	const size_t dims3[] = { (size_t)height, (size_t)width, colors };
	output[0] = mxCreateNumericArray(3, dims1, mxSINGLE_CLASS, mxREAL);
	output[1] = mxCreateNumericArray(3, dims1, mxINT32_CLASS, mxREAL);
	output[2] = mxCreateNumericArray(3, dims1, mxINT32_CLASS, mxREAL);
	output[3] = mxCreateNumericArray(3, dims3, mxSINGLE_CLASS, mxREAL);
	

	MexImage<float> Alpha(output[0]);
	MexImage<uint32_t> DispF(output[1]);
	MexImage<uint32_t> DispB(output[2]);
	MexImage<float> Reconstructed(output[3]);

	//MexImage<float> CostAlpha(output[2]);	
	//MexImage<float> CostError(output[3]);
	//MexImage<float> CostWeights(width, height, layers);
	//CostWeights.setval(0.f);
	//CostAlpha.setval(0.f);
	//CostError.setval(0.f);
	//CostL.setval(0.f)
	MexImage<float> ErrorL(width, height);
	MexImage<float> ErrorR(width, height);
	MexImage<float> TmpAlphaL(width, height);
	MexImage<float> TmpAlphaR(width, height);
	MexImage<float> BestError(width, height);
	MexImage<float> TemporalL(width, height);
	MexImage<float> TemporalR(width, height);
	//MexImage<float> BestErrorR(width, height);
	BestError.setval(FLT_MAX);
	//BestErrorR.setval(FLT_MAX);

	for (int fi = 0; fi < layers; fi++)
	{
		const int dispFg = maxdisp - fi;
		for (int bi = fi + 1; bi < layers; bi++)
		{
			const int dispBg = maxdisp - bi;
			#pragma omp parallel for
			for (long i = 0; i < HW; i++)
			{
				// central camera coordinates
				const int x = i / height;
				const int y = i % height;
				
				// left camera coordinates
				const int xLf = x + dispFg < 0 ? 0 : (x + dispFg >= width ? width - 1 : x + dispFg);
				const int xLb = x + dispBg < 0 ? 0 : (x + dispBg >= width ? width - 1 : x + dispBg);
				
				// right camera coordinates
				const int xRf = x - dispFg < 0 ? 0 : (x - dispFg >= width ? width - 1 : x - dispFg);
				const int xRb = x - dispBg < 0 ? 0 : (x - dispBg >= width ? width - 1 : x - dispBg);

				float alphaL = 0.f;
				float alphaR = 0.f;

				for (int c = 0; c < colors; c++)
				{
					const float aL = abs((float(Central(x, y, c)) - float(Left(xLb, y, c))) / (float(Left(xLf, y, c)) - float(Left(xLb, y, c))));
					//const float aL = abs((float(Right(xRf, y, c)) - float(Left(xLb, y, c))) / (float(Left(xLf, y, c)) - float(Left(xLb, y, c))));
					alphaL += aL < 0.f ? 0.f : aL > 1.f ? 1.f : aL;

					const float aR = abs((float(Central(x, y, c)) - float(Right(xRb, y, c))) / (float(Right(xRf, y, c)) - float(Right(xRb, y, c))));
					//const float aR = abs((float(Left(xLf, y, c)) - float(Right(xRb, y, c))) / (float(Right(xRf, y, c)) - float(Right(xRb, y, c))));
					alphaR += aR < 0.f ? 0.f : aR > 1.f ? 1.f : aR;					
				}
				alphaL /= colors;
				alphaR /= colors;
				
				TmpAlphaL[i] = alphaL;
				TmpAlphaR[i] = alphaR;

				float errorL = 0;
				float errorR = 0;
				for (int c = 0; c < colors; c++)
				{
					errorL += abs(float(Central(x, y, c)) - alphaL * float(Left(xLf, y, c)) - (1 - alphaL) * float(Left(xLb, y, c)));
					errorR += abs(float(Central(x, y, c)) - alphaR * float(Right(xRf, y, c)) - (1 - alphaR) * float(Right(xRb, y, c)));					
					//errorL += abs(float(Right(xRf, y, c)) - alphaL * float(Left(xLf, y, c)) - (1 - alphaL) * float(Left(xLb, y, c)));
					//errorR += abs(float(Left(xLf, y, c)) - alphaR * float(Right(xRf, y, c)) - (1 - alphaR) * float(Right(xRb, y, c)));
				}

				ErrorL[i] = errorL / colors;
				ErrorR[i] = errorR / colors;
			}

			
			#pragma omp parallel sections
			{
				#pragma omp section
				{
					fastGaussian(ErrorL, TemporalL, sigma);
				}
				#pragma omp section
				{
					fastGaussian(ErrorR, TemporalR, sigma);
				}
			}
						
			#pragma omp parallel for
			for (long i = 0; i < HW; i++)
			{
				const int x = i / height;
				const int y = i % height;
				const int xLf = x + dispFg < 0 ? 0 : (x + dispFg >= width ? width - 1 : x + dispFg);
				const int xLb = x + dispBg < 0 ? 0 : (x + dispBg >= width ? width - 1 : x + dispBg);
				const int xRf = x - dispFg < 0 ? 0 : (x - dispFg >= width ? width - 1 : x - dispFg);
				const int xRb = x - dispBg < 0 ? 0 : (x - dispBg >= width ? width - 1 : x - dispBg);
				
				const float errorL = ErrorL[i];
				if (errorL < BestError[i])
				{
					BestError[i] = errorL;
					Alpha[i] = TmpAlphaL[i];
					DispF[i] = dispFg;
					DispB[i] = dispBg;
					for (int c = 0; c < colors; c++)
					{
						Reconstructed(x, y, c) = TmpAlphaL[i] * float(Left(xLf, y, c)) + (1 - TmpAlphaL[i])* float(Left(xLb, y, c));
					}
				}

				const float errorR = ErrorR[i];
				if (errorR < BestError[i])
				{
					BestError[i] = errorR;
					Alpha[i] = TmpAlphaR[i];
					DispF[i] = dispFg;
					DispB[i] = dispBg;
					for (int c = 0; c < colors; c++)
					{
						Reconstructed(x, y, c) = TmpAlphaR[i] * float(Right(xRf, y, c)) + (1 - TmpAlphaR[i])* float(Right(xRb, y, c));
					}
				}

			}
		}
	}
}
