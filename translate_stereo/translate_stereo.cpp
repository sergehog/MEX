/** 
*	@file translate_stereo.cpp
*	@date 27.04.2012
*	@author Sergey Smirnov <sergey.smirnov@tut.fi>
*/

#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}
#endif // __cplusplus
//#include "../common/defines.h"
#include "../common/meximage.h"
#include <cmath>
#include <algorithm>
#include <memory>

typedef unsigned char uint8;
using namespace mymex;

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")


void fast_filtering(const MexImage<float> &Signal, const MexImage<float> &Temporal, const float sigma)
{
	const int width = Signal.width;
	const int height = Signal.height;
	const int layers = Signal.layers;
	const long HW = Signal.layer_size;
	std::unique_ptr<float[]> temporal1(new float[std::max(width, height)]);
	std::unique_ptr<float[]> temporal2(new float[std::max(width, height)]);

	for (int l = 0; l<layers; l++)
	{
		// Temporal.setval(0.f);
		// Horisontal (left-to-right & right-to-left) pass
		for (int y = 0; y<height; y++)
		{
			temporal1[0] = Signal(0, y, l);
			temporal2[width - 1] = Signal(width - 1, y, l);
			for (int x1 = 1; x1<width; x1++)
			{
				const int x2 = width - x1 - 1;
				temporal1[x1] = (Signal(x1, y, l) + temporal1[x1 - 1] * sigma);
				temporal2[x2] = (Signal(x2, y, l) + temporal2[x2 + 1] * sigma);
			}
			for (int x = 0; x<width; x++)
			{
				Temporal(x, y) = (temporal1[x] + temporal2[x]) - Signal(x, y, l);
			}
		}

		// Vertical (up-to-down & down-to-up) pass
		for (int x = 0; x<width; x++)
		{
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
				Signal(x, y, l) = (temporal1[y] + temporal2[y]) - Temporal(x, y);
			}
		}
	}
}


//template<typename T> void render(T*, float*, T*, float*, bool*, size_t, size_t, size_t);

const float skip_delta = 3;
const unsigned skip_numb = 1;

void mexFunction(int nout, mxArray* output[], int in, const mxArray* input[])
{
	if (in < 4 || in > 8 || nout != 2
		|| mxGetClassID(input[0]) != mxUINT8_CLASS || mxGetClassID(input[1]) != mxUINT8_CLASS
		|| mxGetClassID(input[2]) != mxSINGLE_CLASS || mxGetClassID(input[3]) != mxSINGLE_CLASS)
	{
		mexErrMsgTxt("USAGE: [Virtual, Disp] = translate_stereo(uint8(L), uint8(R), single(DispL), single(DispR), <position=0...1, radius, sigma_distance, fill_holes>); ");
	}

	MexImage<uint8> Left(input[0]);
	MexImage<uint8> Right(input[1]);
	MexImage<float> DispL(input[2]);
	MexImage<float> DispR(input[3]);

	const int height = Left.height;
	const int width = Left.width;
	const long HW = Left.layer_size;
	const int colors = Left.layers;
	const float nan = sqrt(-1.f);
	const float disp_thr = 2.f;
	if (height != DispL.height || width != DispL.width || height != DispR.height || width != DispR.width)
	{
		mexErrMsgTxt("ERROR: Sizes of all images must be the same.");
	}
	if (height != Right.height || width != Right.width || colors != Right.layers)
	{
		mexErrMsgTxt("ERROR: Sizes of all images must be the same.");
	}

	const float position = (in > 4) ? std::max(-1.f, std::min(2.f, (float)mxGetScalar(input[4]))) : 0.5f;
	const unsigned radius = (in > 5) ? (unsigned)mxGetScalar(input[5]) : 1;
	const float sigma = (in > 6) ? (float)mxGetScalar(input[6]) : 0.8f;
	const bool fill_holes = (in > 7) ? (bool)mxGetScalar(input[7]) : true;
	
	size_t dims[] = { (unsigned)height, (unsigned)width, (unsigned)colors };
	size_t dims2d[] = { (unsigned)height, (unsigned)width, 1 };

	output[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
	output[1] = mxCreateNumericArray(3, dims2d, mxSINGLE_CLASS, mxREAL);

	MexImage<float> Rendered(output[0]);
	MexImage<float> Disparity(output[1]);

	if (position == 0.f)
	{
		for (int i = 0; i < HW; i++)
		{
			Disparity[i] = DispL[i];
			for (int c = 0; c < colors; c++)
			{
				Rendered[i + c*HW] = Left[i + c*HW];
			}
		}
		return;
	}

	if (position == 1.f)
	{
		for (int i = 0; i < HW; i++)
		{
			Disparity[i] = DispR[i];
			for (int c = 0; c < colors; c++)
			{
				Rendered[i + c*HW] = Right[i + c*HW];
			}
		}
		return;
	}


	Disparity.setval(nan);

	// check for mindisp/maxdisp
	float mindisp = DispL[0];
	float maxdisp = DispL[0];
	#pragma omp for
	for (int i = 1; i < HW; i++)
	{
		mindisp = DispL[i] < mindisp ? DispL[i] : mindisp;
		maxdisp = DispL[i] > maxdisp ? DispL[i] : maxdisp;
		mindisp = DispR[i] < mindisp ? DispR[i] : mindisp;
		maxdisp = DispR[i] > maxdisp ? DispR[i] : maxdisp;
	}

	// Left -> Virtual pass
	#pragma omp for
	for (long i = 0; i < HW; i++)
	{
		const int x = i / height;
		const int y = i % height;
		const float disp = DispL[i];
		if (isnan(disp))
		{
			continue;
		}

		const int xv = std::round(x - disp*position);
		if (xv < 0 || xv >= width)
		{
			continue;
		}
		const float dispv = Disparity(xv, y);
		if (isnan(dispv) || dispv < disp)
		{
			Disparity(xv, y) = disp;
		}
	}

	// Right -> Virtual pass
	#pragma omp for
	for (long i = 0; i < HW; i++)
	{
		const int x = i / height;
		const int y = i % height;
		const float disp = DispR[i];
		if (isnan(disp))
		{
			continue;
		}

		const int xv = std::round(x + disp*(1 - position));
		if (xv < 0 || xv >= width)
		{
			continue;
		}
		const float dispv = Disparity(xv, y);
		if (isnan(dispv) || dispv < disp)
		{
			Disparity(xv, y) = disp;
		}
	}

	// fill holes in the virtual disparity map
	MexImage<float> Temporal3(width, height, 3);
	MexImage<float> Weights(width, height);
	MexImage<float> TemporalC(width, height);

	if (fill_holes)
	{
		MexImage<float> DispTmp(Temporal3, 0);
		
#pragma omp for
		for (long i = 0; i < HW; i++)
		{
			// background-favored weighting scheme
			Weights[i] = isnan(Disparity[i]) ? 0 : (maxdisp - Disparity[i]);
			DispTmp[i] = isnan(Disparity[i]) ? 0 : Disparity[i] * Weights[i];
		}
		fast_filtering(DispTmp, TemporalC, sigma);
		fast_filtering(Weights, TemporalC, sigma);
#pragma omp for
		for (long i = 0; i < HW; i++)
		{
			// predict disparity if absent
			Disparity[i] = isnan(Disparity[i]) ? DispTmp[i] / Weights[i] : Disparity[i];
		}
	}
	Rendered.setval(nan);
	Weights.setval(nan);

#pragma omp for
	for (long i = 0; i < HW; i++)
	{
		const int x = i / height;
		const int y = i % height;
		const float disp = Disparity[i];

		const float xl = x + disp*position;
		if (!isnan(disp) && xl > 0.f && xl <= float(width - 2))
		{
			const int xl1 = std::floor(xl);
			const float wl = (xl - xl1);
			const float dispL = DispL(xl1, y)*(1 - wl) + DispL(xl1 + 1, y)*wl;
			const float w = abs(disp - dispL);
			if (dispL < disp + 1)
			{
				Weights(x, y) = w;
				for (int c = 0; c < colors; c++)
				{
					Rendered(x, y, c) = (Left(xl1, y, c)*(1 - wl) + Left(xl1 + 1, y, c)*wl);
				}
			}
		}

		const float xr = x - disp*(1 - position);
		if (xr > 0.f && xr <= float(width - 2))
		{
			const int xr1 = std::floor(xr);
			const float wr = (xr - xr1);
			const float dispR = DispR(xr1, y)*(1 - wr) + DispR(xr1 + 1, y)*wr;
			const float w = abs(disp - dispR);
			if (dispR < disp + 1)
				//if (w<1)
				if (isnan(Weights(x, y)) || w < Weights(x, y))
				{
					Weights(x, y) = w;
					for (int c = 0; c < colors; c++)
					{
						Rendered(x, y, c) = (Right(xr1, y, c)*(1 - wr) + Right(xr1 + 1, y, c)*wr);
					}
				}
		}
	}
	if (fill_holes)
	{
		MexImage<float> ToFilter(width, height, colors);

#pragma omp for
		for (long i = 0; i < HW; i++)
		{
			if (isnan(Weights[i]))
			{
				for (int c = 0; c < colors; c++)
				{
					ToFilter[i + c*HW] = 0;
				}
				Weights[i] = 0;
			}
			else
			{
				Weights[i] = (maxdisp - Disparity[i]);
				for (int c = 0; c < colors; c++)
				{
					ToFilter[i + c*HW] = Rendered[i + c*HW] * Weights[i];
				}
			}
		}
		fast_filtering(Weights, TemporalC, sigma);
		fast_filtering(ToFilter, Temporal3, sigma);

#pragma omp for
		for (long i = 0; i < HW; i++)
		{
			if (isnan(Rendered[i]))
			{
				for (int c = 0; c < colors; c++)
				{
					Rendered[i + c*HW] = ToFilter[i + c*HW] / Weights[i];
				}
			}
		}
	}
	else
	{
		#pragma omp for
		for (long i = 0; i < HW; i++)
		{
			if (isnan(Rendered[i]))
			{
				for (int c = 0; c < colors; c++)
				{
					Rendered[i + c*HW] /= Weights[i];
				}
			}
		}
	}

	//#pragma omp for
	//for (long i = 0; i < HW; i++)
	//{
	//	// if (!isnan(Weights[i]))
	//	{
	//		for (int c = 0; c < colors; c++)
	//		{
	//			Rendered[i + c*HW] /= 2;// Weights[i];
	//		}
	//	}
	//}
	//// Left->Virtual pass
	//for(int x=width-1; x>=0; x--)
	//{
	//	for(int y=0; y<height; y++)
	//	{
	//		long index = Left.Index(x, y);
	//		float disp = DispL[index];
	//		if(_isnan(disp))
	//			continue;
	//		float xVirt = x - disp*position;
	//		int xv = mymex::round(xVirt);
	//		for(int xj=xv-(int)radius; xj<=xv+(int)radius; xj++)
	//		{
	//			if(xj<0 || xj>=width)
	//				continue;
	//			long indexj = Virtual.Index(xj, y);
	//			
	//			float weight = exp(-(xj - xVirt)*(xj - xVirt)/sigma_distance);

	//			float currD = Disparity[indexj];
	//			float currW = Weights[indexj];
	//			if(_isnan(currD) || ((disp-currD)>disp_thr))// && weight > currW))			
	//			{
	//				Weights[indexj] = weight;
	//				for(int c=0; c<colors; c++)
	//				{
	//					long cHW = c*HW;
	//					Virtual[indexj+cHW] = Left[index+cHW]*weight;
	//				}
	//				Disparity[indexj] = disp;
	//			}
	//			else if(abs(disp-currD) <= disp_thr)
	//			{
	//				Weights[indexj] += weight;
	//				for(int c=0; c<colors; c++)
	//				{
	//					long cHW = c*HW;
	//					Virtual[indexj+cHW] += Left[index+cHW]*weight;
	//				}
	//			}
	//			
	//		}
	//	}
	//}


	//// Right->Virtual pass
	//for(int x=0; x<width; x++)
	//{
	//	for(int y=0; y<height; y++)
	//	{
	//		long index = Right.Index(x, y);
	//		float disp = DispR[index];
	//		if(_isnan(disp))
	//			continue;
	//		float xVirt = x + disp*(1-position);
	//		int xv = mymex::round(xVirt);
	//		for(int xj=xv-(int)radius; xj<=xv+(int)radius; xj++)
	//		{
	//			if(xj<0 || xj>=width)
	//				continue;
	//			long indexj = Virtual.Index(xj, y);
	//			
	//			float weight = exp(-(xj - xVirt)*(xj - xVirt)/sigma_distance);

	//			float currD = Disparity[indexj];
	//			float currW = Weights[indexj];

	//			if(_isnan(currD) || ((disp-currD)>disp_thr))// && weight > currW))
	//			{
	//				Weights[indexj] = weight;
	//				for(int c=0; c<colors; c++)
	//				{
	//					long cHW = c*HW;
	//					Virtual.data[indexj+cHW] = Right[index+cHW]*weight;
	//				}
	//				Disparity[indexj] = disp;
	//			}
	//			else if(abs(disp-currD) <= disp_thr)
	//			{
	//				Weights[indexj] += weight;
	//				for(int c=0; c<colors; c++)
	//				{
	//					long cHW = c*HW;
	//					Virtual[indexj+cHW] += Right[index+cHW]*weight;
	//				}
	//			}				
	//		}
	//	}
	//}	

	//#pragma omp parallel for
	//for(long i=0; i<HW; i++)
	//{
	//	float weight = Weights[i];
	//	if(weight > 0.001f)
	//	{
	//		for(int c=0; c<colors; c++)
	//		{
	//			long cHW = c*HW;
	//			float value = Virtual[i + cHW];			
	//			Rendered.data[i + cHW] = std::floor(0.5 + value/weight);			
	//		}	
	//	}
	//	else if(predict && i != 0) // predict 
	//	{
	//		int y = i % height;
	//		int x = i / height;				

	//		float predict_value[5] = {0.f, 0.f, 0.f, 0.f, 0.f};

	//		int pixels = 0;
	//		if(x>0)
	//		{
	//			long j = Rendered.Index(x-1, y);
	//			pixels ++;
	//			for(int c=0; c<colors; c++)
	//			{
	//				long cHW = c*HW;
	//				predict_value[c] += Rendered[j + cHW];
	//			}
	//		}
	//		if(y>0)
	//		{
	//			long j = Rendered.Index(x, y-1);
	//			pixels ++;
	//			for(int c=0; c<colors; c++)
	//			{
	//				long cHW = c*HW;
	//				predict_value[c] += Rendered[j + cHW];
	//			}
	//		}
	//		if(x>0 && y>0)
	//		{
	//			long j = Rendered.Index(x-1, y-1);
	//			pixels ++;
	//			for(int c=0; c<colors; c++)
	//			{
	//				long cHW = c*HW;
	//				predict_value[c] += Rendered[j + cHW];
	//			}
	//		}
	//		for(int c=0; c<colors; c++)
	//		{
	//			long cHW = c*HW;
	//			Rendered.data[i + cHW] = std::round(predict_value[c]/pixels);
	//		}


	//		//long j = i > 0 ? i-1 : 0;
	//		//Rendered.data[i + cHW] = Rendered.data[j + cHW];
	//	}

	//}

}

