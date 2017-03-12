/** background_inpaint
* @file background_inpaint.cpp
* @author Sergey Smirnov <sergey.smirnov@tut.fi>
* @date 16.07.2015
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
#include <vector>
#ifndef _DEBUG
#include <omp.h>
#endif

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")

#define colors 3

//typedef unsigned char uint8;
using namespace mymex;
using namespace std;


void calculateSource(MexImage<bool> &Valid, MexImage<float> &IntegralConf, MexImage<bool> &Source, const int x0, const int y0, const int x1, const int y1, const int radius);
//void calculateAverage(MexImage<float> &Image, MexImage<bool> &Valid, MexImage<float> &Average, const int x0, const int y0, const int x1, const int y1, const int radius);
void calculateGradient(MexImage<float> &Image, MexImage<bool> &Valid, MexImage<float> &GradientX, MexImage<float> &GradientY, const int x0, const int y0, const int x1, const int y1);
void calculateNormal(MexImage<bool> &Valid, MexImage<int8_t> &NormalX, MexImage<int8_t> &NormalY, const int x0, const int y0, const int x1, const int y1);


void mexFunction(const int nout, mxArray* output[], const int in, const mxArray* input[])
{
#ifndef _DEBUG
	omp_set_num_threads(std::max<int>(4, omp_get_max_threads()));
	omp_set_dynamic(std::max<int>(2, omp_get_max_threads() / 2));
#endif

	if (in != 6 || nout != 4 || mxGetClassID(input[0]) != mxSINGLE_CLASS || mxGetClassID(input[1]) != mxSINGLE_CLASS || mxGetClassID(input[2]) != mxUINT8_CLASS || mxGetClassID(input[3]) != mxUINT8_CLASS)
	{
		mexErrMsgTxt("USAGE: [Lh, Rh, DLh, DRh] = background_inpaint(single(L), single(R), single(DL), single(DR), radius, search);\n");
	}

	MexImage<float> Left(input[0]);
	MexImage<float> Right(input[1]);
	MexImage<float> LeftDisp(input[2]);
	MexImage<float> RightDisp(input[3]);
		
	const int width = Left.width;
	const int height = Left.height;
	const long HW = Left.layer_size;

	if (Right.width != width || LeftDisp.width != width || RightDisp.width != width)
	{
		mexErrMsgTxt("Resolution mismatch!\n");
	}

	if (Right.height != width || LeftDisp.height != width || RightDisp.height != width)
	{
		mexErrMsgTxt("Resolution mismatch!\n");
	}

	if (Left.layers != colors || Right.layers != colors)
	{
		mexErrMsgTxt("Color mismatch!\n");
	}

	const int radius = std::max<unsigned>(1u, std::min<unsigned>(100, mxGetScalar(input[4])));
	const int _search = mxGetScalar(input[5]);
	const int search = _search == 0 ? std::max(width, height) : std::max<int>(2u, std::min<int>(100, _search));
	
	const unsigned maxiter = 10000;

	const float nan = sqrt(-1.f);
	const size_t dims3[] = { (size_t)height, (size_t)width, (size_t)colors };
	const size_t dims1[] = { (size_t)height, (size_t)width, 1 };
	output[0] = mxCreateNumericArray(3, dims3, mxSINGLE_CLASS, mxREAL);
	output[1] = mxCreateNumericArray(3, dims3, mxSINGLE_CLASS, mxREAL);
	output[2] = mxCreateNumericArray(3, dims1, mxSINGLE_CLASS, mxREAL);
	output[3] = mxCreateNumericArray(3, dims1, mxSINGLE_CLASS, mxREAL);

	MexImage<float> LeftOut(output[0]);
	MexImage<float> RightOut(output[1]);
	MexImage<float> LeftDispOut(output[2]);
	MexImage<float> RightDispOut(output[3]);
	
	MexImage<bool> ValidLeft(width, height);
	MexImage<bool> ValidRight(width, height);
	MexImage<float> ConfidenceLeft(width, height);

	#pragma omp parallel for
	for (long i = 0; i < HW; i++)
	{
		for (int c = 0; c < colors; c++)
		{
			LeftOut[i + HW*c] = Left[i + HW*c];
			RightOut[i + HW*c] = Right[i + HW*c];
		}

		LeftDispOut[i] = LeftDisp[i];
		RightDispOut[i] = RightDisp[i];
		ValidLeft[i] = !isnan(LeftDisp[i]);
		ValidRight[i] = !isnan(RightDisp[i]);
		ConfidenceLeft[i] = float(!isnan(LeftDisp[i]));
	}

	MexImage<float> DataTerm(width, height);
	MexImage<float> Average(width, height, colors);
	MexImage<float> GradientX(width, height);
	MexImage<float> GradientY(width, height);
	MexImage<float> IntegralConf(width, height);
	MexImage<int8_t> NormalX(width, height);
	MexImage<int8_t> NormalY(width, height);
	MexImage<bool> Source(width, height);

	// normals map (used to find boundary patches)	
	NormalX.setval(0);
	NormalY.setval(0);
	calculateNormal(ValidLeft, NormalX, NormalY, 0, 0, width - 1, height - 1);

	Average.setval(0.f);
	DataTerm.setval(0.f);
	Source.setval(false);
	GradientX.setval(0.f);
	GradientY.setval(0.f);

	IntegralConf.set(ConfidenceLeft);
	IntegralConf.IntegralImage(true);

	// fast binary map of valid patches	
	calculateSource(ValidLeft, IntegralConf, Source, 0, 0, width, height, radius);

	calculateGradient(Left, ValidLeft, GradientX, GradientY, 0, 0, width - 1, height - 1);

	unsigned iteration = 0;
	bool found_holes = true;
	long updated = 1;

	unsigned long long searches = 0;

	// process goes iteratively
	while (updated > 0 && (!maxiter || iteration < maxiter))
	{
		float highest_priority = 0.f;
		float best_conf = nan;
		long best_index = -1;

		// 1. find appropriate pixel, where to inpaint
		#pragma omp parallel for
		for (long i = 0; i < HW; i++)
		{
			const int x = i / height;
			const int y = i % height;

			if (ValidLeft[i] || (!NormalX[i] && !NormalY[i]))
			{
				continue;
			}

			const float conf = IntegralConf.getIntegralAverage(x, y, radius);

			float priority = conf;

			if (true)
			{
				float gx = GradientY[i];
				float gy = -GradientX[i];
				gx = isnan(gx) ? 0 : gx;
				gy = isnan(gy) ? 0 : gy;

				//float data = abs(gx*NormalX[i]) + abs(gy*NormalY[i]);
				float data = abs(gx*NormalX[i] + gy*NormalY[i]);
				data = data > 0.f ? data : 0.001f;
				//data = data > 0.001 ? data : 0.001;
				DataTerm.data[i] = data;
				priority *= data;
			}

			if (priority > highest_priority)
			{
				#pragma omp critical
				{
					highest_priority = priority;
					best_index = i;
				}
				best_conf = conf;
			}

		}

		if (best_index < 0)
		{
			mexPrintf("ERROR: No appropriate patch to inpaint was found :-(\n");
			break;
		}

		const int best_x = best_index / height;
		const int best_y = best_index % height;
		//mexPrintf("\t inpainting pixel: (%d, %d), conf = %3.2f\n", best_x, best_y, best_conf);

		// seek for the best valid patch over whole image o_O
		float best_diff = 1000000.f;
		long best_sindex = -1;
		//float best_avg = 1000000.f;				
		unsigned long checked = 0;
		unsigned long runned = 0;
		
		#pragma omp parallel for reduction(+:checked,runned)
		//for(long i=0; i<HW; i++)
		for (int x = std::max(radius, best_x - search); x <= std::min(width - 1 - radius, best_x + search); x++)
			for (int y = std::max(radius, best_y - search); y <= std::min(height - 1 - radius, best_y + search); y++)
			{
				runned++;
				long i = Source.Index(x, y);
				if (!Source[i])
					continue;


				float diff = 0;
				int pixels = 0;
				for (int dx = -radius; dx <= radius; dx++)
				{
					int sx = x + dx;
					int bx = best_x + dx;
					if (bx < 0 || bx >= width)
						continue;

					for (int dy = -radius; dy <= radius; dy++)
					{
						int sy = y + dy;
						int by = best_y + dy;

						if (by < 0 || by >= height)
							continue;

						long sindex = LeftOut.Index(sx, sy);
						long bindex = LeftOut.Index(bx, by);

						if (!ValidLeft[bindex])
						{
							continue;
						}

						for (int c = 0; c<colors; c++)
						{
							diff += abs(Left[sindex + c*HW] - LeftOut[bindex + c*HW]);
						}

						// Additional constraint which penalize usage of not-confident parts of the image while seeking for the best coressponding patch
						// However, effect of such penalizing is pretty minimal
						//if(Confidence->data[sindex])
						//{
						//	diff += 100*colors;
						//}

						pixels++;
					}
				}

				diff /= pixels;

				if (diff < best_diff)
				{
#pragma omp critical
				{
					best_diff = diff;
					best_sindex = i;
				}

				//if(earlyTerm && avgdiff < best_avg*2)
				//{
				//	#pragma omp critical
				//	{
				//		best_avg = avgdiff;
				//	}
				//}
				}

				checked++;
			}

		if (best_sindex < 0)
		{
			mexPrintf("ERROR: No appropriate match for patch at (%d, %d) was found (%d (%d) patches were checked, ) \n", best_x, best_y, checked, runned);
			break;
		}

		searches += checked;

		const int best_sx = best_sindex / height;
		const int best_sy = best_sindex % height;
		mexPrintf("\t best patch: (%d, %d), diff = %3.2f\n", best_sx, best_sy, best_diff);

		// update found hole with found patch
#pragma omp parallel for
		for (int dx = -radius; dx <= radius; dx++)
		{
			int sx = best_sx + dx;
			int bx = best_x + dx;
			if (bx < 0 || bx >= width)
				continue;

			for (int dy = -radius; dy <= radius; dy++)
			{
				int sy = best_sy + dy;
				int by = best_y + dy;

				if (by < 0 || by >= height)
					continue;

				long sindex = Left.Index(sx, sy);
				long bindex = Left.Index(bx, by);

				if (ValidLeft[bindex])
				{
					continue;
				}

				ValidLeft[bindex] = true;
				ConfidenceLeft[bindex] = best_conf;

				for (int c = 0; c<colors; c++)
				{
					LeftOut[bindex + c*HW] = Left[sindex + c*HW];
				}

				updated++;
			}
		}


		// update normals map
		calculateNormal(ValidLeft, NormalX, NormalY, std::max<int>(0, best_x - radius - 1), std::max<int>(0, best_y - radius - 1), std::min<int>(width - 1, best_x + radius + 1), std::min<int>(height - 1, best_y + radius + 1));

		
		calculateGradient(LeftOut, ValidLeft, GradientX, GradientY, std::max<int>(0, best_x - radius - 1), std::max<int>(0, best_y - radius - 1), std::min<int>(width - 1, best_x + radius + 1), std::min<int>(height - 1, best_y + radius + 1));

		//holes -= updated;		
		iteration++;

		for (long i = 0; i<HW; i++)
		{
			IntegralConf.data[i] = ConfidenceLeft[i];
		}

		IntegralConf.IntegralImage(true);
		//mexPrintf("Iteration %d; updated %d; checked %d; remaining holes %d\n", iteration, updated, checked, holes);
		mexPrintf("Iteration %d; updated %d; checked %d\n", iteration, updated, checked);
	}	
	//mexPrintf("Inpainted pixels: %d; using patches: %d; total matches: %d. \n", holes_saved-holes, iter, searches);
}


void calculateAverage(MexImage<float> &Image, MexImage<bool> &Valid, MexImage<float> &Average, const int x0, const int y0, const int x1, const int y1, const int radius)
{
	const int width = Image.width;
	const int height = Image.height;
	const int HW = Image.layer_size;
	//const int colors = Image.layers;

#pragma omp parallel for
	for (int x = std::max(0, x0); x <= std::min(width - 1, x1); x++)
	{
		for (int y = std::max(0, y0); y <= std::min(height - 1, y1); y++)
		{
			long index = Image.Index(x, y);
			for (int c = 0; c<colors; c++)
			{
				Average.data[index + c*HW] = 0;
			}
			int pixels = 0;

			for (int sx = std::max(0, x - radius); sx <= std::min(width - 1, x + radius); sx++)
			{
				for (int sy = std::max(0, y - radius); sy <= std::min(height - 1, y + radius); sy++)
				{
					long sindex = Average.Index(sx, sy);
					if (!Valid[sindex])
					{
						continue;
					}
					for (int c = 0; c<colors; c++)
					{
						Average.data[index + c*HW] += Image[sindex + c*HW];
					}
					pixels++;
				}
			}
			if (pixels > 0)
			{
				for (int c = 0; c<colors; c++)
				{
					Average.data[index + c*HW] /= pixels;
				}
			}
		}
	}
}

void calculateGradient(MexImage<float> &Image, MexImage<bool> &Valid, MexImage<float> &GradientX, MexImage<float> &GradientY, const int x0, const int y0, const int x1, const int y1)
{
	const int width = Image.width;
	const int height = Image.height;
	const int HW = Image.layer_size;
	//const int colors = Image.layers;

#pragma omp parallel for
	for (int x = std::max(0, x0); x <= std::min(width - 1, x1); x++)
	{
		for (int y = std::max(0, y0); y <= std::min(height - 1, y1); y++)
		{
			long index = Image.Index(x, y);

			if (!Valid[index])
				continue;

			GradientX.data[index] = 0;
			GradientY.data[index] = 0;

			int xl = std::max(0, x - 1);
			int xr = std::min(width - 1, x + 1);
			int yu = std::max(0, y - 1);
			int yd = std::min(height - 1, y + 1);

			long indexl = Image.Index(xl, y);
			long indexr = Image.Index(xr, y);
			long indexu = Image.Index(x, yu);
			long indexd = Image.Index(x, yd);

			indexl = Valid[indexl] ? indexl : index;
			indexr = Valid[indexr] ? indexr : index;
			indexu = Valid[indexu] ? indexu : index;
			indexd = Valid[indexd] ? indexd : index;

			for (int c = 0; c<colors; c++)
			{
				GradientX.data[index] += (Image[indexl + c*HW] - Image[indexr + c*HW]) / (colors * 255);
				GradientY.data[index] += (Image[indexu + c*HW] - Image[indexd + c*HW]) / (colors * 255);
			}
		}
	}
}

void calculateSource(MexImage<bool> &Valid, MexImage<float> &IntegralConf, MexImage<bool> &Source, const int x0, const int y0, const int x1, const int y1, const int radius)
{
	const int width = Valid.width;
	const int height = Valid.height;
	const int HW = Valid.layer_size;
	//const int colors = Image.layers;
	const int diameter = radius * 2 + 1;
	const int window = diameter*diameter;

#pragma omp parallel for
	for (int x = std::max(radius, x0); x <= std::min(width - radius - 1, x1); x++)
	{
		for (int y = std::max(radius, y0); y <= std::min(height - radius - 1, y1); y++)
		{
			long index = Valid.Index(x, y);

			if (!Valid[index])
				continue;

			int pixels = 0;

			for (int i = 0; i<window; i++)
			{
				int sx = x + i / diameter - radius;
				int sy = y + i%diameter - radius;
				long sindex = Valid.Index(sx, sy);

				if (!Valid[sindex])
					break;

				pixels++;
			}
			//float conf = IntegralConf.getIntegralAverage(x,y,radius);
			if (pixels == window /*&& conf > 0.5*/)
			{
				Source.data[index] = true;
			}
		}
	}
}


void calculateNormal(MexImage<bool> &Valid, MexImage<int8_t> &NormalX, MexImage<int8_t> &NormalY, const int x0, const int y0, const int x1, const int y1)
{
	const int width = Valid.width;
	const int height = Valid.height;
	const int HW = Valid.layer_size;
	//const int colors = Valid.layers;

#pragma omp parallel for
	for (int x = std::max(0, x0); x <= std::min(width - 1, x1); x++)
	{
		for (int y = std::max(0, y0); y <= std::min(height - 1, y1); y++)
		{
			long index = Valid.Index(x, y);

			NormalX.data[index] = 0;
			NormalY.data[index] = 0;

			int xl = std::max(0, x - 1);
			int xr = std::min(width - 1, x + 1);
			int yu = std::max(0, y - 1);
			int yd = std::min(height - 1, y + 1);

			long indexl = Valid.Index(xl, y);
			long indexr = Valid.Index(xr, y);
			long indexu = Valid.Index(x, yu);
			long indexd = Valid.Index(x, yd);

			NormalX.data[index] = -((int8_t)Valid[indexl] - (int8_t)Valid[indexr]);
			NormalY.data[index] = -((int8_t)Valid[indexu] - (int8_t)Valid[indexd]);

		}
	}
}
