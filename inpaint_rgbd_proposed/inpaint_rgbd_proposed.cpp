/** My implementation of the Criminisi inpainting approach
*	Extended with optional Priority map, which can define direction of inpainting
*	@file inpaint_rgbd_proposed.cpp
*	@date 21.06.2016
*	@author Sergey Smirnov <sergey.smirnov@tut.fi>
*/

#include "../common/meximage.h"
#include <float.h>
#include <cmath>
#include <atomic>
#include <algorithm>
#include <vector>
#include <string.h>

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

typedef signed char int8;
typedef unsigned char uint8;
constexpr int colors = 3;

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")

using namespace mymex;
static const float mynan = sqrt(-1.f);

template<int radius>
void calculateSource(MexImage<bool> &Valid, MexImage<bool> &Source);

template<int radius>
void calculateAverage(MexImage<float> &Image, MexImage<bool> &Valid, MexImage<float> &Average, const int x0, const int y0, const int x1, const int y1);
void calculateGradient(MexImage<float> &Image, MexImage<bool> &Valid, MexImage<float> &GradientX, MexImage<float> &GradientY, const int x0, const int y0, const int x1, const int y1);
void calculateNormal(MexImage<bool> &Valid, MexImage<int8> &NormalX, MexImage<int8> &NormalY, const int x0, const int y0, const int x1, const int y1);

template<int radius>
float getAverage(MexImage<float> &Confidence, const int x0, const int y0)
{
	const int width = Confidence.width;
	const int height = Confidence.height;

	int pixels = 0;
	float value = 0;
	for (int x = std::max(0, x0 - radius); x <= std::min(width - 1, x0 + radius); x++)
	{
		for (int y = std::max(0, y0 - radius); y <= std::min(height - 1, y0 + radius); y++)
		{
			value += Confidence(x, y);
			pixels ++;
		}
	}
	return value/pixels;
}

template<int radius>
float getAverage(MexImage<float> &Signal, MexImage<bool> &Valid, const int x0, const int y0)
{
	const int width = Signal.width;
	const int height = Signal.height;

	int pixels = 0;
	float value = 0;
	for (int x = std::max(0, x0 - radius); x <= std::min(width - 1, x0 + radius); x++)
	{
		for (int y = std::max(0, y0 - radius); y <= std::min(height - 1, y0 + radius); y++)
		{
			if (Valid(x, y))
			{
				value += Signal(x, y);
				pixels++;
			}
		}
	}
	return value / pixels;
}

template<int radius>
void processInpainting(MexImage<const float> &Color, MexImage<const float> &Depth, MexImage<float> &Filled, MexImage<float> &FilledDepth, const float mse_thr)
{
	constexpr int diameter = radius * 2 + 1;
	constexpr int window = diameter*diameter;
	constexpr float nan = sqrt(-1.f);
	
	MexImage<float> Confidence(width, height);
	//MexImage<float> AverageConfidence(width, height);

	// Binary map of valid pixels	
	MexImage<bool> Valid(width, height);	
	// Binary map of valid patches	
	MexImage<bool> Source(width, height);
	//MexImage<float> DataTerm(width, height);

	MexImage<float> GradientX(width, height);
	MexImage<float> GradientY(width, height);	
	MexImage<int8> NormalX(width, height);
	MexImage<int8> NormalY(width, height);


#pragma omp parallel for
	for (long i = 0; i<HW; i++)
	{
		bool ishole = false;
		for (int c = 0; c<colors; c++)
		{
			ishole |= bool(isnan(Color[i + c*HW]));
		}

		for (int c = 0; c<colors; c++)
		{
			Filled[i + c*HW] = (ishole) ? mynan : Color[i + c*HW];
		}

		Valid[i] = !ishole;
		Confidence[i] = float(!ishole);
	}


#pragma omp parallel sections
	{
#pragma omp section
		{
			// normals map (used to find boundary patches)	
			calculateNormal(Valid, NormalX, NormalY, 0, 0, width - 1, height - 1);
		}
#pragma omp section
		{
			// calculated just once, no updates will be possible
			calculateSource<radius>(Valid, Source);
		}
#pragma omp section
		{
			// inital estimation, updated after each iteration
			calculateGradient(Signal, Valid, GradientX, GradientY, 0, 0, width - 1, height - 1);
		}
	}
		
	unsigned iter = 0;
	long updated = 1;

	unsigned long long searches = 0;

	// process goes iteratively
	while (updated > 0 && (!maxiter || iter < maxiter))
	{
		updated = 0;

		long best_index = -1;
		float high_priority = 0.f;
		float best_conf = -0.1f;
		float average_depth = nan;

		// 1. find appropriate pixel, where to inpaint
#pragma omp parallel for schedule (dynamic)
		for (long i = 0; i<HW; i++)
		{
			const int x = i / height;
			const int y = i % height;

			if(!NormalX[i] && !NormalY[i])
			{
				continue;
			}
			const float conf = getAverage(Confidence, x, y);
			const float depth = getAverage(FilledDepth, Valid, x, y);
			//const float conf = IntegralConf.getIntegralAverage(x, y, radius, 0);
			// Try less decaying confidence 
			//const float conf = exp(IntegralConf.getIntegralAverage(x, y, radius, 0) - 1.f);

			

			
			float gx = GradientY[i];
			float gy = -GradientX[i];
			gx = isnan(gx) ? 0 : gx;
			gy = isnan(gy) ? 0 : gy;

			//float data = abs(gx*NormalX[i]) + abs(gy*NormalY[i]);
			float data = abs(gx*NormalX[i] + gy*NormalY[i]);
			data = data > 0.f ? data : 0.001;
			//data = data > 0.001 ? data : 0.001;
			//DataTerm.data[i] = data;
			const float priority = sqrt(sqrt(conf * data) * (1.f - depth/255.f));
			
			if (priority > high_priority)
			{
				#pragma critical
				{
					best_index = i;
					high_priority = priority;
					best_conf = conf;
					average_depth = depth;
				}				
			}
		}
		if (best_index < 0)
		{
			mexPrintf("ERROR: No appropriate patch to inpaint was found :-(\n");
			break;
		}

		const int best_x = best_index / height;
		const int best_y = best_index % height;

		// seek for the best valid patch over whole image o_O
		float best_diff = 1000000.f;
		long best_sindex = -1;
		//float best_avg = 1000000.f;				
		unsigned long checked = 0;
		unsigned long runned = 0;

		
		#pragma omp parallel for reduction(+:checked,runned) schedule(dynamic)
		for(long spiral=1; spiral<HW; spiral++)		
		{
			int r = 1;
			int accum = 8; 
			while (accum < spiral)
			{
				r++;
				int l = r * 8;
				accum += l;
			}
			const int dx = (spiral - accum)
			
			if (!Source[i])
				continue;
			
			const int x = i / height;
			const int y = i % height;
						
			runned++;

			float diff = 0;
			int pixels = 0;
			for (int dx = -radius; dx <= radius; dx++)
			{
				const int sx = x + dx;
				const int bx = best_x + dx;
				if (bx < 0 || bx >= width)
					continue;

				for (int dy = -radius; dy <= radius; dy++)
				{
					const int sy = y + dy;
					const int by = best_y + dy;

					if (by < 0 || by >= height)
						continue;

					//long sindex = Filled.Index(sx, sy);
					//long bindex = Filled.Index(bx, by);

					if (!Valid(bx, by))
					{
						continue;
					}

					for (int c = 0; c<colors; c++)
					{
						float diffC = (Signal[sindex + c*HW] - Filled[bindex + c*HW]);
						diff += diffC * diffC;
					}

					pixels++;
				}
			}

			diff /= pixels;
			diff = sqrt(diff);

			if (diff < best_diff)
			{
				#pragma omp critical
				{
					best_diff = diff;
					best_sindex = i;
				}				
			}

			checked++;

			if (best_diff <= mse_thr)
			{
				break;
			}
		}

		if (best_sindex < 0)
		{
			mexPrintf("ERROR: No appropriate match for patch at (%d, %d) was found (%d (%d) patches were checked, ) \n", best_x, best_y, checked, runned);
			break;
		}

		searches += checked;

		const int best_sx = best_sindex / height;
		const int best_sy = best_sindex % height;
		//mexPrintf("\t best patch: (%d, %d), diff = %3.2f\n", best_sx, best_sy, best_diff);

		// update found hole with found patch
#pragma omp parallel for
		for (int dx = -update; dx <= update; dx++)
		{
			int sx = best_sx + dx;
			int bx = best_x + dx;
			if (bx < 0 || bx >= width)
				continue;

			for (int dy = -update; dy <= update; dy++)
			{
				int sy = best_sy + dy;
				int by = best_y + dy;

				if (by < 0 || by >= height)
					continue;

				long sindex = Filled.Index(sx, sy);
				long bindex = Filled.Index(bx, by);

				if (Valid[bindex])
				{
					continue;
				}

				Valid[bindex] = true;
				Confidence[bindex] = best_conf;

				for (int c = 0; c<colors; c++)
				{
					Filled[bindex + c*HW] = Signal[sindex + c*HW];
				}

				updated++;
			}
		}


		// update normals map
		calculateNormal(Valid, NormalX, NormalY, std::max(0, best_x - update - 1), std::max(0, best_y - update - 1), std::min(width - 1, best_x + update + 1), std::min(height - 1, best_y + update + 1));

		// update averages map
		if (earlyTerm)
		{
			calculateAverage(Filled, Valid, Average, std::max(0, best_x - radius - update), std::max(0, best_y - radius - update), std::min(width - 1, best_x + radius + update), std::min(height - 1, best_y + radius + update), radius);
		}

		// update gradients
		if (useDataTerm)
		{
			if (averagedGradient)
			{
				calculateGradient(Average, Valid, GradientX, GradientY, std::max(0, best_x - update - 1), std::max(0, best_y - update - 1), std::min(width - 1, best_x + update + 1), std::min(height - 1, best_y + radius + update + 1));
			}
			else
			{
				calculateGradient(Filled, Valid, GradientX, GradientY, std::max(0, best_x - update - 1), std::max(0, best_y - update - 1), std::min(width - 1, best_x + update + 1), std::min(height - 1, best_y + update + 1));
			}
		}

		//calculateAverage(GradientX, Valid, GradientXAvg, std::max(0,best_x-radius-update), std::max(0,best_y-radius-update), std::min(width-1,best_x+radius+update), std::min(height-1,best_y+radius+update), radius);
		//calculateAverage(GradientY, Valid, GradientYAvg, std::max(0,best_x-radius-update), std::max(0,best_y-radius-update), std::min(width-1,best_x+radius+update), std::min(height-1,best_y+radius+update), radius);

		//holes -= updated;		
		iter++;

		for (long i = 0; i<HW; i++)
		{
			IntegralConf.data[i] = Confidence[i];
		}

		IntegralConf.IntegralImage(true);

		//if(upd_source)
		//{
		//	calculateSource(Valid, IntegralConf, Source, best_x-update-radius, best_y-update-radius, best_x+update+radius, best_y+update+radius, radius+1);
		//}

		//if(! iter % 100)
		//mexPrintf("Iteration %d; updated %d; remaining holes %d\n\n", iter, updated, holes);
		//mexPrintf("Iteration %d; updated %d; checked %d; remaining holes %d\n", iter, updated, checked, holes);
		//avg_checked = (avg_checked*(iter-1) + checked)/iter;
	}
}

void mexFunction(int nout, mxArray* output[], int in, const mxArray* input[])
{
#ifndef _DEBUG
	omp_set_num_threads(std::max<int>(4, omp_get_max_threads()));
	omp_set_dynamic(std::max<int>(2, omp_get_max_threads() / 2));;
#endif	

	if (in < 3 || in > 4 || nout != 2 || mxGetClassID(input[0]) != mxSINGLE_CLASS || mxGetClassID(input[1]) != mxSINGLE_CLASS)
	{
		mexErrMsgTxt("USAGE: [FilledColor, FilledDepth] = inpaint_rgbd_proposed(Color, Depth, patch_radius, <mse_threshold>);\n");
	}

	MexImage<const float> Color(input[0]);
	MexImage<const float> Depth(input[1]);

	const int height = Color.height;
	const int width = Color.width;
	const int HW = Color.layer_size;
	const int radius = std::max(1, (int)mxGetScalar(input[2]));
	const float mse_thr = in > 3 ? std::abs((float)mxGetScalar(input[3])) : 0.f;

	if (Color.layers != colors || Depth.width != width || Depth.height != height)
	{
		mexErrMsgTxt("ERROR: Something wrong with resolutions/colors.");
	}

	matlab_size dims3d[] = { (matlab_size)height, (matlab_size)width, colors };
	matlab_size dims2d[] = { (matlab_size)height, (matlab_size)width, 1 };
	output[0] = mxCreateNumericArray(3, dims3d, mxSINGLE_CLASS, mxREAL);
	output[1] = mxCreateNumericArray(3, dims2d, mxSINGLE_CLASS, mxREAL);
	MexImage<float> Filled(output[0]);
	MexImage<float> FilledDepth(output[1]);
	
	switch (radius)
	{
	case 1: processInpainting<1>(Color, Depth, Filled, FilledDepth, mse_thr); break;
	case 2: processInpainting<2>(Color, Depth, Filled, FilledDepth, mse_thr); break;
	case 3: processInpainting<3>(Color, Depth, Filled, FilledDepth, mse_thr); break;
	case 4: processInpainting<4>(Color, Depth, Filled, FilledDepth, mse_thr); break;
	case 5: processInpainting<5>(Color, Depth, Filled, FilledDepth, mse_thr); break;
	case 6: processInpainting<6>(Color, Depth, Filled, FilledDepth, mse_thr); break;
	case 7: processInpainting<7>(Color, Depth, Filled, FilledDepth, mse_thr); break;
	case 8: processInpainting<8>(Color, Depth, Filled, FilledDepth, mse_thr); break;
	case 9: processInpainting<9>(Color, Depth, Filled, FilledDepth, mse_thr); break;
	case 10: processInpainting<10>(Color, Depth, Filled, FilledDepth, mse_thr); break;
	case 11: processInpainting<11>(Color, Depth, Filled, FilledDepth, mse_thr); break;
	case 12: processInpainting<12>(Color, Depth, Filled, FilledDepth, mse_thr); break;
	default: mexErrMsgTxt("Wrong patch_radius!");
	}

}

template<int radius>
void calculateAverage(MexImage<float> &Image, MexImage<bool> &Valid, MexImage<float> &Average, const int x0, const int y0, const int x1, const int y1)
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
	const int colors = Image.layers;

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

template<int radius>
void calculateSource(MexImage<bool> &Valid, MexImage<bool> &Source)
{
	Source.setval(false);
	const int width = Valid.width;
	const int height = Valid.height;
	const int HW = Valid.layer_size;
	constexpr int diameter = radius * 2 + 1;
	constexpr int window = diameter*diameter;

#pragma omp parallel for
	for (int x = radius; x <= width - radius - 1; x++)
	{
		for (int y = radius; y <= height - radius - 1; y++)
		{
			if (!Valid(x,y))
				continue;

			int pixels = 0;

			for (int i = 0; i<window; i++)
			{
				const int sx = x + i / diameter - radius;
				const int sy = y + i % diameter - radius;

				if (!Valid(sx, sy))
					break;

				pixels++;
			}

			if (pixels == window)
			{
				Source.data[index] = true;
			}
		}
	}
}


void calculateNormal(MexImage<bool> &Valid, MexImage<int8> &NormalX, MexImage<int8> &NormalY, const int x0, const int y0, const int x1, const int y1)
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

			NormalX.data[index] = -((int8)Valid[indexl] - (int8)Valid[indexr]);
			NormalY.data[index] = -((int8)Valid[indexu] - (int8)Valid[indexd]);
		}
	}
}
