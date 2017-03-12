/** Inpainting for RGB+D images
*	@file inpaint_rgbd_jtdi.cpp
*	@date 18.05.2016
*	@author Sergey Smirnov <sergey.smirnov@tut.fi>
*/

#include "../common/meximage.h"
#include <float.h>
#include <cmath>
#include <algorithm>
#include <vector>
#ifndef _DEBUG
#include <omp.h>
#endif

#include <string.h>

#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}
#endif // __cplusplus

typedef signed char int8;

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")

using namespace mymex;

//! Mark valid source (binary image)
void calculateSource(MexImage<bool> &Valid, MexImage<float> &IntegralConf, MexImage<bool> &Source, const int x0, const int y0, const int x1, const int y1, const int radius);

void calculateGradient(MexImage<float> &Image, MexImage<bool> &Valid, MexImage<float> &GradientX, MexImage<float> &GradientY, const int x0, const int y0, const int x1, const int y1);

void calculateNormal(MexImage<bool> &Valid, MexImage<int8> &NormalX, MexImage<int8> &NormalY, const int x0, const int y0, const int x1, const int y1);

std::pair<float, float> depthMeanAndVariance(const MexImage<float> &FilledDepth, const int x, const int y, const int radius);


void mexFunction(int nout, mxArray* output[], int in, const mxArray* input[])
{
#ifndef _DEBUG
	omp_set_num_threads(std::max<int>(4, omp_get_max_threads()));
	omp_set_dynamic(std::max<int>(2, omp_get_max_threads() / 2));;
#endif	

	if (in < 2 || in > 4 || nout != 4 || mxGetClassID(input[0]) != mxSINGLE_CLASS || mxGetClassID(input[1]) != mxSINGLE_CLASS)
	{
		mexPrintf("USAGE: [FilledColor, FilledDepth, Confidence, DataTerm] = inpaint_rgbd_jtdi(Color, Depth, <search, radius>);\n");
		mexPrintf("Color - multi-color single-valued image.\n");
		mexPrintf("Depth - depth or disparity map, Nan-s are treated as holes which needs to be inpainted.\n");		
		mexPrintf("search - radius of the search area. <default: 0 - whole image>\n");
		mexPrintf("radius - radius of the patch size. <default: 3>\n");
		mexErrMsgTxt("Wrong input/output parameters!");
	}

	MexImage<float> Color(input[0]);
	MexImage<float> Depth(input[1]);

	const int height = Color.height;
	const int width = Color.width;
	const int HW = Color.layer_size;
	const int colors = Color.layers;

	if (Depth.width != width || Depth.height != height || Depth.layers != 1)
	{
		mexErrMsgTxt("Wrong Depth dimensions!");
	}

	const int _search = (in > 2) ? std::abs((int)mxGetScalar(input[2])) : 0;
	const int search = _search ? _search : std::max(width, height);
	const int radius = (in > 3) ? std::max(1, (int)mxGetScalar(input[3])) : 3;

	const int diameter = radius * 2 + 1;
	const int window = diameter*diameter;
	const float nan = sqrt(-1.f);

	matlab_size dims3d[] = { (matlab_size)height, (matlab_size)width, colors };
	matlab_size dims2d[] = { (matlab_size)height, (matlab_size)width, 1 };

	output[0] = mxCreateNumericArray(3, dims3d, mxSINGLE_CLASS, mxREAL);
	output[1] = mxCreateNumericArray(3, dims2d, mxSINGLE_CLASS, mxREAL);

	output[2] = mxCreateNumericArray(3, dims2d, mxSINGLE_CLASS, mxREAL);
	output[3] = mxCreateNumericArray(3, dims2d, mxSINGLE_CLASS, mxREAL);


	MexImage<float> Filled(output[0]);
	MexImage<float> FilledDepth(output[1]);
	MexImage<float> Confidence(output[2]);
	MexImage<float> DataTerm(output[3]);
	
	MexImage<float> IntegralConf(width, height);
	MexImage<float> IntegralDepth(width, height);
	MexImage<int8> NormalX(width, height);
	MexImage<int8> NormalY(width, height);
	MexImage<float> GradientX(width, height);
	MexImage<float> GradientY(width, height);
	MexImage<bool> Valid(width, height);
	MexImage<bool> Source(width, height);
	
	IntegralConf.setval(0.f);
	DataTerm.setval(0.f);
	NormalX.setval(false);
	NormalY.setval(false);
	Valid.setval(false);
	Source.setval(false);


	unsigned long holes = 0;

	#pragma omp parallel for reduction(+:holes)
	for (long i = 0; i<HW; i++)
	{
		bool ishole = isnan(Depth[i]);
		for (int c = 0; c<colors; c++)
		{
			ishole |= isnan(Color[i + c*HW]);
		}

		holes += ishole;
		Valid[i] = !ishole;
	}

	for (long i = 0; i<HW; i++)
	{
		bool ishole = !Valid[i];

		FilledDepth[i] = (ishole) ? nan : Depth[i];
		for (int c = 0; c<colors; c++)
		{
			Filled[i + c*HW] = (ishole) ? nan : Color[i + c*HW];
		}

		holes += ishole;
		Valid[i] = !ishole;

		Confidence[i] = float(!ishole);
		IntegralConf[i] = Confidence[i];
		IntegralDepth[i] = ishole ? 0.f : Depth[i];
	}

#pragma omp parallel sections
	{
#pragma omp section
		{
			// calculate depth averages
			IntegralDepth.IntegralImage(true);
		}

#pragma omp section
		{
			// calculate fast confidence
			IntegralConf.IntegralImage(true);
		}
#pragma omp section
		{
			// normals map (used to find boundary patches)
			calculateNormal(Valid, NormalX, NormalY, 0, 0, width - 1, height - 1);
		}

	}

	// prepare mask of valid source
	calculateSource(Valid, IntegralConf, Source, 0, 0, width, height, radius);

	// Prepare Gradients for Data Term 
	calculateGradient(Color, Valid, GradientX, GradientY, 0, 0, width - 1, height - 1);
	
	
	unsigned iter = 0;
	long updated = 1;

	//double avg_checked = 0;
	unsigned long long searches = 0;
	const unsigned long holes_saved = holes;

	// process goes iteratively
	while (holes > 0 && updated > 0)
	{
		updated = 0;

		long best_index = -1;
		float high_priority = 0.f;
		float best_conf = -0.1f;
		std::pair<float, float> best_mean_variance;

		// order of inpainting - find appropriate hole :-)
		#pragma omp parallel for
		for (long i = 0; i<HW; i++)
		{
			int x = i / height;
			int y = i % height;
			
			// central pixel of a patch for inpainting can be inside the hole, but normals must still be available
			if(!NormalX[i] && !NormalY[i])
			{
				continue;
			}			
						
			// Onion-peel Confidence
			const float conf = IntegralConf.getIntegralAverage(x, y, radius);

			// Estimate DataTerm
			float gx = GradientY[i];
			float gy = -GradientX[i];
			gx = isnan(gx) ? 0 : gx;
			gy = isnan(gy) ? 0 : gy;
			//float gdx = abs(DepthGradientY[i]);
			//float gdy =- abs(DepthGradientX[i]);
			//gdx = isnan(gdx) ? 0 : gdx;
			//gdy = isnan(gdy) ? 0 : gdy;

			// calculate isophote here
			float data = abs(gx*NormalX[i] + gy*NormalY[i]);
			//data *= abs(gdx*NormalX[i] - gdy*NormalY[i]);				

			data = data < 0.001 ? 0.001 : data;
			DataTerm.data[i] = data;

			// Calculate Depth Variance Term
			std::pair<float, float> mean_variance = depthMeanAndVariance(FilledDepth, x, y, radius);

			// JTDI priority term
			const float priority = (data + conf + mean_variance.second) * (255 - mean_variance.first);			

			if (priority > high_priority)
			{
				#pragma omp critical
				{
					high_priority = priority;
					best_index = i;
					best_mean_variance = mean_variance;
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

		//// calculate average depth of the patch-to-inpaint
		//float average_depth = 0;
		//int pxls = 0;
		//for(int dx=-radius, i=0; dx<=radius; dx++)
		//{
		//	for(int dy=-radius; dy<=radius; dy++, i++)
		//	{
		//		const int xx = best_x + dx;
		//		const int yy = best_y + dy;
		//		if(xx<0 || xx>= width || yy<0 || yy>= height)
		//		{
		//			continue;
		//		}
		//		const long indexx = FilledDepth.Index(xx,yy);
		//		if(!Valid[indexx])
		//		{
		//			continue;
		//		}
		//		average_depth += FilledDepth[indexx];
		//		pxls ++;
		//	}
		//}
		//average_depth /= pxls;

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

						long sindex = Filled.Index(sx, sy);
						long bindex = Filled.Index(bx, by);

						if (!Valid[bindex])
						{
							continue;
						}


						for (int c = 0; c<colors; c++)
						{
							diff += abs(Color[sindex + c*HW] - Filled[bindex + c*HW]);
						}

						//diff += abs((Depth[sindex]-average_search_depth) - (FilledDepth[bindex]-average_depth));					

						//diff /= (colors);
						//diff += abs(Depth[sindex] - FilledDepth[bindex]);
						//diff += abs(DepthGradientX[sindex] - DepthGradientX[bindex]);
						//diff += abs(DepthGradientY[sindex] - DepthGradientY[bindex]);

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

		//const float depth_source_average = IntegralDepth.getIntegralAverage(best_sx, best_sy, radius);
		auto sourse_mv = depthMeanAndVariance(FilledDepth, best_sx, best_sy, radius);
		const float depth_source_average = sourse_mv.first;

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

				long sindex = Filled.Index(sx, sy);
				long bindex = Filled.Index(bx, by);

				if (Valid[bindex])
				{
					continue;
				}

				Valid.data[bindex] = true;
				Confidence.data[bindex] = best_conf;

				for (int c = 0; c<colors; c++)
				{
					Filled.data[bindex + c*HW] = Color[sindex + c*HW];
				}
				FilledDepth.data[bindex] = Depth[sindex] - depth_source_average + best_mean_variance.first;
				FilledDepth.data[bindex] = FilledDepth.data[bindex] < 0 ? 0 : FilledDepth.data[bindex];

				updated++;
			}
		}

		// update normals map
		calculateNormal(Valid, NormalX, NormalY, std::max(0, best_x - radius - 1), std::max(0, best_y - radius - 1), std::min(width - 1, best_x + radius + 1), std::min(height - 1, best_y + radius + 1));
		
		// update gradients required for Data Term
		calculateGradient(Filled, Valid, GradientX, GradientY, std::max(0, best_x - radius - 1), std::max(0, best_y - radius - 1), std::min(width - 1, best_x + radius + 1), std::min(height - 1, best_y + radius + 1));

		holes -= updated;
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

	mexPrintf("Inpainted pixels: %d; using patches: %d; total matches: %d. \n", holes_saved - holes, iter, searches);
}

inline int block_index(const int ix, const int iy, const int diameter)
{
	return ix*diameter + iy;
}

//! L(p) Depth Variance term, from Daribo opaper
std::pair<float, float> depthMeanAndVariance(const MexImage<float> &FilledDepth, const int x, const int y, const int radius)
{
	const int width = FilledDepth.width;
	const int height = FilledDepth.height;
	const int HW = FilledDepth.layer_size;
	float average = 0, variance = 0;
	int valid_pixels = 0;
	for (int xx = std::max(0, x - radius); xx <= std::min(width - 1, x + radius); xx++)
	{
		for (int yy = std::max(0, y - radius); yy <= std::min(height - 1, y + radius); yy++)
		{
			if (!isnan(FilledDepth(xx, yy)))
			{
				average += FilledDepth(xx, yy);
				valid_pixels++;
			}
		}
	}

	if (valid_pixels == 0)
	{
		return std::make_pair(0.f, 0.f);
	}

	average /= valid_pixels;

	for (int xx = std::max(0, x - radius); xx <= std::min(width - 1, x + radius); xx++)
	{
		for (int yy = std::max(0, y - radius); yy <= std::min(height - 1, y + radius); yy++)
		{
			if (!isnan(FilledDepth(xx, yy)))
			{
				variance += (FilledDepth(xx, yy) - average)*(FilledDepth(xx, yy) - average);
			}
		}
	}
	const float variance_term = valid_pixels / (valid_pixels + variance);

	return std::make_pair(average, variance_term);
}

void calculateAverage(MexImage<float> &Image, MexImage<bool> &Valid, MexImage<float> &Average, const int x0, const int y0, const int x1, const int y1, const int radius)
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
			if (indexl == indexr || indexu == indexd)
			{
				Valid.data[index] = false;
			}
			else
			{
				for (int c = 0; c<colors; c++)
				{
					GradientX.data[index] += abs(Image[indexr + c*HW] - Image[indexl + c*HW]) / (colors * 255);
					GradientY.data[index] += abs(Image[indexd + c*HW] - Image[indexu + c*HW]) / (colors * 255);
				}
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


void calculateNormal(MexImage<bool> &Valid, MexImage<int8> &NormalX, MexImage<int8> &NormalY, const int x0, const int y0, const int x1, const int y1)
{
	const int width = Valid.width;
	const int height = Valid.height;
	const int HW = Valid.layer_size;
	const int colors = Valid.layers;

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

			NormalX.data[index] = -((int8)Valid[indexr] - (int8)Valid[indexl]);
			NormalY.data[index] = -((int8)Valid[indexd] - (int8)Valid[indexu]);
		}
	}
}

