/** My implementation of the Criminisi inpainting approach
*	Extended with optional Priority map, which can define direction of inpainting
*	@file criminisi_inpainting.cpp
*	@date 11.03.2013
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

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")

using namespace mymex;
static const float mynan = sqrt(-1.f);

void calculateSource(MexImage<bool> &Valid, MexImage<float> &IntegralConf, MexImage<bool> &Source, const int x0, const int y0, const int x1, const int y1, const int radius);
void calculateAverage(MexImage<float> &Image, MexImage<bool> &Valid, MexImage<float> &Average, const int x0, const int y0, const int x1, const int y1, const int radius);
void calculateGradient(MexImage<float> &Image, MexImage<bool> &Valid, MexImage<float> &GradientX, MexImage<float> &GradientY, const int x0, const int y0, const int x1, const int y1);
void calculateNormal(MexImage<bool> &Valid, MexImage<int8> &NormalX, MexImage<int8> &NormalY, const int x0, const int y0, const int x1, const int y1);

void mexFunction (int nout, mxArray* output[], int in, const mxArray* input[])
{	
	#ifndef _DEBUG
	omp_set_num_threads(std::max<int>(4, omp_get_max_threads()));
	omp_set_dynamic(std::max<int>(2, omp_get_max_threads()/2));;
	#endif	

	if(in < 1 || in > 8 || nout != 2 || mxGetClassID(input[0]) != mxSINGLE_CLASS)
	{
		mexPrintf("USAGE: [Filled, Confidence] = criminisi_inpainting(Signal, {Priority}, <search, radius, option1, option2, ...>);\n");
		mexPrintf("Filled - all NaN values in 'Single' replaced by predicted values.\n");
		mexPrintf("Signal - multi-color single-valued image with NaN values.\n");		
		mexPrintf("Priority - single-valued image with 0..1 values.\n");		
		mexPrintf("search - radius of the search area. <default: 0 - whole image>\n");
		mexPrintf("radius - radius of the patch size. <default: 3>\n");
		mexPrintf("Possible options: \n");
		mexPrintf("'UpdateSource' - uses inpainted area as a source.\n");
		mexPrintf("'EarlyTermination' - Tries to speed-up exhausive patch search.\n");
		mexPrintf("'AveragedGradient' - requires 'EarlyTermination', not compatible with 'NoDataTerm'.\n");
		mexPrintf("'NoDataTerm' - Onion-peel updating approach.\n");		
		mexPrintf("'SmallerUpdate' - Updated area is smaller than a patch size.\n");
		mexPrintf("'I100' - Performs 100 iterations at max (number can vary).\n");		
		//update, upd_source,
		mexErrMsgTxt("Wrong input/output parameters!");
    }

	MexImage<float> Signal(input[0]);	

	const int height = Signal.height;
	const int width = Signal.width;
	const int HW = Signal.layer_size;
	const int colors = Signal.layers;

	matlab_size dims3d[] = { (matlab_size)height, (matlab_size)width, colors };
	matlab_size dims2d[] = { (matlab_size)height, (matlab_size)width, 1 };
	output[0] = mxCreateNumericArray(3, dims3d, mxSINGLE_CLASS, mxREAL);
	output[1] = mxCreateNumericArray(3, dims2d, mxSINGLE_CLASS, mxREAL);
	MexImage<float> Filled(output[0]);
	MexImage<float> Confidence(output[1]);
	MexImage<bool> Valid(width, height);


	#pragma omp parallel for
	for (long i = 0; i<HW; i++)
	{
		bool ishole = false;
		for (int c = 0; c<colors; c++)
		{
			ishole |= bool(isnan(Signal[i + c*HW]));
		}

		for (int c = 0; c<colors; c++)
		{
			Filled[i + c*HW] = (ishole) ? mynan : Signal[i + c*HW];
		}

		Valid[i] = !ishole;
		Confidence[i] = float(!ishole);
	}
	
	int _offset = 0;
	if (in > 1 && mxGetClassID(input[1]) == mxSINGLE_CLASS && mxGetWidth(input[1]) == width && mxGetHeight(input[1]) == height)
	{
		_offset = 1;
		MexImage<float> Priority(input[1]);
		#pragma omp parallel for
		for (long i = 0; i<HW; i++)
		{
			Confidence[i] *= Priority[i] < 0.f ? 0.f : Priority[i] > 1.f ? 1.f : Priority[i];
		}
	}

	const int _search = (in > 1+_offset) ? std::abs((int)mxGetScalar(input[1+_offset])) : 0;	
	const int search = _search ? _search : std::max(width, height);
	const int radius = (in > 2+_offset) ? std::max(1, (int)mxGetScalar(input[2+_offset])) : 3;			
	
	// Preliminary input parameter flags
	bool _upd_source = false;
	unsigned _maxiter = 0;
	bool _smaller_update = false;
	bool _earlyTerm = false;
	bool _noDataTerm = false;
	bool _averagedGradient = false;

	// Parsing of input parameters
	if(in > 3+_offset)
	{
		char buffer[40];
		for(int i=3+_offset; i<in; i++)
		{			
			const mxArray* cell = input[i];
			if(mxGetClassID(cell) == mxCHAR_CLASS && !mxGetString(cell, buffer, 39))
			{
				if(!strcmp(buffer, "UpdateSource") || !strcmp(buffer, "us"))
				{
					_upd_source = true;
				}
				else if(!strcmp(buffer, "EarlyTermination") || !strcmp(buffer, "et"))
				{
					_earlyTerm = true;
				}
				else if(!strcmp(buffer, "SmallerUpdate") || !strcmp(buffer, "su"))
				{
					_smaller_update = true;
				}
				else if(!strcmp(buffer, "NoDataTerm") || !strcmp(buffer, "ndt"))
				{
					_noDataTerm = true;
				}
				else if(!strcmp(buffer, "AveragedGradient") || !strcmp(buffer, "ag"))
				{
					_averagedGradient = true;
				}
				else if(!strncmp(buffer, "I", 1))
				{
					_maxiter = atoi(buffer+1);
				}
				else
				{
					mexErrMsgTxt("Cannot understand optional parameter.");
				}
			}
			else
			{
				mexErrMsgTxt("Optional parameter is not a string.");
			}
		}
	}

	// 
	const bool upd_source = _upd_source;		
	const unsigned maxiter = _maxiter;
	const int update = (_smaller_update && radius>0) ? radius-1 : radius;
	const bool earlyTerm = _earlyTerm;
	const bool useDataTerm = !_noDataTerm;
	const bool averagedGradient  = useDataTerm && _averagedGradient && earlyTerm;

	const int diameter = radius*2 + 1;
	const int window = diameter*diameter;
	const float nan = sqrt(-1.f);

	MexImage<float> DataTerm (width, height);   
	MexImage<float> Average (width, height, colors);
	MexImage<float> GradientX (width, height);
	MexImage<float> GradientY (width, height);
	MexImage<float> IntegralConf(width, height);	
	MexImage<int8> NormalX (width, height);
	MexImage<int8> NormalY (width, height);	
	MexImage<bool> Source(width, height);	

	#pragma omp parallel sections
	{
		#pragma omp section
		{
			// normals map (used to find boundary patches)	
			NormalX.setval(0);
			NormalY.setval(0);			
			calculateNormal(Valid, NormalX, NormalY, 0, 0, width - 1, height - 1);
		}
		#pragma omp section
		{
			Average.setval(0.f);
			DataTerm.setval(0.f);
			Source.setval(false);
			GradientX.setval(0.f);
			GradientY.setval(0.f);
		}

		#pragma omp section
		{
			// calculate fast confidence
			IntegralConf.set(Confidence);
			IntegralConf.IntegralImage(true);
		}
	}	


	// fast binary map of valid patches	
	calculateSource(Valid, IntegralConf, Source, 0, 0, width, height, radius);

	// average map (used in early termination)
	if(earlyTerm)
	{
		calculateAverage(Signal, Valid, Average, 0, 0, width-1, height-1, radius);
	}
	
	// gradients map (isophotes for data term)
	if(useDataTerm)
	{
		if(averagedGradient)
		{
			calculateGradient(Average, Valid, GradientX, GradientY, 0, 0, width-1, height-1);		
		}
		else
		{
			calculateGradient(Signal, Valid, GradientX, GradientY, 0, 0, width-1, height-1);		
		}
	}

		
	//bool hasHoles = true;
	unsigned iter = 0;	
	long updated = 1;
		
	unsigned long long searches = 0;	

	// process goes iteratively
	while(updated > 0 && (!maxiter || iter < maxiter))
	{		
		updated = 0;
		
		long best_index = -1;
		float high_priority = 0.f;
		float best_conf = -0.1f;
		
		// 1. find appropriate pixel, where to inpaint
		#pragma omp parallel for
		for(long i=0; i<HW; i++)
		{
			const int x = i / height;
			const int y = i % height;

			/*if(!NormalX[i] && !NormalY[i])
			{
				continue;
			}*/

			if ((useDataTerm && !NormalX[i] && !NormalY[i]) || (!useDataTerm && Valid[i]))
			{
				continue;
			}			

			const float conf = IntegralConf.getIntegralAverage(x, y, radius, 0);			
			// Try less decaying confidence 
			//const float conf = exp(IntegralConf.getIntegralAverage(x, y, radius, 0) - 1.f);
			
			float priority = conf;

			if(useDataTerm)
			{
				float gx = GradientY[i];
				float gy = -GradientX[i];
				gx = isnan(gx) ? 0 : gx;
				gy = isnan(gy) ? 0 : gy;

				//float data = abs(gx*NormalX[i]) + abs(gy*NormalY[i]);
				float data = abs(gx*NormalX[i] + gy*NormalY[i]);
				data = data > 0.f ? data : 0.001;
				//data = data > 0.001 ? data : 0.001;
				DataTerm.data[i] = data;
				priority *= data;
			} 
			/*
#if ATOMIC_LLONG_LOCK_FREE != 2
#error "CANNOT COMPILE LOCK-FREE VERSION"
#endif
			std::atomic<unsigned long long> best_value;

			while (true)
			{
				unsigned long long value = best_value.load();

				float 
			}
			*/
			if(priority > high_priority)
			{
				//std::compare_exchange_strong();
				
				#pragma omp critical
				{
					high_priority = priority;
					best_index = i;
				}
				best_conf = conf;
			}
		}
		if(best_index < 0)
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
		for(int x=std::max(radius, best_x-search); x<=std::min(width-1-radius, best_x+search); x++)
		for(int y=std::max(radius, best_y-search); y<=std::min(height-1-radius, best_y+search); y++)
		{
			runned ++;
			long i = Source.Index(x,y);
			if(!Source[i])
				continue;
			
			// Early termination with averaging
			float avgdiff = 0;

			for(int c=0; c<colors && earlyTerm; c++)
			{
				avgdiff += abs(Average[best_index+c*HW] - Average[i + c*HW]);			
			}

			if(earlyTerm && avgdiff > best_diff)// && best_sindex > 0)
				continue;			

			//if(earlyTerm && avgdiff > best_avg)// && best_sindex > 0)
			//	continue;			
						
			//for(int c=0; c<colors; c++)
			//{
			//	avgdiff += abs(GradientXAvg[best_index+c*HW] - GradientXAvg[i + c*HW])/colors;			
			//}
			//
			//if(avgdiff > best_diff)// && best_sindex > 0)
			//	continue;			

			//for(int c=0; c<colors; c++)
			//{
			//	avgdiff += abs(GradientYAvg[best_index+c*HW] - GradientYAvg[i + c*HW])/colors;			
			//}

			//if(avgdiff > best_diff)// && best_sindex > 0)
			//	continue;			

									
			float diff = 0;
			int pixels = 0;
			for(int dx=-radius; dx<=radius; dx++)
			{
				int sx = x + dx;
				int bx = best_x + dx;
				if(bx < 0 || bx >= width)
					continue;

				for(int dy=-radius; dy<=radius; dy++)
				{
					int sy = y + dy;
					int by = best_y + dy;
					
					if(by < 0 || by >= height)
						continue;
					
					long sindex = Filled.Index(sx, sy);
					long bindex = Filled.Index(bx, by);

					if(!Valid[bindex])
					{
						continue;
					}
					
					for(int c=0; c<colors; c++)
					{
						diff += abs(Signal[sindex+c*HW] - Filled[bindex+c*HW]);
					}

					// Additional constraint which penalize usage of not-confident parts of the image while seeking for the best coressponding patch
					// However, effect of such penalizing is pretty minimal
					//if(Confidence->data[sindex])
					//{
					//	diff += 100*colors;
					//}

					pixels ++;
				}				
			}
			
			diff /= pixels;

			if(diff < best_diff)
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

			checked ++;			
		}
		
		if(best_sindex < 0)
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
		for(int dx=-update; dx<=update; dx++)
		{
			int sx = best_sx + dx;
			int bx = best_x + dx;
			if(bx < 0 || bx >= width)
				continue;

			for(int dy=-update; dy<=update; dy++)
			{
				int sy = best_sy + dy;
				int by = best_y + dy;
					
				if(by < 0 || by >= height)
					continue;
					
				long sindex = Filled.Index(sx, sy);
				long bindex = Filled.Index(bx, by);

				if(Valid[bindex])
				{
					continue;
				}

				Valid[bindex] = true;
				Confidence[bindex] = best_conf;

				for(int c=0; c<colors; c++)
				{
					Filled[bindex+c*HW] = Signal[sindex+c*HW];
				}

				updated ++;
			}				
		}
		

		// update normals map
		calculateNormal(Valid, NormalX, NormalY, std::max(0,best_x-update-1), std::max(0,best_y-update-1), std::min(width-1,best_x+update+1), std::min(height-1,best_y+update+1));			

		// update averages map
		if(earlyTerm)
		{
			calculateAverage(Filled, Valid, Average, std::max(0,best_x-radius-update), std::max(0,best_y-radius-update), std::min(width-1,best_x+radius+update), std::min(height-1,best_y+radius+update), radius);
		}
		
		// update gradients
		if(useDataTerm)
		{
			if(averagedGradient)
			{
				calculateGradient(Average, Valid, GradientX, GradientY, std::max(0,best_x-update-1), std::max(0,best_y-update-1), std::min(width-1,best_x+update+1), std::min(height-1,best_y+radius+update+1));			
			}
			else
			{
				calculateGradient(Filled, Valid, GradientX, GradientY, std::max(0,best_x-update-1), std::max(0,best_y-update-1), std::min(width-1,best_x+update+1), std::min(height-1,best_y+update+1));			
			}			
		}

		//calculateAverage(GradientX, Valid, GradientXAvg, std::max(0,best_x-radius-update), std::max(0,best_y-radius-update), std::min(width-1,best_x+radius+update), std::min(height-1,best_y+radius+update), radius);
		//calculateAverage(GradientY, Valid, GradientYAvg, std::max(0,best_x-radius-update), std::max(0,best_y-radius-update), std::min(width-1,best_x+radius+update), std::min(height-1,best_y+radius+update), radius);

		//holes -= updated;		
		iter ++;
		
		for(long i=0; i<HW;i++)
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
	
	//mexPrintf("Inpainted pixels: %d; using patches: %d; total matches: %d. \n", holes_saved-holes, iter, searches);
	
}

void calculateAverage(MexImage<float> &Image, MexImage<bool> &Valid, MexImage<float> &Average, const int x0, const int y0, const int x1, const int y1, const int radius)
{
	const int width = Image.width;
	const int height = Image.height;
	const int HW = Image.layer_size;
	const int colors = Image.layers;

	#pragma omp parallel for
	for(int x=std::max(0,x0); x <= std::min(width-1,x1); x++)
	{			
		for(int y=std::max(0,y0); y <= std::min(height-1,y1); y++)
		{
			long index = Image.Index(x,y);
			for(int c=0; c<colors; c++)
			{
				Average.data[index + c*HW] = 0;
			}
			int pixels = 0;

			for(int sx=std::max(0, x-radius); sx<=std::min(width-1, x+radius); sx++)
			{
				for(int sy=std::max(0, y-radius); sy<=std::min(height-1, y+radius); sy++)
				{
					long sindex = Average.Index(sx, sy);
					if(!Valid[sindex])
					{
						continue;
					}
					for(int c=0; c<colors; c++)
					{
						Average.data[index + c*HW] += Image[sindex + c*HW];
					}
					pixels ++;
				}
			}
			if(pixels > 0)
			{
				for(int c=0; c<colors; c++)
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
	for(int x=std::max(0,x0); x <= std::min(width-1,x1); x++)
	{			
		for(int y=std::max(0,y0); y <= std::min(height-1,y1); y++)
		{
			long index = Image.Index(x,y);
			
			if(!Valid[index])
				continue;

			GradientX.data[index] = 0;
			GradientY.data[index] = 0;

			int xl = std::max(0, x-1);
			int xr = std::min(width-1, x+1);
			int yu = std::max(0, y-1);
			int yd = std::min(height-1, y+1);

			long indexl = Image.Index(xl, y);
			long indexr = Image.Index(xr, y);
			long indexu = Image.Index(x, yu);
			long indexd = Image.Index(x, yd);

			indexl = Valid[indexl] ? indexl : index;
			indexr = Valid[indexr] ? indexr : index;
			indexu = Valid[indexu] ? indexu : index;
			indexd = Valid[indexd] ? indexd : index;

			for(int c=0; c<colors; c++)
			{
				GradientX.data[index] += (Image[indexl+c*HW] - Image[indexr+c*HW])/(colors*255);
				GradientY.data[index] += (Image[indexu+c*HW] - Image[indexd+c*HW])/(colors*255);
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
	const int diameter = radius*2+1;
	const int window = diameter*diameter;

	#pragma omp parallel for
	for(int x=std::max(radius,x0); x <= std::min(width-radius-1,x1); x++)
	{			
		for(int y=std::max(radius,y0); y <= std::min(height-radius-1,y1); y++)
		{
			long index = Valid.Index(x,y);
			
			if(!Valid[index])
				continue;

			int pixels = 0;

			for(int i=0; i<window; i++)
			{
				int sx = x + i/diameter - radius;
				int sy = y + i%diameter - radius;
				long sindex = Valid.Index(sx, sy);

				if(!Valid[sindex])
					break;

				pixels ++;
			}			
			//float conf = IntegralConf.getIntegralAverage(x,y,radius);
			if(pixels == window /*&& conf > 0.5*/)
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
	for(int x=std::max(0,x0); x <= std::min(width-1,x1); x++)
	{			
		for(int y=std::max(0,y0); y <= std::min(height-1,y1); y++)
		{
			long index = Valid.Index(x,y);			
			
			NormalX.data[index] = 0;
			NormalY.data[index] = 0;

			int xl = std::max(0, x-1);
			int xr = std::min(width-1, x+1);
			int yu = std::max(0, y-1);
			int yd = std::min(height-1, y+1);

			long indexl = Valid.Index(xl, y);
			long indexr = Valid.Index(xr, y);
			long indexu = Valid.Index(x, yu);
			long indexd = Valid.Index(x, yd);

			NormalX.data[index] = -((int8)Valid[indexl] - (int8)Valid[indexr]);
			NormalY.data[index] = -((int8)Valid[indexu] - (int8)Valid[indexd]);
			
		}
	}
}
