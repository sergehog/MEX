/** Simple hole filling algorithm. 
*	@file hole_filling.cpp
*	@date 02.07.2012
*	@author Sergey Smirnov <sergey.smirnov@tut.fi>
*/

#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}
#endif // __cplusplus

#include "../common/meximage.h"
#include <float.h>
#include <vector>
#include <cmath>
//#include <algorithm>
#include <omp.h>

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")

using namespace mymex;

static int height;
static int width;
static int layers;
static long HW;
static float nan;


//! represents either some active pixel or center of a patch
struct ImagePixel
{
	int x; 
	int y;
	long index;

	//! multi-purpose value, used for sorting/comparing
	float priority; 

public: 
	// default values are not valid
	ImagePixel() : x(-1), y(-1), priority(0.f)
	{
	};

	// copying constructor
	ImagePixel(ImagePixel &smpl) : x(smpl.x), y(smpl.y), priority(smpl.priority)
	{
	};

	bool isValid()
	{
		return x>=0 && y>=0;
	}

	bool operator==(ImagePixel &pixel)
	{
		return pixel.x==x && pixel.y==y;
	}
};


//! finds the most appropriate coordiante to fill-in.
ImagePixel findNextHole(MexImage<float> &Filled, MexImage<bool> &Mask, MexImage<float> &IntegralMask, const int radius)
{
	ImagePixel pixel;
	pixel.x = -1;
	pixel.y = -1;
	pixel.priority = 0.f;

	const int height = Filled.height;	
	const long HW = Filled.layer_size;

	for(long i=0; i<HW; i++)
	{		
		if(!Mask[i])
		{
			int x = i / height;
			int y = i % height;			
			
			float priority = IntegralMask.getIntegralAverage(x, y, radius);

			if(priority > pixel.priority && priority < 1.f)
			{
				pixel.priority = priority;
				pixel.x = x;
				pixel.y = y;
			}
		}
	}

	return pixel;
}


bool hasHoles(MexImage<bool> &Mask)
{
	for(long i=0; i<HW; i++)
	{
		if(!Mask[i])
		{
			return true;
		}
	}

	return false;
}

ImagePixel findBestPatch(MexImage<float> &Signal, MexImage<bool> &Mask, MexImage<float> &IntegralMask, ImagePixel& copyTo, const int radius, const int search)
{	
	const int xt = copyTo.x;
	const int yt = copyTo.y;	
	const long indext = Signal.Index(xt, yt);

	const int diameter = radius*2+1;
	const int window = diameter*diameter;	
	//const float sigma_distance = 2.f;
	
	ImagePixel bestPatch;
	bestPatch.x = -1;
	bestPatch.y = -1;
	bestPatch.priority = FLT_MAX;

	int valid_pixels = 0;

	for(int x = std::max(radius, xt-search) ; x < std::min(width-radius, xt+search); x++)
	{
		for(int y = std::max(radius, yt-search); y < std::min(height-radius, yt+search); y++)
		{
			long index = Signal.Index(x, y);		
						
			float pixels = IntegralMask.getIntegralAverage(x, y, radius);
			
			if(pixels < 0.999f)
				continue;			

			float distance = 0;
			valid_pixels = 0;

			for(int dx = -radius; dx <= radius; dx++)
			{
				#pragma omp parallel for reduction(+: distance, valid_pixels)
				for(int dy = -radius; dy <= radius; dy++)
				{
					int xd = x + dx;
					int yd = y + dy;
					int xtd = xt + dx;
					int ytd = yt + dx;

					if(xd<0 || yd < 0 || xtd < 0 || ytd < 0)
						continue;
					
					if(xd >= width || xtd >= width)
						continue;

					if(yd >= height || ytd >= height)
						continue;

					long indexd = Signal.Index(xd, yd);
					long indextd = Signal.Index(xtd, ytd);

					if(!Mask[indextd] || !Mask[indexd])
						continue;				
					
					valid_pixels ++;
					
					#pragma omp parallel for reduction(+: distance)
					for(int c=0; c<layers; c++)
					{
						long cHW = HW*c;
						distance += std::abs(Signal[indexd+cHW]-Signal[indextd+cHW]);
					}
				}
			}

			//distance /= layers*valid_pixels;
			distance /= valid_pixels;

			if(distance < bestPatch.priority)
			{
				bestPatch.x = x;
				bestPatch.y = y;
				bestPatch.priority = distance;
			}
		}
	}

	if (bestPatch.x >=0)
	{
		return bestPatch;
	}
	else if(radius > 1)
	{
		return findBestPatch(Signal, Mask, IntegralMask, copyTo, radius-1, search);
	}
	//else 
	//	return findBestPatch(Signal, Mask, IntegralMask, copyTo, radius, search+1);
	return bestPatch;	
}

void applyPatch(MexImage<float> &Signal, MexImage<bool> &Mask, ImagePixel& copyTo, ImagePixel& copyFrom, const int radius)
{
	for(int dx=-radius; dx<=radius; dx++)
	{
		for(int dy=-radius; dy<=radius; dy++)
		{
			int xT = copyTo.x + dx;
			int yT = copyTo.y + dy;
			int xF = copyFrom.x + dx;
			int yF = copyFrom.y + dy;

			if(xT < 0 || yT < 0 || xF < 0 || yF < 0)
				continue;
				
			if(xT >= width || xF >= width)				
				continue;

			if(yT >= height || yF >= height)
				continue;

			long indexT = Signal.Index(xT, yT);
			long indexF = Signal.Index(xF, yF);

			if(!Mask[indexT])
			{
				for(int c=0; c<layers; c++)
				{
					Signal.data[indexT + c*HW] = Signal.data[indexF + c*HW];
				}

				Mask.data[indexT] = true;				
			}
		}
	}
}

void updateMask(MexImage<float> &Signal, MexImage<bool> &Mask)
{
	for(long i=0; i<HW; i++)
	{
		bool hole = false;
		for(int c=0; c<layers; c++)
		{
			hole = hole || _isnan(Signal[i+ c*HW]);
		}
		
		Mask.data[i] = hole ? false : true;
	}
}

void integrateMask(MexImage<bool> &Mask, MexImage<float> &IntegralMask)
{
	for(long i=0; i<HW; i++)
	{
		IntegralMask.data[i] = Mask[i] ? 1.f : 0.f;
	}

	IntegralMask.IntegralImage(true);
}

void mexFunction (int nout, mxArray* output[], int in, const mxArray* input[])
{	
	omp_set_num_threads(std::max(2, omp_get_num_threads()));
	omp_set_dynamic(0);

	if(in < 1 || in > 4 || nout != 1 || mxGetClassID(input[0]) != mxSINGLE_CLASS)
	{
		mexErrMsgTxt("USAGE: Without_NaNs = hole_filling_simple(single(Signal_with_NaNs), <radius, search, max_iterations>); ");//, aggregate, number
    }

	MexImage<float> Signal(input[0]);
	height = Signal.height;
	width = Signal.width;
	layers = Signal.layers;
	HW = Signal.layer_size;	
	nan = sqrt(-1.f);	
		
	const int radius = (in > 1) ? std::min(width/2,std::max(2, (int)mxGetScalar(input[1]))) : 5;	
	const int search = (in > 2) ? (int)mxGetScalar(input[2]) : 10;	
	const long max_iterations = (in > 3) ? (long)mxGetScalar(input[3]) : 0;	
	
	mexPrintf("radius=%d;\nsearch=%d;\nmax_iterations=%d\n", radius, search, max_iterations);
	//const int number = (in+paramsOffset > 4) ? std::max(3, (int)mxGetScalar(input[3+paramsOffset])) : 3;	

	size_t dims2d[] = {(unsigned)height, (unsigned)width, (unsigned)layers};

	output[0] = mxCreateNumericArray(3, dims2d, mxSINGLE_CLASS, mxREAL);
	MexImage<float> Filled(output[0]);

	// pre-fill output image
	for(long i=0; i<HW*layers; i++)
	{
		Filled.data[i] = Signal[i];
	}

	MexImage<bool> Mask(width, height);
	MexImage<float> IntegralMask(width, height);	
	updateMask(Signal, Mask);
	

	int n = 0; 
	while(hasHoles(Mask) && (max_iterations == 0 || n < max_iterations))
	{		
		
		integrateMask(Mask, IntegralMask);		

		//find most appropriate pixel to fill-in
		ImagePixel copyTo = findNextHole(Filled, Mask, IntegralMask, radius);
		ImagePixel copyFrom = findBestPatch(Filled, Mask, IntegralMask, copyTo, radius, search);
				
		applyPatch(Filled, Mask, copyTo, copyFrom, radius);

		mexPrintf("%d: (%d,%d|%5.2f)  <-- (%d,%d|%5.2f)\n", n, copyTo.x, copyTo.y, copyTo.priority,  copyFrom.x, copyFrom.y, copyFrom.priority);

		n++;
	}
}
