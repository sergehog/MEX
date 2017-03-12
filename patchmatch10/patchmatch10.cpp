/** 
* @file patchmatch10.cpp
* @author Sergey Smirnov <sergey.smirnov@tut.fi> / 3D Media Group / Tampere University of Technology
* @date 31.10.2014
*/


#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}
#endif // __cplusplus

#include <algorithm>
#include <memory>
#include <stdlib.h>
#include <time.h>

#ifndef _DEBUG
#include <omp.h>
#endif
#include "../common/meximage.h"

#define isnan _isnan
#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")
//#pragma warning(disable:4244)
//#pragma warning(disable:4018)

using namespace mymex;
//#define cost_thr 20.f
//#define DEBUG
//#define MAXDIFF 255*3+1
#define N 10

float get_cost(const MexImage<float> &Image, const int x, const int y, const short dx, const short dy, const int radius)
{
	const int width = Image.width;
	const int height = Image.height;
	const int colors = Image.layers;

	float cost = 0;
	int pixels = 0;
	for(int rx=-radius; rx<=radius; rx++)
	{
		for(int ry=-radius; ry<=radius; ry++)
		{
			const int x1 = x + rx;
			const int x2 = x + rx + dx;
			const int y1 = y + ry;
			const int y2 = y + ry + dy;

			if(x1 <0 || x1 >= width || x2<0 || x2 >= width)
			{
				continue;
			}

			if(y1 <0 || y1 >= height || y2<0 || y2 >= height)
			{
				continue;
			}

			pixels ++;
			for(int c=0; c<colors; c++)
			{
				cost += abs(Image(x1,y1,c) - Image(x2,y2,c));
			}
		}
	}

	return cost/(pixels*colors);
}

void sorted_push(MexImage<short> &DX, MexImage<short> &DY, MexImage<float> &COST, const int x, const int y, short dx, short dy, float cost)
{
	// sorted push
	for(int l=0; l<N; l++)
	{
		// hypothesis already in the list
		if(DX(x,y,l) == dx && DY(x,y,l) == dy) 
		{
			break;
		}

		// enforse sorted COST
		if(cost < COST(x,y,l))
		{
			int dx0 = DX(x,y,l);
			int dy0 = DY(x,y,l);
			float cost0 = COST(x,y,l);

			DX(x,y,l) = dx;
			DY(x,y,l) = dy;
			COST(x,y,l) = cost;

			dx = dx0;
			dy = dy0;
			cost = cost0;
		}								
	}
}

void mexFunction (int nout, mxArray* output[], int in, const mxArray* input[])
{
	#ifndef _DEBUG
	omp_set_num_threads(std::max(4,omp_get_max_threads())); 
	omp_set_dynamic(std::max(2, omp_get_max_threads() / 2));
	#endif

	const int _offset = 6; // 5 obligatory params goes before variable numner of camera pairs

	if (in != 3 || nout != 4 || mxGetClassID(input[0]) != mxSINGLE_CLASS || mxGetClassID(input[1]) != mxSINGLE_CLASS)
	{
		mexPrintf("Returns 10 best matches for each pixel of Reference Image \n");
		mexErrMsgTxt("USAGE: [Filtered, DX, DY, BestCost] = patchmath10(single(Signal), single(Image), patch_radius);");
	}
	
	MexImage<float> Signal(input[0]);
	MexImage<float> Image(input[1]);
	const int patch = std::max(1,static_cast<int>(mxGetScalar(input[2])));	
	const int width = Signal.width;
	const int height = Signal.height;
	const int layers = Signal.layers;
	const int colors = Image.layers;
	const long HW = Signal.layer_size;

	if(Image.width != width || Image.height != height)
	{
		mexErrMsgTxt("Resolution of Signal and Image does not coincide!");
	}

	const mwSize dim[] = {(size_t)height, (size_t)width, N};
	const mwSize dimL[] = {(size_t)height, (size_t)width, layers};
		
	//Matlab-allocated variables
	output[0] = mxCreateNumericArray(3, dimL, mxSINGLE_CLASS, mxREAL);
	output[1] = mxCreateNumericArray(3, dim, mxINT16_CLASS, mxREAL);
	output[2] = mxCreateNumericArray(3, dim, mxINT16_CLASS, mxREAL);
	output[3] = mxCreateNumericArray(3, dim, mxSINGLE_CLASS, mxREAL);

	MexImage<float> Filtered(output[0]);
	MexImage<short> DX(output[1]);
	MexImage<short> DY(output[2]);
	MexImage<float> COST(output[3]);
	DX.setval(0);
	DY.setval(0);
	COST.setval(10000000);
	unsigned *a = new unsigned(5);
	*a = unsigned(a); // adds more randomness to the initial seed :-)
	srand (time(NULL) + HW + size_t(*a));
	delete a;
	
	const int search = std::max(width, height);
	//while(search >= 10)
	unsigned long updated = 1;
	int iter = 0;
	//while(updated > 10 && iter < 2)
	{
		updated = 0;
		iter ++;
		
		// check 10 random offsets for each pixel
		#pragma omp parallel for
		for(long i=0; i<HW; i++)
		{
			const int x = i / height;
			const int y = i % height;

			for(int l=0; l<N; l++)
			{
				//int dx = rand()%(max-min) + (max+min)/2-x;
				const int minX = patch;//std::max(0, x-search);
				const int maxX = width-patch-1;//std::min(width-1, x+search);
				const int minY = patch;//std::max(0, y-search);
				const int maxY = height-patch-1;//std::min(height-1, y+search);

				const short dx = rand()%(maxX-minX) + (maxX+minX)/2-x;						
				const short dy = rand()%(maxY-minY)-(maxY+minY)/2-y;
				const float cost = get_cost(Image, x, y, dx, dy, patch);

				sorted_push(DX, DY, COST, x, y, dx, dy, cost);				
			}
		}

		// check neighbours
		#pragma omp parallel for
		for(long i=0; i<HW; i++)
		{
			const int x = i / height;
			const int y = i % height;
			
			for(int l=0; l<N && (x > 0); l++)
			{
				const short dx = DX(x-1, y, l);
				const short dy = DY(x-1, y, l);

				if(x + dx >= 0 && x + dx < width && y + dy >= 0 && y + dy <= height)
				{
					float cost = get_cost(Image, x, y, dx, dy, patch);
					sorted_push(DX, DY, COST, x, y, dx, dy, cost);
				}
			}
			

			for(int l=0; l<N && (x<width-1); l++)
			{
				const short dx = DX(x+1, y, l);
				const short dy = DY(x+1, y, l);

				if(x + dx >= 0 && x + dx < width && y + dy >= 0 && y + dy <= height)
				{
					float cost = get_cost(Image, x, y, dx, dy, patch);
					sorted_push(DX, DY, COST, x, y, dx, dy, cost);
				}
			}

			for(int l=0; l<N && (y > 0); l++)
			{
				const short dx = DX(x, y-1, l);
				const short dy = DY(x, y-1, l);

				if(x + dx >= 0 && x + dx < width && y + dy >= 0 && y + dy <= height)
				{
					float cost = get_cost(Image, x, y, dx, dy, patch);
					sorted_push(DX, DY, COST, x, y, dx, dy, cost);
				}
			}

			for(int l=0; l<N && (y<height-1); l++)
			{
				const short dx = DX(x, y+1, l);
				const short dy = DY(x, y+1, l);

				if(x + dx >= 0 && x + dx < width && y + dy >= 0 && y + dy <= height)
				{
					float cost = get_cost(Image, x, y, dx, dy, patch);
					sorted_push(DX, DY, COST, x, y, dx, dy, cost);
				}
			}
			
			
		}

		//search = search/2;
	}

	Filtered.setval(0.f);
	float const sigma = 5;

	#pragma omp parallel for
	for(long i=0; i<HW; i++)
	{
		const int x = i / height;
		const int y = i % height;
		float weights = 0.f;

		for(int l=0; l<N; l++)
		{
			const short dx = DX(x, y, l);
			const short dy = DY(x, y, l);
			const float weight = exp(COST(x, y, l)/sigma);
			weights += weight;
			for(int s=0; s<layers; s++)
			{
				Filtered(x, y, s) += Signal(x+dx, y+dy, s) * weight;
			}						
		}

		for(int s=0; s<layers; s++)
		{
			Filtered(x, y, s) /= weights;
		}
		
	}

}