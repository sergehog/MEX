/** Trial on plane-fitting WTA
*	@file wta_plane.cpp
*	@date 12.03.2012
*	@author Sergey Smirnov
*/

#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}
#endif // __cplusplus

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")

#include "../common/meximage.h"
#include "../common/defines.h"
#ifndef _DEBUG
	#include <omp.h>
#endif

#include <math.h>
#include <algorithm>
#include <utility>

using namespace mymex;
using namespace std;
/*

void mexFunction (int nout, mxArray* output[], int in, const mxArray* input[])
{	
#ifndef _DEBUG
	omp_set_num_threads(std::max(2, omp_get_max_threads()));
	omp_set_dynamic(true);
#endif
	if(in < 4 || in > 6 || nout != 2 || mxGetClassID(input[0])!=mxSINGLE_CLASS || mxGetClassID(input[1])!=mxUINT8_CLASS)
	{
		mexErrMsgTxt("USAGE: [c, Cost] = wta_plane(single(Cost), uint8(I), a, b, [, mindisp, radius]");
    }

	MexImage<float> Cost(input[0]);
	MexImage<unsigned char> Image(input[1]);
	const int height = Cost.height;
	const int width = Cost.width;
	const int layers = Cost.layers;
	const int colors = Image.layers;
	const int HW = width*height;
	const int HW2 = HW*2;

	const float a = (float)mxGetScalar(input[2]);
	const float b = (float)mxGetScalar(input[3]);
	const int mindisp = (in > 4) ? (int)mxGetScalar(input[4]) : 0;
	const int radius = (in > 5) ? (int)mxGetScalar(input[5]) : 2;
    const int diameter = DIAMETER(radius);
    const int window = WINDOW(radius);
	const float sigma = 20.f;

	const mwSize cdims[] = {height, width, 1};
	output[0] = mxCreateNumericArray(3, cdims, mxSINGLE_CLASS, mxREAL); 
	output[1] = mxCreateNumericArray(3, cdims, mxSINGLE_CLASS, mxREAL); 

	MexImage<float> Disp(output[0]);
	MexImage<float> CostValues(output[1]);

	CostValues.setval(10000.f);
	Disp.setval(sqrt(-1.f));

	// filling of 1st layer Cost volumes
	#pragma omp parallel 
	{
		float *weights = new float[window];
		#pragma omp for
		for(int index=0; index<HW; index++)
		{
			int y = index % height;
			int x = index / height;	

			for(int i=0; i<window; i++) 
			{
				weights[i] = 0.f;
			}
			
			for(int dx=-radius, i=0; dx<=radius; dx++)
			{
				int xx = x + dx;
				if(xx<0 || xx>=width)
					continue;

				for(int dy=-radius; dy<=radius; dy++, i++)
				{
					int yy = y + dy;
					if(yy<0 || yy>=height)
						continue;
					int indexx = Cost.Index(xx, yy);
					int diff = 0;
					for(int c=0; c<colors; c++)
					{
						diff += abs(int(Image[indexx+c*HW])-int(Image[index+c*HW]));
					}
					
					weights[i] = exp(-diff/(3.f*sigma));
				}
			}
		
			for(int d=0; d<layers; d++)
			{
				float weights_sum = 0;
				float cost = 0;

				for(int dx=-radius, i=0; dx<=radius; dx++)
				{
					int xx = x + dx;
					if(xx<0 || xx>=width)
						continue;

					for(int dy=-radius; dy<=radius; dy++, i++)
					{
						int yy = y + dy;
						if(yy<0 || yy>=height)
							continue;
						int indexx = Cost.Index(xx, yy);
					
						float dd = a*dx + b*dy + d;					
						int dd_ceil = ceil(dd);					
						int dd_floor = floor(dd);

						dd_ceil = dd_ceil < 0 ? 0 : dd_ceil>=layers ? layers-1 : dd_ceil;
						dd_floor = dd_floor < 0 ? 0 : dd_floor>=layers ? layers-1 : dd_floor;

						//if(dd_ceil>=layers || dd_floor < 0)
						//	continue;

						float cost_ceil = Cost[indexx + HW*dd_ceil];
						float cost_floor = Cost[indexx + HW*dd_floor];
						float interpolated = (dd_floor == dd_ceil) ? cost_ceil : cost_ceil * (1-abs(dd_ceil-dd)) + cost_floor*(1-abs(dd-dd_floor));

						cost += interpolated*weights[i];
						weights_sum += weights[i];
					}
				}

			
				if(weights_sum > 0)
				{
					cost /= weights_sum;
					if(cost < CostValues[index])
					{						
						Disp.data[index] = d + mindisp;
						CostValues.data[index] = cost;
					} 
				}
			}

		
		}

		delete[] weights;
	}

}
*/