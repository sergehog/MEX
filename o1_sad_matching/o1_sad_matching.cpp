/** O(1)-complexity Block-Matching with SAD dissimilarity
*	@file o1_sad_matching.cpp
*	@date 7.10.2011
*	@author Sergey Smirnov <sergey.smirnov@tut.fi>
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
#include <cmath>
#include <algorithm>
#include <omp.h>

typedef unsigned char uint8;
using namespace mymex;

void o1_sad_match(MexImage<uint8> &Reference, MexImage<uint8> &Template, int mindisp, int maxdisp, int direction, int radius, float cost_threshold, MexImage<float> &Disp, MexImage<float> *Conf)
{
	//const float NAN = sqrt(-1.f);
	const int height = Reference.height;
	const int width = Reference.width;
	const int colors = Reference.layers;	
	const int HW = Reference.layer_size;
	const int layers = maxdisp - mindisp + 1;

	// create buffers for calculation and for integration of current cost slice
	MexImage<float> *Cost = new MexImage<float>(width, height);
	MexImage<float> *CostMin = new MexImage<float>(width, height);;  // best cost values
	MexImage<float> *CostPrev = new MexImage<float>(width, height); // cost values at d-1 slice, where d=optimal
	MexImage<float> *CostNext = new MexImage<float>(width, height); // cost values at d+1 slice, where d=optimal
	MexImage<float> *CostMin2 = new MexImage<float>(width, height); // second minimum cost value (not connected to optimal d)
	MexImage<float> *CostMin3 = new MexImage<float>(width, height); // second minimum cost value (not connected to optimal d)
	// Since CostLast is a temporal buffer we'll utilize Confidence to save some memory
	MexImage<float> *CostLast = Conf;//new MexImage<float>(width, height);

	//Cost->setval(cost_threshold);
	CostLast->setval((float)cost_threshold);
	CostMin->setval((float)cost_threshold);
	CostPrev->setval((float)cost_threshold);
	CostNext->setval((float)cost_threshold);
	CostMin2->setval((float)cost_threshold);
	CostMin3->setval((float)cost_threshold);
	//Disp.setval(NAN);

	for(int d=0; d<layers; d++)
	{
		Cost->setval(cost_threshold);
		//Cost->setval(0.f);

		int disp = d + mindisp;
		#pragma omp parallel for
		for(long index=0; index<HW; index++)
		{
			int y = index % height;
			int x = index / height;

			//x-coordinate in template image
			int xT = x + direction*disp; 
			xT = (xT < 0) ? 0 : xT;
			xT = (xT >= width) ? width-1 : xT;

			//if(xT >= 0 && xT < width)				
			{
				long indexT = Template.Index(xT, y);
				float cost = 0;
				for(int c=0; c<colors; c++)
				{
					long cHW = c*HW;
					int diff = (int)(Reference.data[index + cHW]) - (int)(Template.data[indexT + cHW]);
					cost += ((diff > 0) ? diff : - diff);
				}
				cost /= colors;
				Cost->data[index] = (cost > cost_threshold) ? cost_threshold : cost;
			}
			//else
			//{
			//	Cost->data[index] = cost_threshold;
			//}
		}


		if(radius > 0)
		{
			Cost->IntegralImage(true);
		}

		//int x_low = (direction > 0) ? 0 : disp;
		//int x_up = (direction > 0) ? width-disp-1 : width-1;
		
		#pragma omp parallel for
		for(long index=0; index<HW; index++)
		{
			long y = index % height;
			long x = index / height;
			float last_valid_cost = CostLast->data[index];
			//float agg_cost = (float)Cost->getBoundedIntegralAverage(x, y, radius, x_low, x_up, 0, height-1, last_valid_cost);
			
			float agg_cost = 0;
			if(radius > 0)
			{
				agg_cost += (float)Cost->getIntegralAverage(x, y, radius);
			}
			else
			{
				agg_cost += Cost->data[index];
			}						

			if(agg_cost < (CostMin->data[index]))
			{
				CostMin3->data[index] = CostMin2->data[index];
				CostMin2->data[index] = CostMin->data[index];
				CostMin->data[index] = agg_cost;
				CostPrev->data[index] = CostLast->data[index];
				CostNext->data[index] = (float)cost_threshold;
				Disp.data[index] = (float)disp;					
			}
			else if(agg_cost < (CostMin2->data[index]))
			{
				CostMin3->data[index] = CostMin2->data[index];
				CostMin2->data[index] = agg_cost;
			}
			else if(agg_cost < (CostMin3->data[index]))
			{
				CostMin3->data[index] = agg_cost;
			}

			if(disp == ((int)(Disp.data[index]))+1)
			{
				CostNext->data[index] = agg_cost;
			}
			CostLast->data[index] = agg_cost;
		}
	}

	// no need in these buffers any more
	//delete CostLast;
	CostLast = NULL; 
	delete Cost; Cost = NULL;	

	#pragma omp parallel for
	for(long index=0; index<HW; index++)
	{
		//int y = index % height;
		//int x = index / height;
		float costMin = CostMin->data[index];
		float costMin2 = CostMin2->data[index];
		float costMin3 = CostMin3->data[index];		
		float costPrev = CostPrev->data[index];
		float costNext = CostNext->data[index];

		// sub-pixel interpolation should be applied
		if(std::min(costPrev, costNext) <= costMin2)
		{
			Disp.data[index] = Disp.data[index]-(costNext-costPrev)/(2*(costNext+costPrev-2*costMin));
			Conf->data[index] = (costMin3-costMin)/costMin3;
		}
		else
		{
			Conf->data[index] = (costMin2-costMin)/costMin2;
		}
	}	
	
	delete CostMin;
	delete CostPrev;
	delete CostNext;
	delete CostMin2;
	delete CostMin3;
}

void mexFunction (int nout, mxArray* output[], int in, const mxArray* input[])
{
	omp_set_num_threads(std::min(2, omp_get_max_threads()));
	omp_set_dynamic(0);
	
	if(in < 4 || in > 6 || nout < 2 || nout > 4 || mxGetClassID(input[0])!=mxUINT8_CLASS || mxGetClassID(input[1])!=mxUINT8_CLASS)
	{
		mexErrMsgTxt("USAGE: [DispL, DispR, <ConfL, ConfR>] = o1_sad_matching(uint8(Left), uint8(Right), mindisp, maxdisp[, radius, cost_threshold]);");
    }

	MexImage<uint8> Left(input[0]);
	MexImage<uint8> Right(input[1]);
	const int height = Left.height;
	const int width = Left.width;
	const int colors = Left.layers;
	//const int HW = Left.layer_size;

	if(Right.height != height || Right.width != width || Right.layers != colors)
	{
		mexErrMsgTxt("ERROR: Input images must be of the same size!");
	}
		
	const int mindisp = static_cast<int>(mxGetScalar(input[2]));
	const int maxdisp = static_cast<int>(mxGetScalar(input[3]));
	//const int layers = maxdisp - mindisp + 1;
	const int radius = (in > 4) ? std::max(0, (int)mxGetScalar(input[4])) : 1;
	const float cost_threshold = (in > 5) ? (float)mxGetScalar(input[5]) : 100.;

	if(maxdisp <= mindisp)
	{
		mexErrMsgTxt("ERROR: maxdisp must be larger than mindisp");
	}

	const size_t depthdims[] = {(unsigned)height, (unsigned)width, 1};

	output[0] = mxCreateNumericArray(3, depthdims, mxSINGLE_CLASS, mxREAL); 
	output[1] = mxCreateNumericArray(3, depthdims, mxSINGLE_CLASS, mxREAL);
	MexImage<float> DispL(output[0]);
	MexImage<float> DispR(output[1]);
	MexImage<float> * ConfL = NULL;
	MexImage<float> * ConfR = NULL;
					
	if(nout > 2)
	{
		output[2] = mxCreateNumericArray(3, depthdims, mxSINGLE_CLASS, mxREAL);
		ConfL = new MexImage<float>(output[2]);		
	}
	else
	{
		ConfL = new MexImage<float>(width, height);
	}	
	if(nout > 3)
	{
		output[3] = mxCreateNumericArray(3, depthdims, mxSINGLE_CLASS, mxREAL);
		ConfR = new MexImage<float>(output[3]);
	}
	else
	{
		ConfR = new MexImage<float>(width, height);
	}


	// Paralallizes Left-to-Right and Right-to-Left matching		
	#pragma omp parallel sections num_threads(2)
	{
		#pragma omp section
		{

			o1_sad_match(Left, Right, mindisp, maxdisp, -1, radius, cost_threshold, DispL, ConfL);
		}

		#pragma omp section
		{
			o1_sad_match(Right, Left, mindisp, maxdisp, 1, radius, cost_threshold, DispR, ConfR);
		}
	}
	
	delete ConfL;
	delete ConfR;
}
