/** O(1)-complexity Block-Matching with SAD dissimilarity
*	@file o1_sad_matching.cpp
*	@date 7.10.2011
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


#include <cmath>
#include <algorithm>
#include <omp.h>
#include "../common/meximage.h"

typedef unsigned char uint8;
using namespace mymex;

void o1_zsad_match(MexImage<uint8> &Reference, MexImage<uint8> &Template, MexImage<double> &ReferenceI, MexImage<double> &TemplateI, int mindisp, int maxdisp, int direction, int radius, int radius_z, double cost_threshold, MexImage<float> &Disp, MexImage<float> *Conf)
{
	const float NAN = sqrt(-1.f);
	const int height = Reference.height;
	const int width = Reference.width;
	const int colors = Reference.layers;	
	const int HW = Reference.layer_size;
	const int layers = maxdisp - mindisp + 1;

	// create buffers for calculation and for integration of current cost slice
	MexImage<double> *Cost = new MexImage<double>(width, height);

	MexImage<float> *CostMin = new MexImage<float>(width, height);;  // best cost values
	MexImage<float> *CostPrev = new MexImage<float>(width, height); // cost values at d-1 slice, where d=optimal
	MexImage<float> *CostNext = new MexImage<float>(width, height); // cost values at d+1 slice, where d=optimal
	MexImage<float> *CostMin2 = new MexImage<float>(width, height); // second minimum cost value (not connected to optimal d)
	MexImage<float> *CostMin3 = new MexImage<float>(width, height); // second minimum cost value (not connected to optimal d)
	// Since CostLast is a temporal buffer we'll utilize Confidence to save some memory
	MexImage<float> *CostLast = Conf;//new MexImage<float>(width, height);

	Cost->setval(cost_threshold);
	CostLast->setval((float)cost_threshold);
	CostMin->setval((float)cost_threshold);
	CostPrev->setval((float)cost_threshold);
	//CostNext->setval((float)cost_threshold);
	CostMin2->setval((float)cost_threshold);
	CostMin3->setval((float)cost_threshold);
	Disp.setval(NAN);

	for(int d=0; d<layers; d++)
	{
		Cost->setval(cost_threshold);

		int disp = d + mindisp;
		//#pragma omp parallel for
		for(int index=0; index<HW; index++)
		{
			int y = index % height;
			int x = index / height;

			//x-coordinate in template image
			int xT = x + direction*disp; 

			if(xT >= 0 && xT < width)				
			{
				int indexT = Template.Index(xT, y);
				double cost = 0;
				double refMean[3];
				double templMean[3];
				ReferenceI.getIntegralAverage(x, y, radius_z, refMean);
				TemplateI.getIntegralAverage(xT, y, radius_z, templMean);
				for(int c=0; c<colors; c++)
				{
					int cHW = c*HW;
					double ref = (double)Reference.data[index + cHW] - refMean[c];
					double tmpl = (double)Template.data[indexT + cHW] - templMean[c];
					cost += std::abs(ref - tmpl);
				}
				cost /= colors;
				Cost->data[index] = (cost > cost_threshold) ? cost_threshold : cost;
			}
			else
			{
				Cost->data[index] = cost_threshold;
			}
		}

		Cost->IntegralImage(*Cost);

		#pragma omp parallel for
		for(int index=0; index<HW; index++)
		{
			int y = index % height;
			int x = index / height;
			float agg_cost = (float)Cost->getIntegralAverage(x, y, radius);

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
	for(int index=0; index<HW; index++)
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
	omp_set_num_threads(std::min(2, omp_get_num_threads()));
	omp_set_dynamic(0);
	
	if(in < 4 || in > 7 || nout < 2 || nout > 4 || mxGetClassID(input[0])!=mxUINT8_CLASS || mxGetClassID(input[1])!=mxUINT8_CLASS)
	{
		mexErrMsgTxt("USAGE: [DispL, DispR, <ConfL, ConfR>] = o1_zsad_matching(uint8(Left), uint8(Right), mindisp, maxdisp[, radius, radius_z, cost_threshold]);");
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
	const int radius_z = (in > 5) ? std::max(1, (int)mxGetScalar(input[5])) : 1;	
	const double cost_threshold = (in > 6) ? (double)mxGetScalar(input[6]) : 40.;

	if(maxdisp <= mindisp)
	{
		mexErrMsgTxt("ERROR: maxdisp must be larger than mindisp");
	}

	const unsigned depthdims[] = {(unsigned)height, (unsigned)width, 1};

	output[0] = mxCreateNumericArray(3, depthdims, mxSINGLE_CLASS, mxREAL); 
	output[1] = mxCreateNumericArray(3, depthdims, mxSINGLE_CLASS, mxREAL);
	MexImage<float> DispL(output[0]);
	MexImage<float> DispR(output[1]);
	MexImage<float> * ConfL = NULL;
	MexImage<float> * ConfR = NULL;

	MexImage<double> LeftI(width, height, colors);
	MexImage<double> RightI(width, height, colors);
	LeftI.IntegralImage<uint8>(Left);
	RightI.IntegralImage<uint8>(Right);


	// Paralallizes Left-to-Right and Right-to-Left matching		
	#pragma omp parallel sections num_threads(2)
	{
		#pragma omp section
		{
			if(nout > 2)
			{
				output[2] = mxCreateNumericArray(3, depthdims, mxSINGLE_CLASS, mxREAL);
				ConfL = new MexImage<float>(output[2]);		
			}
			else
			{
				ConfL = new MexImage<float>(width, height);
			}	

			o1_zsad_match(Left, Right, LeftI, RightI, mindisp, maxdisp, -1, radius, radius_z, cost_threshold, DispL, ConfL);
		}

		#pragma omp section
		{
			if(nout > 3)
			{
				output[3] = mxCreateNumericArray(3, depthdims, mxSINGLE_CLASS, mxREAL); //
				ConfR = new MexImage<float>(output[3]);
			}
			else
			{
				ConfR = new MexImage<float>(width, height);
			}
			o1_zsad_match(Right, Left, RightI, LeftI, mindisp, maxdisp, 1, radius, radius_z, cost_threshold, DispR, ConfR);
		}
	}
	
	delete ConfL;
	delete ConfR;
}
