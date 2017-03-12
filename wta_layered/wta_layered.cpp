//wta_layered
/** wta_simple - Winner takes All with simple interfacing
*	@file wta_simple.cpp
*	@date 28.01.2011
*	@author Sergey Smirnov
*/

#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}
#endif // __cplusplus
#include "../common/meximage.h"
#include <utility>
//#include "../common/defines.h"
//#include "../common/matching.h"
using namespace mymex;

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")


void mexFunction (int nout, mxArray* output[], int in, const mxArray* input[])
{
	if (nout != 4 || in < 1 || in > 3 || mxGetClassID(input[0]) != mxSINGLE_CLASS)
	{
		mexErrMsgTxt("USAGE: [Disp1, Disp2, Cost1, Cost2] = wta_layered(Cost, <mindisp_to_be_added, min_dist>);"); 
    } 

	const int mindisp = (in > 1) ? (int) mxGetScalar(input[1]) : 0;
	const int mindist = (in > 2) ? abs((int)mxGetScalar(input[2])) : 1;
	const float nan = sqrt(-1);
	MexImage<float> Cost(input[0]);
	const mwSize depthdims[] = {Cost.height, Cost.width, 1};
	output[0] = mxCreateNumericArray(3, depthdims, mxSINGLE_CLASS, mxREAL);
	output[1] = mxCreateNumericArray(3, depthdims, mxSINGLE_CLASS, mxREAL);
	output[2] = mxCreateNumericArray(3, depthdims, mxSINGLE_CLASS, mxREAL);
	output[3] = mxCreateNumericArray(3, depthdims, mxSINGLE_CLASS, mxREAL);
	MexImage<float> Disp1(output[0]);
	MexImage<float> Disp2(output[1]);
	MexImage<float> BestCost1(output[2]);
	MexImage<float> BestCost2(output[3]);

	
	const int width = Cost.width;
	const int height = Cost.height;	
	const int layers = Cost.layers;
	const long HW = width*height;

	#pragma omp parallel for
	for (long i = 0; i<HW; i++)
	{
		const int x = i / height;
		const int y = i % height;
		BestCost1(x,y) = Cost(x,y,0);
		BestCost2(x,y) = Cost(x,y,0);
		Disp1(x,y) = 0;		
		Disp2(x,y) = 0;

		for(int d=1; d<layers; d++)
		{		
			const bool update1 = (Cost(x,y,d) < BestCost1(x,y));
			Disp1(x,y) = update1 ? d : Disp1(x,y);
			BestCost1(x,y) = update1 ? Cost(x,y,d) : BestCost1(x,y);				
		}
	}

	#pragma omp parallel for
	for (long i = 0; i<HW; i++)
	{
		const int x = i / height;
		const int y = i % height;
		int d1 = int(Disp1(x,y));		

		for(int d=1; d<layers; d++)
		{
			if(abs(d-d1) >= mindist)
			{
				// we look for remaining "local minimum" 
				bool update2 = (Cost(x,y,d) < BestCost2(x,y)) && Cost(x,y,d-1) > Cost(x,y,d);
				update2 = update2 && (d < layers-1 ? Cost(x,y,d+1) > Cost(x,y,d) : true);				

				Disp2(x,y) = update2 ? d : Disp2(x,y);
				BestCost2(x,y) = update2 ? Cost(x,y,d) : BestCost2(x,y);		
			}			
		}

		//if(Disp2(x,y) < Disp1(x,y))
		//{
		//	std::swap(Disp2(x,y), Disp1(x,y));
		//	std::swap(BestCost2(x,y), BestCost1(x,y));
		//}

		Disp2(x,y) += mindisp;
		Disp1(x,y) += mindisp;
	}
}