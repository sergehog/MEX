/** optical_flow.cpp - simple and fast optical flow calculation and motion compensation
    @author Sergey Smirnov
    @date 14.01.2010
	@rewritten 27.05.2013 
*/

#include "../common/meximage.h"
#ifndef _NDEBUG
#include <omp.h>
#endif

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")

void findShifts(mymex::MexImage<float> &FrameA, mymex::MexImage<float> &FrameB, mymex::MexImage<float> &ShiftsAB, mymex::MexImage<float> &ConfAB, const int search, const int radius)
{
	const int sdiameter = search*2+1;
	const int swindow = sdiameter*sdiameter;
	const int diameter = radius*2+1;
	const int window = diameter*diameter;
	const int width = FrameA.width;
	const int height = FrameA.height;
	const int colors = FrameA.layers;
	const long HW = height * width;

	mymex::MexImage<float> Cost(width, height);
	mymex::MexImage<float> MinCost(width, height);
	mymex::MexImage<float> Min2Cost(width, height);

	#pragma omp parallel sections
	{		
		#pragma omp section
		{
			Cost.setval(0.f);
		}
		#pragma omp section
		{
			MinCost.setval(256.f);
		}
		#pragma omp section
		{
			Min2Cost.setval(256.f);
		}
		#pragma omp section
		{
			ConfAB.setval(1.f);
		}		
	}

	
	for(int sw=0; sw<swindow; sw++)
	{
		const int shiftX = sw / sdiameter - search;
		const int shiftY = sw % sdiameter - search;

		// calculate cost 
		#pragma omp parallel for
		for(long indexA=0; indexA<HW; indexA++)
		{
			const int xA = indexA / height;
			const int yA = indexA % height;

			const int xB = (xA + shiftX) < 0 ? 0 : (xA + shiftX)>=width ? width-1 : (xA + shiftX);
			const int yB = (yA + shiftY) < 0 ? 0 : (yA + shiftY)>=height ? height-1 : (yA + shiftY);
			const long indexB = FrameB.Index(xB, yB);

			int diff = 0;
			for(int c=0; c<colors; c++)
			{
				diff += abs(FrameA[indexA+c*HW] - FrameB[indexB+c*HW]);
			}

			Cost.data[indexA] = float(diff)/colors;
		}

		Cost.IntegralImage(true);

		// calculate cost 
		#pragma omp parallel for
		for(long indexA=0; indexA<HW; indexA++)
		{
			const int xA = indexA / height;
			const int yA = indexA % height;

			 float avgcost = Cost.getIntegralAverage(xA,yA,radius);

			if(avgcost <= MinCost[indexA])
			{
				Min2Cost.data[indexA] = MinCost[indexA];
				MinCost.data[indexA] = avgcost;
				ShiftsAB.data[indexA] = shiftX;
				ShiftsAB.data[indexA+HW] = shiftY;
				ConfAB.data[indexA] = (Min2Cost[indexA]-avgcost)/Min2Cost[indexA];
				//ConfAB.data[indexA] = MinCost[indexA];
			}
			else if(avgcost <= Min2Cost[indexA])
			{
				Min2Cost.data[indexA] = avgcost;				
				ConfAB.data[indexA] = (avgcost-MinCost[indexA])/avgcost;
				//ConfAB.data[indexA] = avgcost;
			}
		}
	}
}

void correspondShifts(mymex::MexImage<float> &ShiftsAB, mymex::MexImage<float> &ShiftsBA, mymex::MexImage<float> &ConfAB, const int thr)
{
	const int width = ShiftsAB.width;
	const int height = ShiftsAB.height;
	const long HW = height * width;

	#pragma omp parallel for
	for(long indexA=0; indexA<HW; indexA++)
	{
		const int xA = indexA / height;
		const int yA = indexA % height;

		const int shiftXA = mymex::round(ShiftsAB[indexA]);
		const int shiftYA = mymex::round(ShiftsAB[indexA+HW]);

		const int xB = xA + shiftXA;
		const int yB = yA + shiftYA;

		if(xB < 0 || xB >= width || yB<0 || yB>=height)
		{
			ConfAB.data[indexA] = 0;
			continue;
		}

		const long indexB = ShiftsBA.Index(xB, yB);

		if(abs(ShiftsAB[indexA]+ShiftsBA[indexB]) > thr || abs(ShiftsAB[indexA+HW]+ShiftsBA[indexB+HW]) > thr)
		{
			ConfAB.data[indexA] = 0;
		}
	}
}

void mexFunction (int nout, mxArray* output[], int in, const mxArray* input[])
{
#ifndef _NDEBUG
	omp_set_num_threads(omp_get_max_threads() > 2 ? omp_get_max_threads() : 2);
	omp_set_dynamic(true);
#endif

	if(in < 2 || in > 4  || nout != 4 || mxGetClassID(input[0]) != mxSINGLE_CLASS || mxGetClassID(input[1]) != mxSINGLE_CLASS)
	{
		mexErrMsgTxt("USAGE: [Shifts12, Shifts21, Conf12, Conf21] = optical_flow(single(Frame1), single(Frame2), <search, radius>);"); 
    }

	mymex::MexImage<float> Frame1 (input[0]);
	mymex::MexImage<float> Frame2 (input[1]);

	const int width = Frame1.width;
	const int height = Frame1.height;
	const int colors = Frame1.layers;	
	const long HW = width*height;
    const int thr = 1;

	if(Frame2.width != width || Frame2.height != height || Frame2.layers != colors)
	{
		mexErrMsgTxt("ERROR: Sizes of Reference and Search images must be the same."); 
	}

	const int search = in > 2 ? (int)mxGetScalar(input[2]) : 20;
	const int radius = in > 3 ? (int)mxGetScalar(input[3]) : 4;

    mwSize dimsS[] = {height, width, 2};
    mwSize dimsC[] = {height, width, 1};

	//Matlab-allocated variables
	output[0] = mxCreateNumericArray(3, dimsS, mxSINGLE_CLASS, mxREAL);
	output[1] = mxCreateNumericArray(3, dimsS, mxSINGLE_CLASS, mxREAL);
    output[2] = mxCreateNumericArray(3, dimsC, mxSINGLE_CLASS, mxREAL);
	output[3] = mxCreateNumericArray(3, dimsC, mxSINGLE_CLASS, mxREAL);


	mymex::MexImage<float> Shifts12 (output[0]);
	mymex::MexImage<float> Shifts21 (output[1]);
	mymex::MexImage<float> Conf12 (output[2]);
	mymex::MexImage<float> Conf21 (output[3]);


	#pragma omp parallel sections
	{
		#pragma omp section
		{
			findShifts(Frame1, Frame2, Shifts12, Conf12, search, radius);
		}
		#pragma omp section
		{
			findShifts(Frame2, Frame1, Shifts21, Conf21, search, radius);
		}
	}

	#pragma omp parallel sections
	{
		#pragma omp section
		{
			correspondShifts(Shifts12, Shifts21, Conf12, thr);
		}
		#pragma omp section
		{
			correspondShifts(Shifts21, Shifts12, Conf21, thr);
		}
	}
	

}