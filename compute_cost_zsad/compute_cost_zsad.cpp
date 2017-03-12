/** Computes 3D Cost Volume for 2 (rectified) images. 
* Uses Z-SAD dissimilarity metric. 
* 
* @author Sergey Smirnov
* @date 4.10.2011
*/
#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}
#endif // __cplusplus

#include <cmath>
#include <algorithm>
#include <omp.h>
//#include "../common/defines.h"
#include "../common/meximage.h"
//#include "../common/common.h"
typedef unsigned char uint8;

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")
//#pragma warning(disable:4244)
//#pragma warning(disable:4018)

using namespace mymex;

void mexFunction (int nout, mxArray* output[], int in, const mxArray* input[])
{
	// use at least 2 threads, even if it says one core available
	omp_set_num_threads(std::max(2,omp_get_num_threads())); 
	omp_set_dynamic(0);

	if(in < 3 || in > 7 || nout < 1 || nout > 2 || mxGetClassID(input[0])!=mxUINT8_CLASS || mxGetClassID(input[1])!=mxUINT8_CLASS)
	{
		mexErrMsgTxt("USAGE: [CostL, <CostR>] = compute_cost_zsad(UINT8(Left), UINT8(Right), maxdisp, [mindisp = 0, cost_threshold, mean_radius, mix_SAD_ratio]);"); 
    } 

	if(!MexImage<uint8>::isValidType(input[0]) || !MexImage<uint8>::isValidType(input[1]))
	{
		mexErrMsgTxt("ERROR: Only UINT8 type is allowed for Left and Right."); 
	}

	MexImage<uint8> Left(input[0]);
	MexImage<uint8> Right(input[1]);
	const int width = Left.width;
	const int height = Left.height;
	const int layers = Left.layers;	
	const int HW = Left.layer_size;

	if(width != Right.width || height != Right.height || layers != Right.layers)
	{
		mexErrMsgTxt("ERROR: Sizes of Left and  Right must be the same!"); 
	}	

	const int maxdisp = static_cast<int>(mxGetScalar(input[2]));	
	const int mindisp = (in > 3) ?  static_cast<int>(mxGetScalar(input[3])) : 0;	
	const float cost_threshold = (in > 4) ?  static_cast<float>(mxGetScalar(input[4])) : 50;
	const int radius = (in > 5) ?  std::max(static_cast<int>(mxGetScalar(input[5])),1) : 1;
	const float mix_ratio = (in > 6) ?  std::max<float>(0, std::min<float>(1, static_cast<float>(mxGetScalar(input[6])))) : 0;
	const int dispLayers = static_cast<unsigned>(std::max(maxdisp - mindisp +1, 1));
	const size_t depthcost[] = {(unsigned)height, (unsigned)width, (unsigned)dispLayers};

	//Matlab-allocated variables
	output[0] = mxCreateNumericArray(3, depthcost, mxSINGLE_CLASS, mxREAL);
	MexImage<float> *CostL = new MexImage<float>(output[0]);
    MexImage<float> *CostR = NULL;
	if(nout > 1)
    {
        output[1] = mxCreateNumericArray(3, depthcost, mxSINGLE_CLASS, mxREAL);	
		CostR = new MexImage<float>(output[1]);
    }

	const int lr_max = (nout > 1) ? 2 : 1;
	
	//= (nout > 1) ? () : (new MexImage<float>(Right.width, Right.height, dispLayers));

	MexImage<float> LeftI(Left.width, Left.height, Left.layers);
	MexImage<float> RightI(Right.width, Right.height, Right.layers);
	
	
	#pragma omp parallel sections
	{
		#pragma omp section
		{
			#pragma omp parallel for
			for(long i=0; i<HW; i++)
			{
				for(int c=0; i<layers; c++)
				{
					LeftI.data[i + c*HW] = Left[i + c*HW];
				}
			}
			LeftI.IntegralImage(true);
		}
		#pragma omp section
		{
			#pragma omp parallel for
			for(long i=0; i<HW; i++)
			{
				for(int c=0; i<layers; c++)
				{
					RightI.data[i + c*HW] = Right[i + c*HW];
				}
			}
			Right.IntegralImage(true);
		}
	}	
	
	#pragma omp parallel for
	for(int lr=0; lr<lr_max; lr++) 
	{
		int sign = (lr==0) ? -1 : 1; // disparity direction

		MexImage<float> *Cost = (lr==0) ? CostL : CostR;
		MexImage<uint8> &Reference = (lr==0) ? Left : Right;
		MexImage<uint8> &Template = (lr==0) ? Right : Left;

		MexImage<float> &ReferenceI = (lr==0) ? LeftI : RightI;
		MexImage<float> &TemplateI = (lr==0) ? RightI : LeftI;

		float *meanR = new float[layers];
		float *meanT = new float[layers];

		#pragma omp parallel for		
		for(int index=0; index<HW; index++)
		{
			int y = index % height;
			int x = index / height;

			ReferenceI.getIntegralAverage(x, y, radius, meanR);

			for(int d=0; d<dispLayers; d++)
			{
				int disp = d + mindisp;
				int xT = x + sign*disp; 
				int dHW = d*HW;

				if(xT>=0 && xT<width)
				{
					float cost = 0;

					int indexT = TemplateI.Index(xT, y);
					TemplateI.getIntegralAverage(xT, y, radius, meanT);

					for(int c=0; c<layers; c++)
					{
						int cHW = c*HW;
						float r = (float)Reference[index + cHW] - meanR[c];
						float t = (float)Template[indexT + cHW] - meanT[c];
						float zsad = std::abs(r-t);
						float sad = std::abs((float)Reference[index + cHW] - Template[indexT + cHW]);
						cost += zsad*(1-mix_ratio) + mix_ratio*sad;
					}

					cost /= layers;
					//*(CostL->at(index + dHW)) = (cost > cost_threshold) ? cost_threshold : cost;
					Cost->data[index + dHW] = (cost > cost_threshold) ? cost_threshold : cost;
				}
				else
				{
					Cost->data[index + dHW] = cost_threshold;
				}
			}
		}

		
/*
		float *meanL = new float[layers];
		float *meanR = new float[layers];
		float *meanL2 = new float[layers];
		float *meanR2 = new float[layers];


		#pragma omp for
		for(int x=0; x<width; x++)
		{
			for(int y=0; y<height; y++)
			{
				int index = Left.Index(x, y);
				LeftI.getIntegralAverage(x, y, radius, meanL);
				RightI.getIntegralAverage(x, y, radius, meanR);

				//mymex::window wnd = LeftI.get_window(x, y, radius);
				//LeftI.getIntegralAverage(wnd, meanL);
				//RightI.getIntegralAverage(wnd, meanR);


				for(int d=0; d<dispLayers; d++)
				{
					int disp = d + mindisp;
					int xr = x - disp; 
					int xl = x + disp; 
					int dHW = d*HW;

					//#pragma omp parallel sections
					{
						//#pragma omp section
						{
							if(xr>=0 && xr<width)
							{
								float cost = 0;

								int indexR = Left.Index(xr, y);
								
								//mymex::window wndR = LeftI.get_window(xr, y, radius);
								//RightI.getIntegralAverage(wndR, meanR2);
								RightI.getIntegralAverage(xr, y, radius, meanR2);

								for(int c=0; c<layers; c++)
								{
									int cHW = c*HW;
									float l = (float)Left[index + cHW] - meanL[c];
									float r = (float)Right[indexR + cHW] - meanR2[c];
									float zsad = std::abs(l-r);
									float sad = std::abs((float)Left[index + cHW] - Right[indexR + cHW]);
									cost += zsad*(1-mix_ratio) + mix_ratio*sad;
								}

								cost /= layers;
								//*(CostL->at(index + dHW)) = (cost > cost_threshold) ? cost_threshold : cost;
								CostL->data[index + dHW] = (cost > cost_threshold) ? cost_threshold : cost;
							}
							else
							{
								CostL->data[index + dHW] = cost_threshold;
							}
						}
						
						//#pragma omp section
						{

							if(xl>=0 && xl<width)
							{
								float cost = 0;

								int indexL = Left.Index(xl, y);
								//mymex::window wndL = LeftI.get_window(xl, y, radius);
								LeftI.getIntegralAverage(xl, y, radius, meanL2);

								for(int c=0; c<layers; c++)
								{
									int cHW = c*HW;

									float r = (float)Right[index + cHW] - meanR[c];
									float l = (float)Left[indexL + cHW] - meanL2[c];
									
									float zsad = std::abs(l-r);
									float sad = std::abs((float)Right[index + cHW] - Left[indexL + cHW]);
									cost += zsad*(1-mix_ratio) + mix_ratio*sad;
									//cost += std::abs(l-r);
								}

								cost /= layers;
								//*(CostL->at(index + dHW)) = (cost > cost_threshold) ? cost_threshold : cost;
								CostR->data[index + dHW] = (cost > cost_threshold) ? cost_threshold : cost;
							}
							else
							{
								CostR->data[index + dHW] = cost_threshold;
							}

						}
					}//omp parallel sections
				}
			}
		}

		delete[] meanL;
		delete[] meanR;
		delete[] meanL2;
		delete[] meanR2;
		*/

	}

	// delete "wrappers" of cost volumes
	// Matlab-allocated memory untouched
	delete CostL;
	if(CostR)
		delete CostR;

}

 