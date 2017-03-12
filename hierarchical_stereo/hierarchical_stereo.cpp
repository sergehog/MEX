/** hierarchical_stereo
*	@file hierarchical_stereo.cpp
*	@date 20.01.2014
*	@author Sergey Smirnov <sergey.smirnov@tut.fi>
*/

#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}
#endif // __cplusplus

#include "../common/meximage.h"
//#include "../common/defines.h"
#include <vector>
#include <cmath>
#include <algorithm>
#ifndef _DEBUG
#include <omp.h>
#endif

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")

typedef unsigned char uint8;
using namespace mymex;

void decimate_level(MexImage<unsigned char> &fineImage, MexImage<unsigned char> &fineSignal, MexImage<unsigned char> &coarseImage, MexImage<unsigned char> &coarseSignal, const int layer, const float sigma)
{
	const int signal_layers = fineSignal.layers;
	const int width = fineImage.width;
	const int height = fineImage.height;
	const int colors = fineImage.layers;
	const long HW = fineImage.layer_size;	

	const int fineWidth = fineImage.width; 
	const int fineHeight = fineImage.height;

	const int coarseWidth = coarseImage.width; 
	const int coarseHeight = coarseImage.height;

	if(layer % 2)
	{
		// horisontal decimation
		#pragma omp parallel for
		for(int y=0; y<fineHeight; y++)
		{				
			for(int fx=0; fx<fineWidth; fx+=2)
			{											
				const int cx = fx/2;					
					
				for(int c=0; c<colors; c++)
				{
					coarseImage(cx,y, c) = fineImage(fx,y,c);
				}

				for(int l=0; l<signal_layers; l++)
				{
					coarseSignal(cx,y,l) = fineSignal(fx,y,l);
				}

				// weight from left
				float weight1 = 0;
				if(fx>0)
				{						
					for(int c=0; c<colors; c++)
					{
						weight1 += abs(fineImage(fx,y,c) - fineImage(fx-1,y,c));
					}
					weight1 = exp(-weight1/colors);
				}
					
				// weight from right
				float weight2 = 0;
				if(fx<fineWidth-1)
				{						
					for(int c=0; c<colors; c++)
					{
						weight2 += abs(fineImage(fx,y,c) - fineImage(fx+1,y,c));
					}
					weight2 = exp(-weight2/colors);
				}
					
				//update decimated color image
				for(int c=0; c<colors; c++)
				{
					if(fx>0)
					{
						coarseImage(cx,y, c) += fineImage(fx-1,y,c)*weight1;
					}
					if(fx<fineWidth-1)
					{
						coarseImage(cx,y, c) += fineImage(fx+1,y,c)*weight2;							
					}
					coarseImage(cx,y,c) /= (1 + weight1 + weight2);						
				}
					
				//update decimated signal
				for(int l=0; l<signal_layers; l++)
				{
					if(fx>0)
					{
						coarseSignal(cx,y,l) += fineSignal(fx-1,y,l)*weight1;
					}
					if(fx<fineWidth-1)
					{
						coarseSignal(cx,y,l) += fineSignal(fx+1,y,l)*weight2;

					}
					coarseSignal(cx,y,l) /= (1 + weight1 + weight2);
				}					
			}
		}
	}
	else
	{
		// vertical decimation
		#pragma omp parallel for
		for(int x=0; x<fineWidth; x++)			
		{				
			for(int fy=0; fy<fineHeight; fy+=2)
			{											
				const int cy = fy/2;					
					
				for(int c=0; c<colors; c++)
				{
					coarseImage(x,cy, c) = fineImage(x,fy,c);
				}

				for(int l=0; l<signal_layers; l++)
				{
					coarseSignal(x,cy,l) = fineSignal(x,fy,l);
				}
				// weight from upper pixel
				float weight1 = 0;
				if(fy>0)
				{						
					for(int c=0; c<colors; c++)
					{
						weight1 += abs(fineImage(x,fy,c) - fineImage(x,fy-1,c));
					}
					weight1 = exp(-weight1/colors);
				}

				// weight from lower pixel
				float weight2 = 0;
				if(fy<fineHeight-1)
				{						
					for(int c=0; c<colors; c++)
					{
						weight2 += abs(fineImage(x,fy,c) - fineImage(x,fy+1,c));
					}
					weight2 = exp(-weight2/colors);
				}
					
				//update decimated color image
				for(int c=0; c<colors; c++)
				{
					if(fy>0)
					{
						coarseImage(x,cy, c) += fineImage(x,fy-1,c)*weight1;
					}
					if(fy<fineHeight-1)
					{
						coarseImage(x,cy, c) += fineImage(x,fy+1,c)*weight2;							
					}
					coarseImage(x,cy,c) /= (1 + weight1 + weight2);						
				}

				//update decimated signal
				for(int l=0; l<signal_layers; l++)
				{
					if(fy>0)
					{
						coarseSignal(x,cy,l) += fineSignal(x,fy-1,l)*weight1;
					}
					if(fy<fineHeight-1)
					{
						coarseSignal(x,cy,l) += fineSignal(x,fy+1,l)*weight2;

					}
					coarseSignal(x,cy,l) /= (1 + weight1 + weight2);
				}					
			}
		}
	}		
}



void estimate_disparity(MexImage<unsigned char> &Cost, MexImage<float> &Disparity, MexImage<unsigned char> &Confidence)
{
	const int cost_layers = Cost.layers;
	const int width = Cost.width;
	const int height = Cost.height;	
	const long HW = Cost.layer_size;	

	// temporal buffers	
	MexImage<unsigned char> MinCost1(width, height, 1);
	MexImage<unsigned char> MinCost2(width, height, 1);
	MexImage<unsigned char> MinCost3(width, height, 1);
	MexImage<unsigned char> Disparity2(width, height, 1);

	MinCost1.setval(255);
	MinCost2.setval(255);
	MinCost3.setval(255);


	for(int d=0; d<cost_layers; d++)
	{
		#pragma omp parallel for
		for(long i=0; i<HW; i++)
		{
			const int y = i % height;
			const int x = i / height;

			if(Cost(x,y,d) < MinCost1(x,y))
			{
				Disparity2(x,y) = Disparity(x,y);
				Disparity(x,y) = d;
				MinCost3(x,y) = MinCost2(x,y);
				MinCost2(x,y) = MinCost1(x,y);
				MinCost1(x,y) = Cost(x,y,d);
			}
			else if(Cost(x,y,d) < MinCost2(x,y))
			{
				Disparity2(x,y) = d;
				MinCost3(x,y) = MinCost2(x,y);
				MinCost2(x,y) = Cost(x,y,d);
			}
			else if(Cost(x,y,d) < MinCost3(x,y))
			{
				MinCost3(x,y) = Cost(x,y,d);
			}
		}
	}

	#pragma omp parallel for
	for(long i=0; i<HW; i++)
	{
		const int y = i % height;
		const int x = i / height;

		if(abs(Disparity(x,y)-Disparity2(x,y)) == 1)
		{
			const int d = Disparity(x,y);
			const float c = MinCost1(x,y);
			const float cu = d > 0 ? Cost(x,y,d-1) : 255;
			const float cl = d < cost_layers-1 ? Cost(x,y,d+1) : 255;

			Disparity(x,y) = float(d) - (cu-cl)/(2*(cu+cl-2*c));
			Confidence(x,y) = round(255 * (MinCost3(x,y)-MinCost1(x,y))/MinCost3(x,y));			
		}
		else
		{
			Confidence(x,y) = round(255 * (MinCost2(x,y)-MinCost1(x,y))/MinCost2(x,y));
		}
	}
	
}

void estimate_confidence(MexImage<unsigned char> &CostL, MexImage<unsigned char> &CostR, MexImage<unsigned char> &ConfL, MexImage<unsigned char> &ConfR, const int mindisp, const int maxdisp, const int layer)
{
		//MexImage<float> fineDispL(fineWidth, fineHeight); 
		//MexImage<unsigned char> fineConfL(fineWidth, fineHeight); 
		//estimate_disparity(*fineCostL, fineDispL, fineConfL);

		//MexImage<float> fineDispR(fineWidth, fineHeight); 
		//MexImage<unsigned char> fineConfR(fineWidth, fineHeight); 
		//estimate_disparity(*fineCostR, fineDispR, fineConfR);

		//MexImage<float> coarseDispL(coarseWidth, coarseHeight); 
		//MexImage<unsigned char> coarseConfL(coarseWidth, coarseHeight); 
		//estimate_disparity(*coarseCostL, coarseDispL, coarseConfL);

		//MexImage<float> coarseDispR(coarseWidth, coarseHeight); 
		//MexImage<unsigned char> coarseConfR(coarseWidth, coarseHeight); 
		//estimate_disparity(*coarseCostR, coarseDispR, coarseConfR);

		//if(layer % 2) // vertical upsampling ? 
		//{
		//	#pragma omp parallel for
		//	for(int x=0; x<fineWidth; x++)			
		//	{				
		//		for(int fy=0; fy<fineHeight; fy++)
		//		{																
		//			const int cy = fy/2;

		//			//if(fy%2)
		//			//{
		//			//	float weight = 0; // weight between fine color value and left coarse image value
		//			//	for(int c=0; c<colors; c++)
		//			//	{
		//			//		weight += abs((*fineImage)(x,fy,c) - (*coarseImage)(x,cy,c));
		//			//	}
		//			//	weight = exp(-weight/colors);

		//			//	for(int l=0; l<signal_layers; l++)
		//			//	{
		//			//		(*fineSignal)(x,fy,l) *= (1-weight);
		//			//		(*fineSignal)(x,fy,l) += (*coarseSignal)(x,cy,l)*weight; // just copy corresponding value
		//			//	}
		//			//}
		//			//else
		//			//{
		//			//	float weight1 = 0; // weight between fine color value and left coarse image value
		//			//	for(int c=0; c<colors; c++)
		//			//	{
		//			//		weight1 += abs((*fineImage)(x,fy,c) - (*coarseImage)(x,cy,c));
		//			//	}
		//			//	weight1 = exp(-weight1/colors);

		//			//	float weight2 = 0; // weight between fine color value and right coarse image value
		//			//	if(fy<fineHeight-1)
		//			//	{
		//			//		for(int c=0; c<colors; c++)
		//			//		{
		//			//			weight2 += abs((*fineImage)(x,fy,c) - (*coarseImage)(x,cy+1,c));
		//			//		}
		//			//		weight2 = exp(-weight2/colors);
		//			//	}

		//			//	for(int l=0; l<signal_layers; l++)
		//			//	{
		//			//		(*fineSignal)(x,fy,l) += (*coarseSignal)(x,cy,l) * weight1; 
		//			//		if(fy<fineHeight-1)
		//			//		{
		//			//			(*fineSignal)(x,fy,l) += (*coarseSignal)(x,cy+1,l) * weight2; 
		//			//		}
		//			//		(*fineSignal)(x,fy,l) /= 1 + weight1 + weight2;
		//			//	}						
		//			//}
		//		}
		//	}



		//}
		//else // horisontal upsampling ?  
		//{
		//	#pragma omp parallel for
		//	for(int y=0; y<fineHeight; y++)
		//	{				
		//		for(int fx=0; fx<fineWidth; fx++)
		//		{																
		//			const int cx = fx/2;

		//			if(fx%2)
		//			{
		//				for(int l=0; l<signal_layers; l++)
		//				{
		//					(*fineSignal)(fx,y,l) = (*coarseSignal)(cx,y,l); // just copy corresponding value
		//				}
		//			}
		//			else
		//			{
		//				float weight1 = 0; // weight between fine color value and left coarse image value
		//				for(int c=0; c<colors; c++)
		//				{
		//					weight1 += abs((*fineImage)(fx,y,c) - (*coarseImage)(cx,y,c));
		//				}
		//				weight1 = exp(-weight1/colors);

		//				float weight2 = 0; // weight between fine color value and right coarse image value
		//				if(fx<fineWidth-1)
		//				{
		//					for(int c=0; c<colors; c++)
		//					{
		//						weight2 += abs((*fineImage)(fx,y,c) - (*coarseImage)(cx+1,y,c));
		//					}
		//					weight2 = exp(-weight2/colors);
		//				}

		//				for(int l=0; l<signal_layers; l++)
		//				{
		//					(*fineSignal)(fx,y,l) += (*coarseSignal)(cx,y,l) * weight1; 
		//					if(fx<fineWidth-1)
		//					{
		//						(*fineSignal)(fx,y,l) += (*coarseSignal)(cx+1,y,l) * weight2; 
		//					}
		//					(*fineSignal)(fx,y,l) /= 1 + weight1 + weight2;
		//				}						
		//			}
		//		}
		//	}

		//}
}

void upsample_level(MexImage<unsigned char> &fineCost, MexImage<unsigned char> &fineImage, MexImage<unsigned char> &fineConf, MexImage<unsigned char> &coarseCost, MexImage<unsigned char> &coarseImage, MexImage<unsigned char> &coarseConf, const int layer)
{

}


typedef uchar unsigned char;

struct pyramid_layer
{
private:
	bool full; 

	MexImage<uchar> * const ImageL;
	MexImage<uchar> * const ImageR;

	MexImage<uchar> * const CostL;
	MexImage<uchar> * const CostR;

	MexImage<uchar> * const ConfL;
	MexImage<uchar> * const ConfR;

	void calculate_cost(MexImage<unsigned char> &Reference, MexImage<unsigned char> &Target, MexImage<unsigned char> &CostRef, const int mindisp, const int maxdisp, const bool lefttoright)
	{
		const int cost_layers = CostRef.layers;
		const int width = Reference.width;
		const int height = Reference.height;
		const int colors = Reference.layers;
		const long HW = Reference.layer_size;	

		#pragma omp parallel for
		for(int d=0; d<cost_layers; d ++)
		{
			const int disparity = lefttoright ? (mindisp + d) : -(mindisp + d);
			#pragma omp parallel for
			for(long i=0; i<HW; i++)
			{
				const int y = i % height;
				const int x = i / height;
				unsigned diff = 0;
				for(int c=0; c<colors; c++)
				{
					diff += abs(Reference(x,y,c) - Target(x-disparity, y, c));
				}

				CostRef(x,y,d) = diff > 255u ? 255u : (unsigned char)diff;
			}
		}
	}

public:
	pyramid_layer();

	void init(const int width, const int height, const int colors, const int cost_layers) : 
		full(true),
		ImageL(new MexImage<uchar>(width, height, colors)),
		ImageR(new MexImage<uchar>(width, height, colors)),
		CostL(new  MexImage<uchar>(width, height, cost_layers)),
		CostR(new  MexImage<uchar>(width, height, cost_layers)),
		ConfL(new  MexImage<uchar>(width, height)),
		ConfR(new  MexImage<uchar>(width, height))
	{

	}

	void init(MexImage<uchar> &imageL, MexImage<uchar> &imageR, const int mindisp, const int maxdisp) :
		full(false),
	{
		ImageL = &imageL;
		ImageR = &imageR;

		CostL = new MexImage<uchar>(width, height, cost_layers);
		CostR = new MexImage<uchar>(width, height, cost_layers);

		#pragma omp parallel sections
		{
			#pragma omp section
			{
				calculate_cost(*ImageL, *ImageR, *CostL, mindisp, maxdisp, true);
			}
			#pragma omp section
			{
				calculate_cost(*ImageR, *ImageL, *CostR, mindisp, maxdisp, false);
			}
		}

		ConfL = new MexImage<uchar>(width, height);
		ConfR = new MexImage<uchar>(width, height);
	}


	~pyramid_layer()
	{
		if(full)
		{
			delete ImageL, ImageR, CostL, CostR;
		}
		delete ConfL, ConfR;
	}


};


void mexFunction (int nout, mxArray* output[], int in, const mxArray* input[])
{
#ifndef _DEBUG
	omp_set_num_threads(std::max<int>(4, omp_get_max_threads()));
	omp_set_dynamic(std::max<int>(2, omp_get_max_threads()/2));
#endif
	
	if(in < 3 || in > 4 || nout != 4 || mxGetClassID(input[0]) != mxSINGLE_CLASS || mxGetClassID(input[1]) != mxSINGLE_CLASS)
	{
		mexErrMsgTxt("USAGE: [DispL, DispR, CosfL, ConfR] = hierarchical_stereo(uint8(L), uint8(R), mindisp, maxdisp, layers <, sigma>);");
    }

	MexImage<unsigned char> Left(input[0]);	
	MexImage<unsigned char> Right(input[1]);
	const int mindisp = (int)mxGetScalar(input[2]);		
	const int maxdisp = (int)mxGetScalar(input[3]);		

	const int height = Left.height;
	const int width = Left.width;
	const int colors = Left.layers;
	const int cost_layers = maxdisp - mindisp + 1;
	const long HW = Left.layer_size;
	const float nan = sqrt(-1.f);
	const int r = 1;

	if(Right.height != height || Right.width != width)
	{
		mexErrMsgTxt("Image and Signal must have the same width and height.");
	}
			
	const int layers = std::max(1, (int)mxGetScalar(input[4]));		
	const float sigma = (in > 3) ? (float)mxGetScalar(input[5]) : 10.;
	
	const size_t dims[] = {(size_t)height, (size_t)width, 1};		

	output[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL); 	
	output[1] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL); 	
	output[2] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL); 	
	output[3] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL); 	
	
	MexImage<float> DispL(output[0]);
	MexImage<float> DispR(output[1]);
	MexImage<float> ConfL(output[2]);
	MexImage<float> ConfR(output[3]);
	

	MexImage<unsigned char> CostL(width, height, cost_layers);
	MexImage<unsigned char> CostR(width, height, cost_layers);

	

	pyramid_layer * pyramid = new pyramid_layer[layers+1];
	pyramid[0].init(Left, Right, CostL, CostR);


	// pyramid of signals and images
	MexImage<unsigned char> ** CostsL = new MexImage<unsigned char>*[layers+1];
	MexImage<unsigned char> ** ImagesL = new MexImage<unsigned char>*[layers+1];
	CostsL[0] = &CostL;
	ImagesL[0] = &Left;	

	// pyramid of signals and images
	MexImage<unsigned char> ** CostsR = new MexImage<unsigned char>*[layers+1];
	MexImage<unsigned char> ** ImagesR = new MexImage<unsigned char>*[layers+1];
	CostsR[0] = &CostR;
	ImagesR[0] = &Right;		

	MexImage<unsigned char> ** ConfsL = new MexImage<unsigned char>*[layers+1];
	MexImage<unsigned char> ** ConfsR = new MexImage<unsigned char>*[layers+1];

	// first pass from finer to coarse levels
	for(int layer=1; layer <= layers; layer++)
	{
		MexImage<unsigned char> &fineImageL = *ImagesL[layer-1];
		MexImage<unsigned char> &fineCostL = *CostsL[layer-1];

		MexImage<unsigned char> &fineImageR = *ImagesR[layer-1];
		MexImage<unsigned char> &fineCostR = *CostsR[layer-1];

		const int fineWidth = fineImageL.width; 
		const int fineHeight = fineImageL.height;

		const int coarseWidth = (layer % 2) ? fineWidth/2 + fineWidth%2 : fineWidth;
		const int coarseHeight = (layer % 2) ? fineHeight : fineHeight/2 + fineHeight%2;
		

		#pragma omp parallel sections
		{
			#pragma omp section
			{
				CostsL[layer] = new MexImage<unsigned char>(coarseWidth, coarseHeight, colors);
				ImagesL[layer] = new MexImage<unsigned char>(coarseWidth, coarseHeight, cost_layers);
				MexImage<unsigned char> &coarseImageL = *CostsL[layer];
				MexImage<unsigned char> &coarseCostL = *ImagesL[layer];
				
				decimate_level(fineImageL, fineCostL, coarseImageL, coarseCostL, layer, sigma);
			}
			#pragma omp section
			{				
				CostsR[layer] = new MexImage<unsigned char>(coarseWidth, coarseHeight, colors);				
				ImagesR[layer] = new MexImage<unsigned char>(coarseWidth, coarseHeight, cost_layers);
				MexImage<unsigned char> &coarseImageR = *CostsR[layer];
				MexImage<unsigned char> &coarseCostR = *ImagesR[layer];
				
				decimate_level(fineImageR, fineCostR, coarseImageR, coarseCostR, layer, sigma);				
			}
		}		
	}

	// aliases to cost volumes at coarser level
	MexImage<unsigned char> &CostLs = *CostsL[layers];
	MexImage<unsigned char> &CostRs = *CostsR[layers];

	// create confidence maps for coarser levels
	ConfsL[layers] = new MexImage<unsigned char>(CostLs.width, CostLs.height); 
	ConfsR[layers] = new MexImage<unsigned char>(CostLs.width, CostLs.height); 
	
	// aliases for them
	MexImage<unsigned char> &ConfLs = *ConfsL[layers];
	MexImage<unsigned char> &ConfRs = *ConfsR[layers];

	// now compute!
	estimate_confidence(CostLs, CostRs, ConfLs, ConfRs, mindisp, maxdisp, layers);
	
	
	// second pass from coarse to finer levels
	for(int layer=layers-1; layer >= 0; layer--)
	{
		MexImage<unsigned char> &fineImageL = *ImagesL[layer];
		MexImage<unsigned char> &fineCostL = *CostsL[layer];

		MexImage<unsigned char> &coarseImageL = *ImagesL[layer+1];
		MexImage<unsigned char> &coarseCostL = *CostsL[layer+1];

		MexImage<unsigned char> &fineImageR = *ImagesR[layer];
		MexImage<unsigned char> &fineCostR = *CostsR[layer];

		MexImage<unsigned char> &coarseImageR = *ImagesR[layer+1];
		MexImage<unsigned char> &coarseCostR = *CostsR[layer+1];

		int fineWidth = fineImageL.width; 
		int fineHeight = fineImageL.height;

		ConfsL[layer] = new MexImage<unsigned char>(fineWidth, fineHeight); 
		ConfsR[layer] = new MexImage<unsigned char>(fineWidth, fineHeight); 

		MexImage<unsigned char> &fineConfL = *ConfsL[layer];
		MexImage<unsigned char> &fineConfR = *ConfsR[layer];

		MexImage<unsigned char> &coarseConfL = *ConfsL[layer+1];
		MexImage<unsigned char> &coarseConfR = *ConfsR[layer+1];

		estimate_confidence(fineCostL, fineCostR, fineConfL, fineConfR, mindisp, maxdisp, layer);


		int coarseWidth = coarseImageR.width;
		int coarseHeight = coarseImageR.height;

		
		upsample_level(fineCostL, fineImageL, fineConfL, coarseCostL, coarseImageL, coarseConfL, layer);
		upsample_level(fineCostR, fineImageR, fineConfR, coarseCostR, coarseImageR, coarseConfR, layer);		
	}
	
	#pragma omp parallel for
	for(int layer=1; layer <= layers; layer++)
	{
		delete ImagesL[layer];
		delete CostsL[layer];
		delete ConfsL[layer];

		delete ImagesR[layer];
		delete CostsR[layer];
		delete ConfsR[layer];
	}

	delete[] ImagesL;
	delete[] CostsL;
	delete[] ConfsL;

	delete[] ImagesR;
	delete[] CostsR;
	delete[] ConfsR;	
}
