/** new_aggregation
*	@file new_aggregation.cpp
*	@date 29.08.2014
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

float max4(float a1, float a2, float a3, float a4)
{
	if(a1 > a2)
	{
		if(a1 > a3)
		{
			return a1 > a4 ? a1 : a4;
		}
		else
		{
			return a3 > a4 ? a3 : a4;
		}
	}
	else
	{
		if(a2 > a3)
		{
			return a2 > a4 ? a2 : a4;
		}
		else
		{
			return a3 > a4 ? a3 : a4;
		}
	}

}

class Layer
{
public:
	MexImage<float> * const Image;
	MexImage<float> * const Signal;

	Layer(MexImage<float>* const _Image, MexImage<float>* const _Signal) : Image(_Image), Signal(_Signal), horizontal(NULL), vertical(NULL)
	{
	}
	
	void populate(const unsigned levels)
	{
		const int width = Image->width;
		const int height = Image->height;
		const int colors = Image->layers;
		const int layers = Signal->layers;

		const int width_2 = width%2 ? width/2+1 : width/2;
		const int height_2 = height%2 ? height/2+1 : height/2;

		if(levels > 0)
		{
			#pragma omp parallel sections
			{
				#pragma omp section
				{
					MexImage<float>* const Image_left = new MexImage<float>(width_2, height, colors);
					MexImage<float>* const Signal_left = new MexImage<float>(width_2, height, layers);
					horizontal = new Layer(Image_left, Signal_left);
					//#pragma omp parallel sections
					{
						//#pragma omp section
						{
							downsample_horizontal (Image, Image_left);
						}
						//#pragma omp section
						{
							downsample_horizontal (Signal, Signal_left);
						}
					}					
				}
				#pragma omp section
				{
					MexImage<float>* const Image_right = new MexImage<float>(width, height_2, colors);
					MexImage<float>* const Signal_right = new MexImage<float>(width, height_2, layers);
					vertical = new Layer(Image_right, Signal_right);
					//#pragma omp parallel sections
					{
						//#pragma omp section
						{
							downsample_vertical (Image, Image_right);
						}
						//#pragma omp section
						{
							downsample_vertical (Signal, Signal_right);
						}
					}
				}
			}
			if(width > 4)
			{
				horizontal->populate(levels-1);
			}						

			if(height > 4)
			{
				vertical->populate(levels-1);
			}
		}	
	}

	void process(const unsigned levels, const float sigma_c, const float sigma_d)
	{
		const int width = Image->width;
		const int height = Image->height;
		const long HW = Image->layer_size;
		const int colors = Image->layers;
		const int layers = Signal->layers;

		const int width_2 = width%2 ? width/2+1 : width/2;
		const int height_2 = height%2 ? height/2+1 : height/2;


		if(levels>0 && horizontal != NULL && vertical != NULL)
		{			
			//mexPrintf("process(%d) : w=%d, h=%d \n", levels, width, height);
			//mexPrintf("process(%d) : go_left() \n", levels);
			horizontal->process(levels-1, sigma_c, sigma_d);			
			//mexPrintf("process(%d) : go_right() \n", levels);
			vertical->process(levels-1, sigma_c, sigma_d);			
			//mexPrintf("process(%d) : done() \n", levels);
			MexImage<float> &ImageC = *Image;
			MexImage<float> &ImageH = *(horizontal->Image);
			MexImage<float> &ImageV = *(vertical->Image);

			MexImage<float> &SignalC = *Signal;
			MexImage<float> &SignalH = *(horizontal->Signal);
			MexImage<float> &SignalV = *(vertical->Signal);

			#pragma omp parallel for
			for(long i=0; i<HW; i++)
			{
				const int x = i / height;
				const int y = i % height;

				float diff_left = 0;
				float diff_top = 0;
				
				for(int c=0; c<colors; c++)
				{
					diff_left += abs(ImageC(x,y,c) - ImageH(x/2,y,c));					
					diff_top += abs(ImageC(x,y,c) - ImageV(x,y/2,c));
				}
				const float weight_left = exp(-diff_left/(colors*sigma_c));
				const float weight_top = exp(-diff_top/(colors*sigma_c));

				float diff_right = 0;
				float diff_bottom = 0;

				float weight_right = 0.f;
				if(x%2 && x/2 < width_2-1)
				{
					for(int c=0; c<colors; c++)
					{
						diff_right += abs(ImageC(x,y,c) - ImageH(x/2+1,y,c));
					}
					weight_right = exp(-diff_right/(colors*sigma_c));
				}		
				
				float weight_bottom = 0.f;
				if(y%2 && y/2 < height_2-1)
				{
					for(int c=0; c<colors; c++)
					{
						diff_bottom += abs(ImageC(x,y,c) - ImageV(x,y/2+1,c));
					}
					weight_bottom = exp(-diff_bottom/(colors*sigma_c));
				}

				
				const float weight_summ = weight_left + weight_right + weight_top + weight_bottom;
				const float weight_central = sigma_d*weight_summ;
				//float weight_central = 1-max4(weight_left, weight_right, weight_top, weight_bottom);
				//weight_central = std::min(weight_central, weight_summ/10000.f);
				//weight_summ += weight_central;
				
				
				for(int l=0; l<layers; l++)
				{
					SignalC(x,y,l) *= weight_central;
					SignalC(x,y,l) += SignalH(x/2, y, l) * weight_left;
					SignalC(x,y,l) += SignalV(x, y/2, l) * weight_top;
					if(x%2 && x/2 < width_2-1)
					{						
						SignalC(x,y,l) += SignalH(x/2+1, y, l) * weight_right;
					}
					
					if(y%2 && y/2 < height_2-1)
					{						
						SignalC(x,y,l) += SignalV(x, y/2+1, l) * weight_bottom;
					}					
										
					SignalC(x,y,l) /= (1+sigma_d)*weight_summ;
				}
			}								
		}
		else
		{
			return;	
		}		
	}

	~Layer()
	{
		if(horizontal != NULL)
		{
			delete horizontal->Image;
			delete horizontal->Signal;
			delete horizontal;
			horizontal = NULL;
		}

		if(vertical != NULL)
		{
			delete vertical->Image;
			delete vertical->Signal;
			delete vertical;
			vertical = NULL;
		}
	}

private:

	//void downsample_horizontal(MexImage<float> * const from, MexImage<float> * const to)
	//{		
	//	MexImage<float> &ImageFrom = *from;
	//	MexImage<float> &ImageTo = *to;
	//	ImageTo.setval(0.f);

	//	const int width = ImageFrom.width;
	//	const int height = ImageFrom.height;
	//	const int colors = ImageFrom.layers;				
	//	const int width_2 = ImageTo.width;
	//	for(int y=0; y<height; y++)
	//	{
	//		for(int c=0; c<colors; c++)
	//		{
	//			float buff = ImageFrom(0, y, c) * 3;
	//			
	//			for(int x=0; x<width; x++)
	//			{
	//				buff += ImageFrom(std::min(x+1, width-1), y, c);
	//				buff -= ImageFrom(std::max(x-1, 0), y, c);					
	//				ImageTo(x/2, y, c) += (x%2 == 0) ? buff/3 : 0;										
	//			}
	//		}						
	//	}
	//}

	void downsample_horizontal(MexImage<float> * const from, MexImage<float> * const to)
	{		
		MexImage<float> &ImageFrom = *from;
		MexImage<float> &ImageTo = *to;
		ImageTo.setval(0.f);

		const int width = ImageFrom.width;
		const int height = ImageFrom.height;
		const int colors = ImageFrom.layers;				
		const int width_2 = ImageTo.width;
		
		#pragma omp parallel for
		for(int x_2=0; x_2<width_2; x_2++)
		{
			for(int y=0; y<height; y++)
			{
				for(int xx=std::max(0,x_2*2-1); xx<=std::min(x_2*2+1,width-1); xx++)
				{
					const int norm = x_2==0 || x_2==width_2-1 && width%2==0 ? 2 : 3; 
					for(int c=0; c<colors; c++)
					{
						ImageTo(x_2,y,c) += ImageFrom(xx,y,c)/norm;
					}
				}
			}
		}		
	}

	void downsample_vertical(MexImage<float> * const from, MexImage<float> * const to)	
	{
		MexImage<float> &ImageFrom = *from;
		MexImage<float> &ImageTo = *to;
		ImageTo.setval(0.f);

		const int width = ImageFrom.width;
		const int height = ImageFrom.height;
		const int colors = ImageFrom.layers;	
		const int height_2 = ImageTo.height;
		#pragma omp parallel for
		for(int x=0; x<width; x++)
		{
			for(int y_2=0; y_2<height_2; y_2++)
			{
				for(int yy=std::max(0,y_2*2-1); yy<=std::min(y_2*2+1,height-1); yy++)
				{
					const int norm = y_2==0 || y_2==height_2-1 && height%2==0 ? 2 : 3; 
					for(int c=0; c<colors; c++)
					{
						ImageTo(x,y_2,c) += ImageFrom(x,yy,c)/norm;
					}
				}
			}
		}
	}


	Layer *horizontal;	
	Layer *vertical;

	Layer();
};

void mexFunction (int nout, mxArray* output[], int in, const mxArray* input[])
{
#ifndef _DEBUG
	omp_set_num_threads(std::max<int>(4, omp_get_max_threads()));
	omp_set_dynamic(std::max<int>(2, omp_get_max_threads()/2));
#endif
	
	if(in != 5 || nout != 1 || mxGetClassID(input[0]) != mxSINGLE_CLASS || mxGetClassID(input[1]) != mxSINGLE_CLASS)
	{
		mexErrMsgTxt("USAGE: [Aggregated] = new_aggregation(single(Signal), single(Image), layers, sigma_c, sigma_d);");
    }

	MexImage<float> Signal(input[0]);	
	MexImage<float> Image(input[1]);
	const unsigned levels = (unsigned)mxGetScalar(input[2]);		
	const float sigma_c = (float)mxGetScalar(input[3]);		
	const float sigma_d = (float)mxGetScalar(input[4]);		

	const int height = Signal.height;
	const int width = Signal.width;
	const long HW = Signal.layer_size;
	const int layers = Signal.layers;	
	const int colors = Image.layers;
	if(Image.width != width || Image.height != height)
	{
		mexErrMsgTxt("Resolution of Signal and Image must coincide!");
	}
		
	const float nan = sqrt(-1.f);	
	
	const size_t dims[] = {height, width, layers};		

	output[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL); 	
	
	MexImage<float> Aggregated(output[0]);
	for(long i=0; i<HW*layers; i++)
	{
		Aggregated[i] = Signal[i];
	}
	
	Layer *root = new Layer(&Image, &Aggregated);	
	root->populate(levels);
	root->process(levels, sigma_c, sigma_d);
	delete root;
}
