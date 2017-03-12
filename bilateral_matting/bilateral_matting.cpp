/** bilateral_matting
*	@file bilateral_matting.cpp
*	@date 5.11.2012
*	@author Sergey Smirnov <sergey.smirnov@tut.fi>
*/

#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}
#endif // __cplusplus

#include "../common/meximage.h"
//#include "../common/defines.h"
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <memory>

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")

//typedef unsigned char uint8;
using namespace mymex;

bool isknown(float value, const float trimax)
{
	if(value == 0.f || value == trimax)
		return true;
	return false;
}

void mexFunction (int nout, mxArray* output[], int in, const mxArray* input[])
{
	omp_set_num_threads(std::max(2, omp_get_max_threads()));
	omp_set_dynamic(0);
	
	if(in < 2 || in > 7 || nout < 1 || nout > 3 || mxGetClassID(input[0])!=mxSINGLE_CLASS)
	{
		mexErrMsgTxt("USAGE: [Alpha, [Bg, Fg]] = bilateral_matting(single(Image), single(Trimap), <radius, sigma_color, distance_threshold, significant_colors>);");
    }

	MexImage<float> Image(input[0]);	
	MexImage<float> Trimap(input[1]);
	
	const int height = Image.height;
	const int width = Image.width;	
	const long HW = Image.layer_size;
	const unsigned colors = Image.layers;
	const float nan = sqrt(-1.f);

	float trimax = 0.f;
	for(long i=0; i<HW; i++)
	{
		trimax = Trimap[i] > trimax ? Trimap[i] : trimax;
	}

	if(Trimap.height != height || Trimap.width != width)
	{
		mexErrMsgTxt("Trimap and Image must have the same width and height.");
	}
	
	const int radius = (in > 2) ? std::max(1, (int)mxGetScalar(input[2])) : 1;	
	const int diameter = radius*2+1;
	const int window = diameter*diameter;
	const int pixels_threshold = window/4.f;
	//const float directionality = 100.f;
	

	const float sigma_color = (in > 3) ? std::max(0.1f, (float)mxGetScalar(input[3])) : 10.f;
	const float distance_threshold = (in > 4) ? std::max(0.1f, (float)mxGetScalar(input[4])) : 255.f;	
	const unsigned significant = (in > 5) ? std::max<unsigned>(1, mxGetScalar(input[5])) : colors;	

	const size_t dims[] = {(size_t)height, (size_t)width, 1};	
	const size_t dims3[] = {(size_t)height, (size_t)width, colors};	

	output[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL); 	
	MexImage<float> Alpha(output[0]);
	
	if(nout > 1)
	{
		output[1] = mxCreateNumericArray(3, dims3, mxSINGLE_CLASS, mxREAL); 			
	}
	if(nout > 2)
	{
		output[2] = mxCreateNumericArray(3, dims3, mxSINGLE_CLASS, mxREAL); 	
	}	

	std::auto_ptr<MexImage<float>> Bg ((nout > 1) ? new MexImage<float>(output[1]) : new MexImage<float>(width, height, colors));	
	std::auto_ptr<MexImage<float>> Fg ((nout > 2) ? new MexImage<float>(output[2]) : new MexImage<float>(width, height, colors));	
	std::auto_ptr<MexImage<float>> AlphaTmp (new MexImage<float>(width, height, 1));

	#pragma omp parallel for
	for(long i=0; i<HW; i++)
	{
		if(isknown(Trimap[i], trimax))
		{
			if(Trimap[i] < 0.000001f)
			{
				Alpha.data[i] = 0.f;
				AlphaTmp->data[i] = 0.f;

				for(int c=0; c<colors; c++)
				{
					Bg->data[i+c*HW] = Image[i+c*HW];
					Fg->data[i+c*HW] = nan;
				}
			}
			else
			{
				Alpha.data[i] = 1.f;
				AlphaTmp->data[i] = 1.f;
				for(int c=0; c<colors; c++)
				{
					Bg->data[i+c*HW] = nan;
					Fg->data[i+c*HW] = Image[i+c*HW];
				}
			}
		}
		else
		{
			AlphaTmp->data[i] = nan;
			Alpha.data[i] = nan;
			for(int c=0; c<colors; c++)
			{
				Bg->data[i+c*HW] = nan;
				Fg->data[i+c*HW] = nan;
			}
		}
	}
	
	mexPrintf("width=%d\n", width);
	mexPrintf("height=%d\n", height);
	mexPrintf("colors=%d\n", colors);	
	mexPrintf("radius=%d\n", radius);	
	mexPrintf("sigma_color=%f\n", sigma_color);
	mexPrintf("trimax=%f\n", trimax);
	
	long iter = 0;
	long maxiter = 1000;//Alpha.layer_size;
	long updated = 1;
	
	// work iteratively until all uncertaities are resolved
	while(Alpha.hasNans() && iter < maxiter && updated > 0)
	{	
		//int maxPixels = 0;
		//long bestIndex = -1;
			
		// seek for best unknown position
		#pragma omp parallel
		{
			float *colorsB = new float[colors];
			float *colorsF = new float[colors];

			#pragma omp for
			for(long i=0; i<HW; i++)
			{
				if(!_isnan(Alpha[i]))
				{				
					continue;
				}
				
				int x = i / height;
				int y = i % height;	
				int pixels = 0;
				int pixelsB = 0;
				int pixelsF = 0;
				double weights = 0.f;
				double weightsB = 0.f;
				double weightsF = 0.f;
				double value = 0.f; 
				for(int c=0; c<colors; c++)
				{
					colorsB[c] = 0.f;
					colorsF[c] = 0.f;
				}

				for(int xd=std::max(0,x-radius); xd<=std::min(width-1, x+radius); xd++)
				{
					for(int yd=std::max(0,y-radius); yd<=std::min(height-1, y+radius); yd++)
					{
						long index = Alpha.Index(xd, yd);
						if(!_isnan(Alpha[index]))
						{
							pixels ++;

							float diff = 0;
							for(int c=0; c<significant; c++)
							{
								long cHW = c*HW;
								diff += std::abs(Image[i+cHW] - Image[index+cHW]);
							}

							diff /= significant;

							double weight = exp(-diff/sigma_color);

							value += Alpha[index]*weight;
							weights += weight;					

							if(!_isnan(Bg->data[index]))
							{
								pixelsB ++;
								weightsB += weight;
								for(int c=0; c<colors; c++)
								{
									colorsB[c] += Bg->data[index+c*HW] * weight;
								}
							}

							if(!_isnan(Fg->data[index]))
							{
								pixelsF ++;
								weightsF += weight;
								for(int c=0; c<colors; c++)
								{
									colorsF[c] += Fg->data[index+c*HW] * weight;
								}
							}
						}
					}			
				}

				if(pixelsF > 0 && pixelsB > 0 && weights > 0.f)
				{
					float alpha = value/weights;
					AlphaTmp->data[i] = alpha < 0.f ? 0.f : (alpha > 1.f ? 1.f : alpha);
					for(int c=0; c<colors; c++)
					{
						Fg->data[i + c*HW] = colorsF[c]/weightsF;
						Bg->data[i + c*HW] = colorsB[c]/weightsB;
					}
				}
			

				//if(pixels > maxPixels)
				//{				
				//	maxPixels = pixels;
				//	bestIndex = i;
				//	//mexPrintf(" %d(%d); ", bestIndex, maxPixels);
				//}			
			}

			delete colorsB;
			delete colorsF;
		}

		updated = 0;
		for(long i=0; i<HW; i++)
		{
			if(_isnan(Alpha[i]) && !_isnan(AlphaTmp->data[i]))
			{
				float alpha =  AlphaTmp->data[i];
				if(alpha > 0.95)
				{
					alpha = 1;
					for(int c=0; c<colors; c++)
					{
						Bg->data[i + c*HW] = nan;
					}
				}
				else if(alpha < 0.05)
				{
					alpha = 0;
					for(int c=0; c<colors; c++)
					{
						Fg->data[i + c*HW] = nan;
					}
				}

				Alpha.data[i] = alpha;
				updated ++;
			}
		}
		
		//if(bestIndex < 0)
		//{
		//	mexErrMsgTxt("something wrong :( \n");
		//	return;
		//}

		//const int x = bestIndex / height;
		//const int y = bestIndex % height;	
		//mexPrintf("<%d,%d:(%d)>; ", x, y, maxPixels);
		//if(iter % 20 == 0)
		//	mexPrintf("\n");

		//double weights = 0.f;
		//double value = 0.f; 
		//for(int xd=std::max(0,x-radius); xd<=std::min(width-1, x+radius); xd++)
		//{
		//	for(int yd=std::max(0,y-radius); yd<=std::min(height-1, y+radius); yd++)
		//	{
		//		long index = Alpha.Index(xd, yd);
		//		if(!_isnan(Alpha[index]))
		//		{
		//			float diff = 0;
		//			for(int c=0; c<colors; c++)
		//			{
		//				long cHW = c*HW;
		//				diff += std::abs(Image[bestIndex+cHW] - Image[index+cHW]);
		//			}

		//			diff /= colors;

		//			double weight = exp(-diff/sigma_color);

		//			value += Alpha[index]*weight;
		//			weights += weight;					
		//		}
		//	}
		//}

	//	if(weights > 0.f)
	//	{
	//		float alpha = value/weights;
	//		Alpha.data[bestIndex] = alpha < 0.f ? 0.f : (alpha > 1.f ? 1.f : alpha);
	//	}
	//	else
	//	{
	//		Alpha.data[bestIndex] = 0;
	//	}

		iter ++;
	}

}



