/** yang_hebf
*	@file yang_hebf.cpp
*	@date 28.01.2014
*	@author Sergey Smirnov <sergey.smirnov@tut.fi>
*/

#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}
#endif // __cplusplus

#include "../common/meximage.h"
#include <vector>
#include <algorithm>
#ifndef _DEBUG
#include <omp.h>
#endif

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")

using namespace mymex;

#define radius 2
#define MAXDIFF 256

float getWeight(MexImage<unsigned char> &Image1, MexImage<unsigned char> &Image2, const int x1, const int y1, const int x2, const int y2, float const * const weights)
{
	int diff = 0;
	for(int c=0; c<Image1.layers; c++)
	{
		int dif = abs(int(Image1(x1,y1,c)) - int(Image2(x2,y2,c)));
		diff = dif > diff ? dif : diff;
	}
	return weights[diff];
}


float getWeightOld(MexImage<unsigned char> &Image1, MexImage<unsigned char> &Image2, const int x1, const int y1, const int x2, const int y2, float const * const weights)
{
	int diff = 0;
	for(int c=0; c<Image1.layers; c++)
	{
		diff += abs(int(Image1(x1,y1,c)) - int(Image2(x2,y2,c)));
	}
	return weights[diff];
}

void mexFunction (int nout, mxArray* output[], int in, const mxArray* input[])
{
#ifndef _DEBUG
	omp_set_num_threads(std::max<int>(4, omp_get_max_threads()));
	omp_set_dynamic(std::max<int>(2, omp_get_max_threads()/2));
#endif
	
	if(in < 3 || in > 4 || nout != 1 || mxGetClassID(input[0]) != mxSINGLE_CLASS || mxGetClassID(input[1]) != mxUINT8_CLASS)
	{
		mexErrMsgTxt("USAGE: [Filtered] = yang_hebf(float(Signal), uint8(Image), layers, <sigma>);");
    }
	
	MexImage<float> Signal(input[0]);
	MexImage<unsigned char> Image(input[1]);
	const int width = Signal.width;
	const int height = Signal.height;
	const int layers = Signal.layers;
	const int colors = Image.layers;
	const long HW = Signal.layer_size;
	const int levels = std::max(1, (int)mxGetScalar(input[2])); 	
	const float sigma = in > 3 ? (float)mxGetScalar(input[3]) : 1.f; 	
	
	if(Image.width != width || Image.height != height)
	{
		mexErrMsgTxt("Resolution of Image and Signal must coincide!");
	}

	const size_t dims[] = {(size_t)height, (size_t)width, (size_t)layers};
	output[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL); 	
	MexImage<float> Filtered(output[0]);
	
	float * const weights_lookup = new float[MAXDIFF*colors];
	#pragma omp parallel for
	for(int i=0; i<MAXDIFF*colors; i++)
	{
		weights_lookup[i] = exp(-i/(sigma*colors));
	}
	
	#pragma omp parallel for
	for(long i=0; i<HW*layers; i++)
	{
		Filtered[i] = Signal[i];
	}

	MexImage<unsigned char> **Images = new MexImage<unsigned char>* [levels];
	MexImage<float> **Signals = new MexImage<float>* [levels];	
	MexImage<float> **Buffers = new MexImage<float>* [levels];

	Images[0] = &Image;
	Signals[0] = &Filtered;	
	Buffers[0] = new MexImage<float>(width, height, layers);

	for(int level = 1; level<levels; level++)
	{
		MexImage<unsigned char> &fineImage = *Images[level-1];
		MexImage<float> &fineSignal = *Signals[level-1];

		const int fineWidth = fineImage.width;
		const int fineHeight = fineImage.height;
		const long fineHW = fineImage.layer_size;

		// handles odd resoluiton of Images
		const int coarseWidth = fineWidth/2 + fineWidth%2;
		const int coarseHeight = fineHeight/2 + fineHeight%2;

		Images[level] = new MexImage<unsigned char>(coarseWidth, coarseHeight, colors);
		Signals[level] = new MexImage<float>(coarseWidth, coarseHeight, layers);		
		Buffers[level] = new MexImage<float>(coarseWidth, coarseHeight, layers);		

		MexImage<unsigned char> &coarseImage = *Images[level];
		MexImage<float> &coarseSignal = *Signals[level];
		coarseImage.setval(0);
		coarseSignal.setval(0);
		
		// downsampling of both signal and image
		#pragma omp parallel for
		for(long ci = 0; ci<coarseImage.layer_size; ci++)
		{
			const int cx = ci / coarseHeight;
			const int cy = ci % coarseHeight;
			for(int c=0; c<colors; c++)
			{
				coarseImage(cx, cy, c) = fineImage(cx*2, cy*2, c);
			}
		}

		#pragma omp parallel for
		for(int fx=0; fx<fineWidth; fx++)
		{
			int normX = fineWidth%2 && fx==fineWidth-1 ? 1 : 2;			
			const int cx = fx/2;

			for(int fy=0; fy<fineHeight; fy++)
			{
				int normY = fineHeight%2 && fy==fineHeight-1 ? 1 : 2;
				const int cy = fy/2;

				for(int l=0; l<layers; l++)
				{
					coarseSignal(cx,cy,l) += fineSignal(fx,fy,l)/(normX*normY);
				}
			}
		}		
	}

	for(int level = levels-1; level>=0; level--)
	{
		MexImage<unsigned char> &fineImage = *Images[level];
		MexImage<float> &fineSignal = *Signals[level];
		MexImage<float> &fineBuffer = *Buffers[level];

		const int fineWidth = fineImage.width;
		const int fineHeight = fineImage.height;
		const long fineHW = fineImage.layer_size;

		#pragma omp parallel for
		for(long i=0; i<fineHW*layers; i++)
		{
			fineBuffer[i] = fineSignal[i];
		}

		//adaptively upsample coarseSignal to fineBuffer
		if(level < levels-1)
		{			
			MexImage<unsigned char> &coarseImage = *Images[level+1];
			MexImage<float> &coarseSignal = *Signals[level+1];
			//MexImage<float> &coarseBuffer = *Buffers[level+1];
			fineBuffer.setval(0);

			const int coarseWidth = coarseImage.width;
			const int coarseHeight = coarseImage.height;
						
			#pragma omp parallel for
			for(long fi=0; fi<fineHW; fi++)
			{
				const int fx = fi / fineHeight;
				const int fy = fi % fineHeight;

				const int cx = fx/2;
				const int cy = fy/2;

				//const float alpha = getWeight(fineImage, coarseImage, fx, fy, cx, cy, weights_lookup);				
				float weights = 1;

				for(int l=0; l<layers; l++)
				{
					//fineBuffer(fx, fy, l) += fineSignal(fx,fy,l)*alpha;
					fineBuffer(fx, fy, l) = fineSignal(fx,fy,l);
				}

				for(int xx=std::max(0,fx-radius); xx<=std::min(fineWidth-1,fx+radius); xx++)
				{
					const int cxx = xx/2;
					//const int cxx = std::min(xx/2, coarseWidth-1);

					for(int yy=std::max(0,fy-radius); yy<=std::min(fineHeight-1,fy+radius); yy++)
					{
						const int cyy = yy/2;
						//const int cyy = std::min(yy/2, coarseHeight-1);

						if(xx==fx && yy==fy)
						{
							continue;
						}

						//const float weight = getWeight(coarseImage, coarseImage, cx, cy, cxx, cyy, weights_lookup);
						const float weight = getWeight(fineImage, coarseImage, fx, fy, cxx, cyy, weights_lookup);
					
						for(int l=0; l<layers; l++)
						{
							fineBuffer(fx, fy, l) += coarseSignal(cxx, cyy, l) * weight;// *alpha;
						}

						weights += weight;
					}
				}

				for(int l=0; l<layers; l++)
				{
					//fineBuffer(fx, fy, l) /= ((1-alpha) + weights*alpha);
					fineBuffer(fx, fy, l) /= weights;
				}
			}			
		}


		//compute the joint bilateral response
		#pragma omp parallel for
		for(long fi=0; fi<fineHW; fi++)
		{
			const int fx = fi / fineHeight;
			const int fy = fi % fineHeight;
			float weights = 0;

			for(int xx=std::max(0,fx-radius); xx<=std::min(fineWidth-1,fx+radius); xx++)
			{
				for(int yy=std::max(0,fy-radius); yy<=std::min(fineHeight-1,fy+radius); yy++)
				{
					const float weight = getWeight(fineImage, fineImage, fx, fy, xx, yy, weights_lookup);
					
					for(int l=0; l<layers; l++)
					{
						fineSignal(fx, fy, l) += fineBuffer(fx, fy, l) * weight;
					}

					weights += weight;
				}
			}

			for(int l=0; l<layers; l++)
			{
				fineSignal(fx, fy, l) /= weights;
			}
		}
	}

	#pragma omp parallel for
	for(int level = 0; level<levels; level++)
	{
		if(level > 0)
		{
			delete Images[level];
			delete Signals[level];
		}
		delete Buffers[level];
	}

	delete[] Signals;
	delete[] Images;
	delete[] Buffers;
	delete[] weights_lookup;
	

}
