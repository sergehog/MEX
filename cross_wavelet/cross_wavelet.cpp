/** Cross-filtering with edge-aware wavelet
*	@file cross_wavelet.cpp
*	@date 10.06.2013
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

//#define EXPERIMENTAL_WEIGHT
#define WEIGHTED_DECIMATION
#define MAXDIFF 256
#define colors 3
#define r 2

inline float getWeight(MexImage<float> &img1, MexImage<float> &img2, const int x1, const int y1, const int x2, const int y2, const float * const exp_lookup, const float agg_factor = 1.f)
{
	float diff = 0.f;
	for(int c=0; c<colors; c++)
	{
		diff += abs(img1(x1,y1,c) - img2(x2,y2,c));
	}
	diff = diff * agg_factor;
	return exp_lookup[int(diff+0.5)];
}


//inline float getAggregationWeight(MexImage<float> &img1, MexImage<float> &img2, const int x1, const int y1, const int x2, const int y2, const float * const exp_lookup, const float agg_factor = 1.f)
//{
//	float diff = 0.f;
//	for(int c=0; c<img1.layers; c++)
//	{
//		diff += abs(img1(x1,y1,c) - img2(x2,y2,c));
//	}
//	diff = diff * agg_factor;
//
//	return exp_lookup[int(diff + 0.5)];
//}




void mexFunction (int nout, mxArray* output[], int in, const mxArray* input[])
{
#ifndef _DEBUG
	omp_set_num_threads(std::max<int>(4, omp_get_max_threads()));
	omp_set_dynamic(std::max<int>(2, omp_get_max_threads()/2));
#endif
	
	if(in < 3 || in > 6 || nout != 1 || mxGetClassID(input[0]) != mxSINGLE_CLASS || mxGetClassID(input[1]) != mxSINGLE_CLASS)
	{
		mexErrMsgTxt("USAGE: [Filtered] = cross_wavelet(single(Signal), single(Image), layers <, sigma, sigma2, param>);");
    }

	

	MexImage<float> Signal(input[0]);	
	MexImage<float> Image(input[1]);
		
	const int height = Image.height;
	const int width = Image.width;
	//const int colors = Image.layers;
	const int signal_layers = Signal.layers;
	const long HW = height*width;
	const float nan = sqrt(-1.f);
	//const int r = 2;

	if(Signal.height != height || Signal.width != width)
	{
		mexErrMsgTxt("Image and Signal must have the same width and height.");
	}

	if (Image.layers != colors)
	{
		mexErrMsgTxt("Only RGB images supported at moment! ");
	}

			
	const int layers = std::max(1, (int)mxGetScalar(input[2])); // pyramid layers
	const float sigma = (in > 3) ? (float)mxGetScalar(input[3]) : 10.f;
	//const float sigma2 = (in > 4) ? (float)mxGetScalar(input[4]) : sigma/2;	
	const float sigma2 = sigma / 2;
	const float param = (in > 5) ? std::min(100u, unsigned(mxGetScalar(input[5])))/100.f : 1.f;
	//const float thr = 1.f;
	const size_t dims[] = {(size_t)height, (size_t)width, (size_t)signal_layers};		

	output[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL); 	
	float * const exp_lookup = new float[MAXDIFF*colors];
	for(int i=0; i<MAXDIFF*colors; i++)
	{
		exp_lookup[i] = exp(-float(i)/(colors * sigma));
	}

	float * const exp_lookup2 = new float[MAXDIFF*colors];
	for (int i = 0; i<MAXDIFF*colors; i++)
	{
		exp_lookup2[i] = exp(-float(i) / (colors * sigma2));
	}

	MexImage<float> Filtered(output[0]);		
	//Filtered.setval(0.f);
	#pragma omp parallel for
	for(long i=0; i<HW*signal_layers; i++)
	{
		Filtered[i] = Signal[i];		
	}

	// pyramid of signals and images
	MexImage<float> ** Signals = new MexImage<float>*[layers+1];
	MexImage<float> ** Images = new MexImage<float>*[layers+1];

	Signals[0] = &Filtered;
	Images[0] = &Image;	
	
	
	// first pass from finer to coarse levels
	for(int layer=1; layer <= layers; layer++)
	{		
		MexImage<float> &fineImage = *Images[layer-1];
		MexImage<float> &fineSignal = *Signals[layer-1];

		const int fineWidth = fineImage.width; 
		const int fineHeight = fineImage.height;
		//const long fineHW = fineImage->layer_size;

		//mexPrintf("layer %d: [%d,%d,%d] -> [%d,%d,%d]\n", layer, fineWidth, fineHeight, fineHW, coarseWidth, coarseHeight, coarseHW);
		//mexPrintf("  <[%d,%d,%d|%d]>  \n", coarseSignal->width, coarseSignal->height, coarseSignal->layers, coarseSignal->layer_size);
		

		if(layer % 2)
		{
			const int coarseWidth = fineWidth/2 + fineWidth%2; 
			const int coarseHeight = fineHeight;
			//const long coarseHW = coarseWidth*coarseHeight;
		
			Images[layer] = new MexImage<float>(coarseWidth, coarseHeight, colors);
			Signals[layer] = new MexImage<float>(coarseWidth, coarseHeight, signal_layers);	

			MexImage<float> &coarseImage = *Images[layer];
			MexImage<float> &coarseSignal = *Signals[layer];

			coarseImage.setval(0.f);
			coarseSignal.setval(0.f);


			// horisontal decimation
			#pragma omp parallel for
			for(int y=0; y<fineHeight; y++)
			{				
				for(int fx=0; fx<fineWidth; fx+=2)
				{											
					const int cx = fx/2;					
					
					for(int c=0; c<colors; c++)
					{
						coarseImage(cx,y,c) = fineImage(fx,y,c);
					}

					for(int l=0; l<signal_layers; l++)
					{
						coarseSignal(cx,y,l) = fineSignal(fx,y,l);
					}

					// weight from left
					float weight1 = (fx>0) ? getWeight(fineImage, fineImage, fx, y, fx - 1, y, exp_lookup) : 0.f;
					
					// weight from right
					float weight2 = (fx<fineWidth - 1) ? getWeight(fineImage, fineImage, fx, y, fx + 1, y, exp_lookup) : 0.f;
					
#ifdef WEIGHTED_DECIMATION
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
#endif					
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
			const int coarseWidth = fineWidth; 
			const int coarseHeight = fineHeight/2 + fineHeight%2;
			//const long coarseHW = coarseWidth*coarseHeight;
		
			Images[layer] = new MexImage<float>(coarseWidth, coarseHeight, colors);
			Signals[layer] = new MexImage<float>(coarseWidth, coarseHeight, signal_layers);	

			MexImage<float> &coarseImage = *Images[layer];
			MexImage<float> &coarseSignal = *Signals[layer];

			coarseImage.setval(0.f);
			coarseSignal.setval(0.f);
			

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
					
					// weight from left
					float weight1 = (fy>0) ? getWeight(fineImage, fineImage, x, fy, x, fy - 1, exp_lookup) : 0.f;
					
					// weight from right
					float weight2 = (fy<fineHeight - 1) ? getWeight(fineImage, fineImage, x, fy, x, fy + 1, exp_lookup) : 0.f;

#ifdef WEIGHTED_DECIMATION
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
#endif
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

	// second pass from coarse to finer levels
	for(int layer=layers-1; layer >= 0; layer--)
	{
		const float power = 0.4f;
		const float agg_factor = layer > 1 ? std::min(1.f, powf(float(int(layer / 2)) / int(layers / 2), power)) : 0;

		//const float agg_factor = float(int(layer / 2)) / int(layers / 2);
		//const float agg_factor = int(layer / 2) ? 1 : 0;
		//const float agg_factor = 1;		

		MexImage<float> &fineImage = *Images[layer];
		MexImage<float> &fineSignal = *Signals[layer];

		MexImage<float> &coarseImage = *Images[layer+1];
		MexImage<float> &coarseSignal = *Signals[layer+1];

		int fineWidth = fineImage.width; 
		int fineHeight = fineImage.height;
		//long fineHW = fineImage->layer_size;

		int coarseWidth = coarseImage.width;
		int coarseHeight = coarseImage.height;
		//long coarseHW = coarseImage->layer_size;
		//mexPrintf("layer %d\n", layer);

		if(layer % 2) // vertical upsampling ? 
		{
			
			#pragma omp parallel for
			for(int x=0; x<fineWidth; x++)			
			{				
				for(int fy=0; fy<fineHeight; fy++)
				{																
					const int cy = fy/2;

#ifdef EXPERIMENTAL_WEIGHT
					float weights = 0;
					float weights_list[r*2+1];
					float weight_max = 0;

					for(int cyy=std::max<int>(0,cy-r), w=0; cyy<=std::min<int>(coarseHeight-1, cy+r); cyy++, w++)					
					{
						if (cyy == cy)
							continue;
						weights_list[w] = getWeight(fineImage, coarseImage, x, fy, x, cyy, exp_lookup);
						weights += weights_list[w];
						weight_max = (weights_list[w] > weight_max) ? weights_list[w] : weight_max;
					}

					const float weight0 = (1 - weight_max);
					weights += weight0;
					for (int l = 0; l<signal_layers; l++)
					{
						fineSignal(x, fy, l) *= weight0;
					}

					for (int cyy = std::max<int>(0, cy - r), w = 0; cyy <= std::min<int>(coarseHeight - 1, cy + r); cyy++, w++)
					{
						if (cyy == cy)
							continue;
						const float weight = weights_list[w];
						for (int l = 0; l<signal_layers; l++)
						{
							fineSignal(x, fy, l) += coarseSignal(x, cyy, l)*weight; // just copy corresponding value
						}
					}


					for(int l=0; l<signal_layers; l++)
					{
						fineSignal(x,fy,l) /= weights;
					}


#else

					if(!fy%2)
					{
						// weight between fine color value and corresponding coarse image value												
						const float weight = getWeight(fineImage, coarseImage, x, fy, x, cy, exp_lookup2, agg_factor);
						const float weight0 = std::min(param, (1 - weight)); //std::max(0.f, thr - weight);
						
						for(int l=0; l<signal_layers; l++)
						{
							fineSignal(x, fy, l) *= weight0;
							fineSignal(x,fy,l) += coarseSignal(x,cy,l)*weight; 
							fineSignal(x,fy,l) /= (weight0 + weight);
						}

					}
					else
					{
						const float weight1 = getWeight(fineImage, coarseImage, x, fy, x, cy, exp_lookup2, agg_factor);
						const float weight2 = (cy<coarseHeight - 1) ? getWeight(fineImage, coarseImage, x, fy, x, cy + 1, exp_lookup2, agg_factor) : 0.f;
						const float weight0 = std::min(param, 1 - std::max(weight1, weight2));

						for(int l=0; l<signal_layers; l++)
						{
							fineSignal(x,fy,l) *= weight0;
							fineSignal(x,fy,l) += coarseSignal(x,cy,l) * weight1; 
							if(cy<coarseHeight-1)
							{
								fineSignal(x,fy,l) += coarseSignal(x,cy+1,l) * weight2; 
							}
							fineSignal(x,fy,l) /= (weight0 + weight1 + weight2);
						}
					}
#endif
				}
			}

		}
		else // horisontal upsampling ?  
		{
			#pragma omp parallel for
			for(int y=0; y<fineHeight; y++)
			{				
				for(int fx=0; fx<fineWidth; fx++)
				{																
					const int cx = fx/2;

#ifdef EXPERIMENTAL_WEIGHT
					float weights = 0;
					float weights_list[r * 2 + 1];
					float weight_max = 0;

					for (int cxx = std::max<int>(0, cx - r), w = 0; cxx <= std::min<int>(coarseWidth - 1, cx + r); cxx++, w++)					
					{
						if (cxx == cx)
							continue;
						weights_list[w] = getWeight(fineImage, coarseImage, fx, y, cxx, y, exp_lookup);
						weights += weights_list[w];
						weight_max = (weights_list[w] > weight_max) ? weights_list[w] : weight_max;
					}

					const float weight0 = (1 - weight_max);
					weights += weight0;
					for (int l = 0; l<signal_layers; l++)
					{
						fineSignal(fx, y, l) *= weight0;
					}
					for (int cxx = std::max<int>(0, cx - r), w = 0; cxx <= std::min<int>(coarseWidth - 1, cx + r); cxx++, w++)
					{
						if (cxx == cx)
							continue;
						const float weight = weights_list[w];
						for (int l = 0; l<signal_layers; l++)
						{
							fineSignal(fx, y, l) += coarseSignal(cxx, y, l)*weight; // just copy corresponding value
						}
					}


					for (int l = 0; l<signal_layers; l++)
					{
						fineSignal(fx, y, l) /= weights;
					}

					//float weights = 1;					
					//for(int cxx=std::max<int>(0,cx-r); cxx<=std::min<int>(coarseWidth-1, cx+r); cxx++)
					//{
					//	if (cxx == cx)
					//		continue;
					//	const float weight = getWeight(fineImage, coarseImage, fx, y, cxx, y, exp_lookup);			
					//	for(int l=0; l<signal_layers; l++)
					//	{
					//		fineSignal(fx,y,l) += coarseSignal(cxx,y,l)*weight; // just copy corresponding value
					//	}
					//	weights += weight;
					//}

					//for(int l=0; l<signal_layers; l++)
					//{
					//	fineSignal(fx,y,l) /= weights;
					//}						

#else						
					if(!fx%2)
					{

						// weight between fine color value and left coarse image value
						const float weight = getWeight(fineImage, coarseImage, fx, y, cx, y, exp_lookup2, agg_factor);
						const float weight0 = std::min(param, (1 - weight)); //std::max(0.f, thr - weight);

						for(int l=0; l<signal_layers; l++)
						{							
							fineSignal(fx, y, l) *= weight0;
							fineSignal(fx, y, l) += coarseSignal(cx, y, l)*weight; // just copy corresponding value
							fineSignal(fx, y, l) /= (weight0 + weight);
						}						

					}
					else
					{
						const float weight1 = getWeight(fineImage, coarseImage, fx, y, cx, y, exp_lookup2, agg_factor);
						const float weight2 = (cx<coarseWidth - 1) ? getWeight(fineImage, coarseImage, fx, y, cx + 1, y, exp_lookup2, agg_factor) : 0.f;
						const float weight0 = std::min(param, 1 - std::max(weight1, weight2));
						for(int l=0; l<signal_layers; l++)
						{
							fineSignal(fx,y,l) *= weight0;
							fineSignal(fx,y,l) += coarseSignal(cx,y,l) * weight1; 
							if(cx<coarseWidth-1)
							{
								fineSignal(fx,y,l) += coarseSignal(cx+1,y,l) * weight2; 
							}
							fineSignal(fx,y,l) /= (weight0 + weight1 + weight2);
						}	

						
					}
#endif						
				}
			}

		}
	}
	
	#pragma omp parallel for
	for(int layer=1; layer <= layers; layer++)
	{
		delete Images[layer];
		delete Signals[layer];
	}

	delete[] Signals;
	delete[] Images;
	delete[] exp_lookup;
	delete[] exp_lookup2;
}


/*
//#define EXPERIMENTAL_SUPPORT 
//#define EXPERIMENTAL_WEIGHT
#define EXPERIMENTAL_INTERPOLATION


inline float getWeight(const float weight, const int colors, const float sigma)
{
#ifdef EXPERIMENTAL_WEIGHT
	return exp(-sqrt(weight)/(colors*sigma));
#else
	return exp(-weight/(colors*sigma));
#endif
}


void mexFunction (int nout, mxArray* output[], int in, const mxArray* input[])
{
	omp_set_num_threads(std::max<int>(4, omp_get_max_threads()));
	omp_set_dynamic(std::max<int>(2, omp_get_max_threads()/2));
	
	if(in < 3 || in > 4 || nout != 1 || mxGetClassID(input[0]) != mxSINGLE_CLASS || mxGetClassID(input[1]) != mxSINGLE_CLASS)
	{
		mexErrMsgTxt("USAGE: [Filtered] = cross_wavelet(single(Signal), single(Image), layers <, sigma>);");
    }

	MexImage<float> Signal(input[0]);	
	MexImage<float> Image(input[1]);
		
	const int height = Image.height;
	const int width = Image.width;
	const int colors = Image.layers;
	const int signal_layers = Signal.layers;
	const long HW = Image.layer_size;
	const float nan = sqrt(-1.f);
	const int r = 1;

	if(Signal.height != height || Signal.width != width)
	{
		mexErrMsgTxt("Image and Signal must have the same width and height.");
	}
			
	const int layers = std::max(1, (int)mxGetScalar(input[2]));		
	const float sigma = (in > 3) ? (float)mxGetScalar(input[3]) : 10.;
	
	const size_t dims[] = {(size_t)height, (size_t)width, (size_t)signal_layers};		

	output[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL); 	

	MexImage<float> Filtered(output[0]);		
	#pragma omp parallel for
	for(long i=0; i<HW*signal_layers; i++)
	{
		Filtered.data[i] = Signal[i];		
	}
	// pyramid of signals and images
	MexImage<float> ** Signals = new MexImage<float>*[layers+1];
	MexImage<float> ** Images = new MexImage<float>*[layers+1];

	Signals[0] = &Filtered;
	Images[0] = &Image;	

	// first pass from finer to coarse levels
	for(int layer=1; layer <= layers; layer++)
	{
		MexImage<float> *fineImage = Images[layer-1];
		MexImage<float> *fineSignal = Signals[layer-1];

		const int fineWidth = fineImage->width; 
		const int fineHeight = fineImage->height;
		const long fineHW = fineImage->layer_size;

		const int coarseWidth = fineWidth/2 + fineWidth%2; 
		const int coarseHeight = fineHeight/2 + fineHeight%2;
		const long coarseHW = coarseWidth*coarseHeight;
		
		MexImage<float> *coarseImage = new MexImage<float>(coarseWidth, coarseHeight, colors);
		MexImage<float> *coarseSignal = new MexImage<float>(coarseWidth, coarseHeight, signal_layers);
		coarseImage->setval(0.f);
		coarseSignal->setval(0.f);
		//mexPrintf("layer %d: [%d,%d,%d] -> [%d,%d,%d]\n", layer, fineWidth, fineHeight, fineHW, coarseWidth, coarseHeight, coarseHW);
		//mexPrintf("  <[%d,%d,%d|%d]>  \n", coarseSignal->width, coarseSignal->height, coarseSignal->layers, coarseSignal->layer_size);
		
		Images[layer] = coarseImage;
		Signals[layer] = coarseSignal;		

		// downsampling of both signal and image
		#pragma omp parallel for
		for(long findex = 0; findex < fineHW; findex++)				
		{						
			const int fx = findex / fineHeight;
			const int fy = findex % fineHeight;
			const int cx = fx/2;
			const int cy = fy/2;			

			const long cindex = coarseImage->Index(cx, cy);
			float norm = 1;
			norm *= (fx == fineWidth && fineWidth%2) ? 1 : 2;
			norm *= (fy == fineHeight && fineHeight%2) ? 1 : 2;

			//mexPrintf("   <%d,%d=%d | %d,%d=%d>\n", fx, fy, findex, cx, cy, cindex);

			for(int c=0; c<colors; c++)
			{
				coarseImage->data[cindex + coarseHW*c] += (fineImage->data[findex + fineHW*c])/norm;
			}

			for(int l=0; l<signal_layers; l++)
			{
				coarseSignal->data[cindex + coarseHW*l] += (fineSignal->data[findex + fineHW*l])/norm;				
			}
		}
	}

	
	// second pass from coarse to finer levels
	for(int layer=layers-1; layer >= 0; layer--)
	{
		MexImage<float> *fineImage = Images[layer];
		MexImage<float> *fineSignal = Signals[layer];

		MexImage<float> *coarseImage = Images[layer+1];
		MexImage<float> *coarseSignal = Signals[layer+1];


		int fineWidth = fineImage->width; 
		int fineHeight = fineImage->height;
		long fineHW = fineImage->layer_size;

		int coarseWidth = coarseImage->width;
		int coarseHeight = coarseImage->height;
		long coarseHW = coarseImage->layer_size;
		//mexPrintf("layer %d\n", layer);
		#pragma omp parallel for
		for(long findex = 0; findex < fineHW; findex++)				
		{						
			int fx = findex / fineHeight;
			int fy = findex % fineHeight;
			int cx = fx/2;
			int cy = fy/2;
			float weight = 0.f;
			const long cindex = coarseImage->Index(cx, cy);

#ifdef EXPERIMENTAL_SUPPORT // play with the virtual support 

			for(int c=0; c<colors; c++)
			{
				weight += abs(coarseImage->data[cindex + coarseHW*c] - fineImage->data[findex + fineHW*c]);
			}
			weight = getWeight(weight, colors, sigma);

			for(int l=0; l<signal_layers; l++)
			{
				fineSignal->data[findex + fineHW*l] *= (1-weight);
			}			

			const int r = 1;
			float weights[(r*2+1)*(r*2+1)];
			for(int i=0; i<(r*2+1)*(r*2+1); i++) weights[i] = 0;
			float sum = 0;

			// calculate weights of coarse pixels
			for(int ccx=std::max(0,cx-r), w=0; ccx<=std::min(coarseWidth-1, cx+r); ccx++)
			{
				for(int ccy=std::max(0,cy-r); ccy<=std::min(coarseHeight-1, cy+r); ccy++, w++)
				{
					long ccindex = coarseImage->Index(ccx, ccy);
					float diff = 0;
					for(int c=0; c<colors; c++)
					{
						diff += abs(coarseImage->data[ccindex + coarseHW*c] - fineImage->data[findex + fineHW*c]);
					}
					weights[w] = exp(-diff/(colors));
					sum += weights[w];		
				}
			}
			
			// update fine signal with coarse values
			for(int ccx=std::max(0,cx-r), w=0; ccx<=std::min(coarseWidth-1, cx+r); ccx++)
			{
				for(int ccy=std::max(0,cy-r); ccy<=std::min(coarseHeight-1, cy+r); ccy++, w++)
				{
					long ccindex = coarseImage->Index(ccx, ccy);
					for(int l=0; l<signal_layers; l++)
					{
						fineSignal->data[findex + fineHW*l] += coarseSignal->data[ccindex + coarseHW*l]*weight*weights[w]/sum;
					}
				}
			}

#else
			
			int dirx = (fx==0 || fx==fineWidth-1 && fx%2) ? 0 : (fx%2 ? 1 : -1);
			int diry = (fy==0 || fy==fineHeight-1 && fy%2) ? 0 : (fy%2 ? 1 : -1);
						
			const long cindex0 = cindex;
			const long cindex1 = coarseImage->Index(cx, cy+diry);
			const long cindex2 = coarseImage->Index(cx+dirx, cy);
			const long cindex3 = coarseImage->Index(cx+dirx, cy+diry);

	#ifdef EXPERIMENTAL_INTERPOLATION
			for(int c=0; c<colors; c++)
			{
				//float color13 = (coarseImage->data[cindex1 + coarseHW*c]*3 + coarseImage->data[cindex3 + coarseHW*c]);
				//float color02 = (coarseImage->data[cindex + coarseHW*c]*3 + coarseImage->data[cindex2 + coarseHW*c]);
				//float interpolated_color = (color02*3 + color13)/16;
				
				// bilinear interpolation of the color of the finer grid
				float interpolated_color = (3/4.f)*coarseImage->data[cindex0 + coarseHW*c];
				interpolated_color += (3/8.f)*coarseImage->data[cindex1 + coarseHW*c];
				interpolated_color += (3/8.f)*coarseImage->data[cindex2 + coarseHW*c];
				interpolated_color += (1/8.f)*coarseImage->data[cindex3 + coarseHW*c];
				weight += abs(interpolated_color - fineImage->data[findex + fineHW*c]); // difference between true color value and up-sampled
				//weight += abs(coarseImage->data[cindex + coarseHW*c] - fineImage->data[findex + fineHW*c]);
			}
	#else
			
			for(int c=0; c<colors; c++)
			{
				weight += abs(coarseImage->data[cindex + coarseHW*c] - fineImage->data[findex + fineHW*c]);
			}
			
	#endif
			weight = getWeight(weight, colors, sigma);

			
			float weight0 = 0.f, weight1 = 0.f, weight2 = 0.f, weight3 = 0.f;
			for(int c=0; c<colors; c++)
			{				
				weight0 += abs(coarseImage->data[cindex0 + coarseHW*c] - fineImage->data[findex + fineHW*c]);
				weight1 += abs(coarseImage->data[cindex1 + coarseHW*c] - fineImage->data[findex + fineHW*c]);
				weight2 += abs(coarseImage->data[cindex2 + coarseHW*c] - fineImage->data[findex + fineHW*c]);
				weight3 += abs(coarseImage->data[cindex3 + coarseHW*c] - fineImage->data[findex + fineHW*c]);
			}
						
			weight0 = exp(-weight0/(colors*sigma))*3/4.f;
			weight1 = exp(-weight1/(colors*sigma))*3/8.f;
			weight2 = exp(-weight2/(colors*sigma))*3/8.f;
			weight3 = exp(-weight3/(colors*sigma))*1/8.f;
			//float weightX = 1 - weight0;
			//float weightX = std::min(1.f, std::max(0.f, 1.2f - weight0 - weight1 - weight2 - weight3));
			//const float weights = weightX + weight0 + weight1 + weight2 + weight3;
			const float weights = weight0 + weight1 + weight2 + weight3;
			
			//weightX /= weights;
			weight0 /= weights;
			weight1 /= weights;
			weight2 /= weights;
			weight3 /= weights;

			for(int l=0; l<signal_layers; l++)
			{
				//float interpolated_signal = coarseSignal->data[cindex0 + coarseHW*l]*weight0;
				//interpolated_signal += coarseSignal->data[cindex1 + coarseHW*l]*weight1;
				//interpolated_signal += coarseSignal->data[cindex2 + coarseHW*l]*weight2;
				//interpolated_signal += coarseSignal->data[cindex3 + coarseHW*l]*weight3;
				//interpolated_signal /= weights;
				//fineSignal->data[findex + fineHW*l] = interpolated_signal*weight + fineSignal->data[findex + fineHW*l]*(1-weight);
				fineSignal->data[findex + fineHW*l] *= (1-weight);
				fineSignal->data[findex + fineHW*l] += coarseSignal->data[cindex0 + coarseHW*l]*weight0*weight;
				fineSignal->data[findex + fineHW*l] += coarseSignal->data[cindex1 + coarseHW*l]*weight1*weight;
				fineSignal->data[findex + fineHW*l] += coarseSignal->data[cindex2 + coarseHW*l]*weight2*weight;
				fineSignal->data[findex + fineHW*l] += coarseSignal->data[cindex3 + coarseHW*l]*weight3*weight;
			}
#endif


		}		
	}	

	for(int layer=1; layer <= layers; layer++)
	{
		delete Images[layer];
		delete Signals[layer];
	}

	delete[] Signals;
	delete[] Images;
	
}
*/