/** Depth Occlusion filling using hypothesis filtering
* @author Sergey Smirnov
* @date 30.03.2010
*/
#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}
#endif // __cplusplus
#include <math.h>
#include <vector>
#include <algorithm>
#include "../common/defines.h"
#include "../common/matching.h"
#include "../common/common.h"
//#include "omp.h"

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")

#define MAXRADIUS 40
#define MAXLAYERS 3
#define ISLEFT true
#define MAXWND WINDOW(MAXRADIUS)
#define MAXWND3  (MAXWND*3)
#define RADIUS 10
//#define MAXHEIGHT 2000 
#define CONFIDENCE_DECREASE 0.9f
#define SIGMA_COLOR 20.f
#define SIGMA_DISTANCE 30.f
#define SEARCH_LIMIT 2.5f
#define MAXCost 3.f
#define MAXDIFF 255*3+1 // Maximum SAD distance in RGB24

//!static memory allocation
float weights_table[MAXDIFF], distance[MAXWND];

struct conf_indexes
{
	int indexes[MAXLAYERS];
	float conf[MAXLAYERS];
   
	inline conf_indexes()
	{
		for(int i=0; i<MAXLAYERS; i++)
		{
			indexes[i] = 0;
			conf[i] = 0;
		}
	};

	inline conf_indexes(
      const int*& _indexes, 
      const float*& _conf
	  )
	{
		for(int i=0; i<MAXLAYERS; i++)
		{
			indexes[i] = _indexes[i];
			conf[i] = _conf[i];
		}
	};
};



//! Return layer numbers, sorted by confidence (largest first)
conf_indexes getIndexes(float* Arr, int step, int layers)
{
	conf_indexes indexes;

	//something like stupid bubble-sorting 
	for(int i=0; i<layers; i++)
	{
		for(int j=0; j<layers; j++)
		{
			if(Arr[i*step] > indexes.conf[j])
			{
				for(int jj=layers-1; jj>j; jj--)
				{
					indexes.conf[jj] = indexes.conf[jj-1];
					indexes.indexes[jj] = indexes.indexes[jj-1];
				}
				indexes.conf[j] = Arr[i*step];
				indexes.indexes[j] = i;

				break;
			}
		}
	}
	return indexes;
}

void mexFunction (int nout, mxArray* output[], int in, const mxArray* input[])
{
	if(in < 4 || in > 11 || nout != 2 || mxGetClassID(input[0])!=mxUINT8_CLASS || mxGetClassID(input[1])!=mxSINGLE_CLASS || mxGetClassID(input[2])!=mxLOGICAL_CLASS || mxGetClassID(input[3])!=mxSINGLE_CLASS)
	{
		mexErrMsgTxt("USAGE: [DepthNew, ConfNew] = hypothesis_occlusion_predictor(uint8(Image), single(Depth), logical(OcclMap), single(Confidence) [, isLeft, maxdisp, radius, confidence_decrease, sigma_color, sigma_distance, search_limit]);");
    } 
	const int height = (mxGetDimensions(input[0]))[0];
	const int width = (mxGetDimensions(input[0]))[1];
	const int HW = width*height;
	const int HW_2 = HW*2;

	if(height != (mxGetDimensions(input[1]))[0] || height != (mxGetDimensions(input[2]))[0] || height != (mxGetDimensions(input[3]))[0])
	{
		mexErrMsgTxt("Sizes of input images must coincide."); 
	}
	if(width != (mxGetDimensions(input[1]))[1] || width != (mxGetDimensions(input[2]))[1] || width != (mxGetDimensions(input[3]))[1])
	{
		mexErrMsgTxt("Sizes of input images must coincide."); 
	}

	
	const int layers = mxGetNumberOfDimensions(input[1]) > 2 ? MIN((mxGetDimensions(input[1]))[2], MAXLAYERS) : 1;
	if(layers > 1)
	{
		if(layers != (mxGetDimensions(input[1]))[2] || layers != (mxGetDimensions(input[2]))[2])
		{
			mexErrMsgTxt("Number of layers in Depth, OcclMap, Confidence must coincide."); 
		}
	}

	const UINT8 *Image = (UINT8*)mxGetData(input[0]);
	const float *Depth = (float*)mxGetData(input[1]);
	const bool *Mask = (bool*)mxGetData(input[2]);
    const float *Confidence = (float*)mxGetData(input[3]);
    
    const bool isLeft = (in > 4) ? (bool)mxGetScalar(input[4]) : ISLEFT;
	const int maxdisp = (in > 5) ? MIN((int)mxGetScalar(input[5]), width/2) : 20;
	const int radius = (in > 6) ? MIN((int)mxGetScalar(input[6]), MAXRADIUS) : RADIUS;
	const float confidence_decrease = (in > 7) ? (float)mxGetScalar(input[7]) : CONFIDENCE_DECREASE;
    const float sigma_color = (in > 8) ? (float)mxGetScalar(input[8]) : SIGMA_COLOR;
    const float sigma_distance = (in > 9) ? (float)mxGetScalar(input[9]) : radius;
	const float search_limit = (in > 10) ? (float)mxGetScalar(input[10]) : SEARCH_LIMIT;
    
	const int window = WINDOW(radius);
	const int windowL = window*layers;

	const size_t dims[] = {(unsigned)height, (unsigned)width, (unsigned)layers};
	output[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);	
	output[1] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);	
	float *Filtered = (float*) mxGetData(output[0]);
	float *Conf = (float*) mxGetData(output[1]);

	// pre-calculation of color & distance weights
    init_color_SAD(weights_table, sigma_color, MAXDIFF);    
    init_distance(distance, sigma_distance, radius);

	for(int i=0; i<HW; i++)
	{
		conf_indexes ind = getIndexes((float*)Confidence+i, HW, layers);
		
		for(int c=0; c<layers; c++)
		{
			//bool occl = Mask[i + ind.indexes[c]*HW];
			Conf[i+c*HW] = ind.conf[c];
			Filtered[i+c*HW] = Depth[i + ind.indexes[c]*HW];
		}
	}


	//omp_set_dynamic(0);
	//omp_set_num_threads(2);
	//#pragma omp parallel for 
	for(int xi=0; xi<(int)width; xi++)	
	{
		int x = (xi<=maxdisp) ? (maxdisp-xi) : xi;
		x = (isLeft) ? x : width-1-x;
		for(int y=0; y<static_cast<int>(height); y++)	
		{
			float weights[MAXWND3], blkDepth[MAXWND3], blkConf[MAXWND3];
			int index = INDEX(x,y,height);
			float Dnow = Depth[index];

			// if no occlusion here go to the next pixel
			bool isGood = true;
			for(int c=0; c<layers; c++)
			{
				isGood = isGood && Mask[index];
			}
			if(isGood)
			{
				continue;
			}

			// read depth data
			readBlockFloat_zeropadded((float *)Filtered, blkDepth, x, y, radius, width, height, layers);
			readBlockFloat_zeropadded((float *)Conf, blkConf, x, y, radius, width, height, layers);
            
            float Dmax = Dnow, Dmin = Dnow;
            for(int i=0; i<windowL; i++)
            {
                Dmax = (blkDepth[i] > Dmax) ? blkDepth[i] : Dmax;
                Dmin = (blkDepth[i] < Dmin) ? blkDepth[i] : Dmin;
            }
            //if(abs(Dmax - Dmin) < 0.01)//search_step)
            //{
            //    Filtered[index] = Dnow;
            //    continue;
            //}
            Dmin = floor(Dmin-1);
            Dmax = ceil(Dmax+1);
                        
			//1. calculate color weights as usually
			calculateWeights_zeropadded((UINT8*)Image, weights, weights_table, radius, x, y, width, height);
			
			// 2. copy color weights to other layers
			for(int c=1; c<layers; c++)
			{
				for(int i=0; i<window; i++)
				{
					weights[i+c*window] = weights[i];
				}
			}

			// 3. apply distance weights
			float summ = 0;
			for(int c=0; c<layers; c++)
			{
				for(int i=0; i<window; i++)
				{
					weights[i+c*window] *= distance[i];//*blkConf[i];
					//maxConf = (blkConf[i] > maxConf) ? blkConf[i] : maxConf;
					summ += weights[i+c*window] ;
				}
			}
			//2. normalize 'em
			for(int i=0; i<windowL; i++)
			{
				weights[i] /= summ;
			}

			//3.a apply them to find overal weighted confidence
			//3.b simultaneously reweight to make them confidence-aware
			float avgConf = 0;
			summ = 0;
			for(int i=0; i<windowL; i++)
			{
				avgConf += blkConf[i]*weights[i];
				weights[i] *= blkConf[i];
				summ += weights[i];
			}

			//4. normalize once again
			for(int i=0; i<windowL; i++)
			{
				weights[i] /= summ;
			}

            float Cost[255];
			int hypoMax = (int)ceil((Dmax-Dmin));

            float Cbest = MAXCost;
            float Dbest = Dnow; 
            int Hbest = -1;
            
            for(int hypo = 0; hypo <= hypoMax; hypo++)
            { 
                float d = (float)Dmin + hypo;
                float value = 0;
                for(int i=0; i<windowL; i++)
                {
                    float diff = d-blkDepth[i];
                    diff *= diff;
                    diff = diff>search_limit ? search_limit : diff;
                    value += diff*weights[i];
                }
                Cost[hypo] = value;
            }

			std::pair<float, float> cl = winner_takes_all(Cost, hypoMax, MAXCost);
			Filtered[index] = cl.first+Dmin;
			//Conf[index] = maxConf*confidence_decrease;
			Conf[index] = MIN(avgConf*confidence_decrease, 0.5);
		}	
	}
}