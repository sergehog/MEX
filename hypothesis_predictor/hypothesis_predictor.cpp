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
#include "omp.h"

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")

#define MAXRADIUS 40
#define ISLEFT true
#define MAXWND WINDOW(MAXRADIUS)
#define MAXWND3  (MAXWND*3)
#define RADIUS 10
//#define MAXHEIGHT 2000 
#define CONFIDENCE_DECREASE 0.1f
#define SIGMA_COLOR 20.f
#define SIGMA_DISTANCE 30.f
#define SEARCH_LIMIT 2.5f
#define MAXCost 3.f
#define MAXDIFF 255*3+1 // Maximum SAD distance in RGB24

//static memory allocation
float weights_table[MAXDIFF], distance[MAXWND];

void mexFunction (int nout, mxArray* output[], int in, const mxArray* input[])
{
	if(in < 4 || in > 11 || nout != 2 || mxGetClassID(input[0])!=mxUINT8_CLASS || mxGetClassID(input[1])!=mxSINGLE_CLASS || mxGetClassID(input[2])!=mxLOGICAL_CLASS || mxGetClassID(input[3])!=mxSINGLE_CLASS)
	{
		mexErrMsgTxt("USAGE: [DepthNew, ConfNew] = hypothesis_predictor(uint8(Image), single(Depth), logical(OcclMap), single(Confidence) [, isLeft, maxdisp, radius, confidence_decrease, sigma_color, sigma_distance, search_limit]);");
    } 
	const int height = (mxGetDimensions(input[0]))[0];
	const int width = (mxGetDimensions(input[0]))[1];
	const int HW = width*height;
	const int HW_2 = HW*2;

	if(height != (mxGetDimensions(input[1]))[0] || height != (mxGetDimensions(input[2]))[0] || height != (mxGetDimensions(input[3]))[0])
	{
		mexErrMsgTxt("Sizes of input images must coincide"); 
	}
	if(width != (mxGetDimensions(input[1]))[1] || width != (mxGetDimensions(input[2]))[1] || width != (mxGetDimensions(input[3]))[1])
	{
		mexErrMsgTxt("Sizes of input images must coincide"); 
	}

	const UINT8 *Image = (UINT8*)mxGetData(input[0]);
	const float *Depth = (float*)mxGetData(input[1]);
	const bool *Mask = (bool*)mxGetData(input[2]);
    const float *ConfIn = (float*)mxGetData(input[3]);
    
    const bool isLeft = (in > 4) ? (bool)mxGetScalar(input[4]) : ISLEFT;
	const int maxdisp = (in > 5) ? MIN((int)mxGetScalar(input[5]), width/2) : 20;
	const int radius = (in > 6) ? MIN((int)mxGetScalar(input[6]), MAXRADIUS) : RADIUS;
	const float confidence_decrease = (in > 7) ? (float)mxGetScalar(input[7]) : CONFIDENCE_DECREASE;
    const float sigma_color = (in > 8) ? (float)mxGetScalar(input[8]) : SIGMA_COLOR;
    const float sigma_distance = (in > 9) ? (float)mxGetScalar(input[9]) : SIGMA_DISTANCE;
	const float search_limit = (in > 10) ? (float)mxGetScalar(input[10]) : SEARCH_LIMIT;
    
	int window = WINDOW(radius);

	const unsigned dims[] = {height, width, 1};
	output[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);	
	output[1] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);	
	float *Filtered = (float*) mxGetData(output[0]);
	float *Conf = (float*) mxGetData(output[1]);

	// pre-calculation of color & distance weights
    init_color_SAD(weights_table, sigma_color, MAXDIFF);    
    init_distance(distance, sigma_distance, radius);

	for(int i=0; i<HW; i++)
	{
		Filtered[i] = Depth[i];
		Conf[i] = Mask[i] ? ConfIn[i] : 0.f;
	}

//omp_set_dynamic(0);
//omp_set_num_threads(2);
//#pragma omp parallel for 

	for(int xi=0; xi<(int)width; xi++)	
	{

		int x = (xi<=maxdisp) ? (maxdisp-xi) : xi;
		for(int y=0; y<static_cast<int>(height); y++)	
		{
			float weights[MAXWND], blkDepth[MAXWND], blkConf[MAXWND], blkConfIn[MAXWND], weightsIn[MAXWND];
			
			int index = INDEX(x,y,height);
			float Dnow = Depth[index];

			// is occlusion here?
			if(Mask[index])
			{
				continue;
			}

			// read depth data
			readBlockFloat_zeropadded((float *)Filtered, blkDepth, x, y, radius, width, height);
			readBlockFloat_zeropadded((float *)Conf, blkConf, x, y, radius, width, height);
			readBlockFloat_zeropadded((float *)ConfIn, blkConfIn, x, y, radius, width, height);
            
            float Dmax = Dnow, Dmin = Dnow;
            for(int i=0; i<window; i++)
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
                        
			// prepare color weights
			calculateWeights_zeropadded((UINT8*)Image, weights, weights_table, radius, x, y, width, height);			           
			float maxConf = 0, avgConf = 0;
			float norm = 0, normIn = 0;
			for(int i=0; i<window; i++)
			{
				weights[i] *= distance[i];
				weightsIn[i] = weights[i]*blkConfIn[i];
				weights[i] *= blkConf[i];
				maxConf = (blkConf[i] > maxConf) ? blkConf[i] : maxConf;
				norm += weights[i];
				normIn += weightsIn[i]; 
				
			}			
			//normalizeWeights(weights, radius);
			for(int i=window-1; i>=0; i--)
			{
				weights[i] /= norm;
				weightsIn[i] /= normIn;
				avgConf += blkConf[i]*weights[i];
			}
			//avgConf /= window;

            float Cost[255];
			float Cost2[255];
			int hypoMax = (int)ceil((Dmax-Dmin));

            float Cbest = MAXCost;
            float Dbest = Dnow; 
            int Hbest = -1;
            
            for(int hypo = 0; hypo <= hypoMax; hypo++)
            { 
                float d = (float)Dmin + hypo;
                float value = 0, value2 = 0;
                for(int i=0; i<window; i++)
                {
                    float diff = d-blkDepth[i];
                    diff *= diff;
                    diff = diff>search_limit ? search_limit : diff;
                    value += diff*weights[i];
					value2 += diff*weightsIn[i];
                }
                Cost[hypo] = value;
				Cost2[hypo] = value2;
            }

			std::pair<float, float> predictor = winner_takes_all(Cost, hypoMax, MAXCost);
			std::pair<float, float> filter = winner_takes_all(Cost2, hypoMax, MAXCost);
			
			Filtered[index] = predictor.first+Dmin;

			//Conf[index] = abs(predictor.first-filter.first)<0.1 ? avgConf : avgConf*confidence_decrease;
			Conf[index] = avgConf*confidence_decrease;
		}	
	}



}