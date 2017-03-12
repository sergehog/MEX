/** Bilateral-based signal predition
* @author Sergey Smirnov
* @date 21.10.2012
*/
#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}
#endif // __cplusplus
#include "../common/meximage.h"
#include <math.h>
#include <vector>
#include <algorithm>
//#include "../common/defines.h"
//#include "../common/matching.h"
//#include "../common/common.h"
#include "omp.h"

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")

//#define MAXRADIUS 40
//#define MAXWND WINDOW(MAXRADIUS)
//#define MAXWND3  (MAXWND*3)
//#define RADIUS 10
//#define SIGMA_COLOR 20.f
//#define SIGMA_DISTANCE 30.f
//#define SEARCH_LIMIT 2.5f
//#define MAXCost 3.f
//#define MAXDIFF 255*3+1 // Maximum SAD distance in RGB24

//static memory allocation
//float weights_table[MAXDIFF], distance[MAXWND];

using namespace std;
using namespace mymex;

void mexFunction (int nout, mxArray* output[], int in, const mxArray* input[])
{
	omp_set_num_threads(std::max(2, omp_get_num_threads()));
	omp_set_dynamic(0);

	if(in < 2 || in > 6 || nout != 1 || mxGetClassID(input[0])!=mxSINGLE_CLASS || mxGetClassID(input[1])!=mxSINGLE_CLASS)
	{
		mexErrMsgTxt("USAGE: [Result] = hypothesis_occlusions(single(Signal), single(Image) [, radius, sigma_color, sigma_distance, search_limit]);");
    } 

	MexImage<float> Signal(input[0]);
	MexImage<float> Image(input[1]);

	const int height = Signal.height;
	const int width = Signal.width;
	const int layers = Signal.layers;
	const int colors = Image.layers;
	const long HW = Signal.layer_size;	
	const float nan = sqrt(-1.f);	

	if(height != Image.height || width != Image.width)
	{
		mexErrMsgTxt("Sizes of Signal and Image must coincide!"); 
	}
    
    const int radius = (in > 2) ? min(1, (int)mxGetScalar(input[2])) : 1;
    const float sigma_color = (in > 3) ? (float)mxGetScalar(input[4]) : 20.f;
    const float sigma_distance = (in > 4) ? (float)mxGetScalar(input[5]) : 20.f;
	const float search_limit = (in > 5) ? (float)mxGetScalar(input[6]) : 2.5f;
	const float nan = sqrt(-1.f);
    	

	const unsigned dims[] = {height, width, layers};
	output[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);	
	float *Result = (float*) mxGetData(output[0]);

	// pre-calculation of color & distance weights
    //init_color_SAD(weights_table, sigma_color, MAXDIFF);    
    //init_distance(distance, sigma_distance, radius);

	
	float *minvals = new float[layers*2];
	float *maxvals = minvals + layers;

	for(int i=0; i<layers; i++)
	{
		minvals[i] = nan;
		maxvals[i] = nan;
	}

	for(long i=0; i<HW; i++)
	{
		for(int c=0; c<layers; c++)
		{
			float value = Signal[i+c*HW];
			if(_isnan(value))
			{
				continue;
			}

			minvals[c] = _isnan(minvals[c]) ? value : (value < minvals[c] ? value : minvals[c]);
			maxvals[c] = _isnan(maxvals[c]) ? value : (value > maxvals[c] ? value : maxvals[c]);			
		}
	}
	
	for(int c=0; c<layers; c++)
	{
		minvals[c] = floor(minvals[c]);
		minvals[c] = ceil(minvals[c]);
	}
	
	#pragma omp parallel 
	{
		float *values = new float[layers];
		
		#pragma omp for
		for(long index=0; index<HW; index++)
		{
			int x = index / height;
			int y = index % height;
			bool ishole = false;

			for(int c=0; c<layers; c++)
			{
				values[c] = Signal[index + c*HW];
				ishole = ishole || _isnan(values[c]);
			}	

			// is no hole - copy input value
			if(!ishole)
			{
				for(int c=0; c<layers; c++)
				{
					Result[index + c*HW] = values[c];
				}
				continue;
			}

			float best_cost = 1000.f;			

			for(int c=0; c<layers; c++)
			{
				for(float value=minvals[c]; value<=minvals[c]; value++)
				{
					float cost = 0.f;
					float weight = 0.f;
					int pixels = 0;

					for(int xx=max(0, x-radius); xx<=min(width-1, x+radius); xx++)
					{
						for(int yy=max(0, y-radius); yy<=min(height-1, y+radius); yy++)
						{
							long indexx = Signal.Index(xx, yy);

							if(!_isnan(Signal.data[indexx + c*HW]))
							{

							}

							cost += 
							for()
						}
					}
				}

			}
			

			for(int i=0; i<window; i++)
			{
				Dmax = (blkDepth[i] > Dmax) ? blkDepth[i] : Dmax;
				Dmin = (blkDepth[i] < Dmin) ? blkDepth[i] : Dmin;
			}

			// read depth data
			readBlockFloat_zeropadded((float *)Depth, blkDepth, x, y, radius, width, height);
			readBlockFloat_zeropadded((float *)Confidence, blkConf, x, y, radius, width, height);
            
            
			//if(abs(Dmax - Dmin) < 0.01)//search_step)
			//{
			//    Filtered[index] = Dnow;
			//    continue;
			//}
			Dmin = floor(Dmin-1);
			Dmax = ceil(Dmax+1);
                        
			// prepare color weights
			calculateWeights_zeropadded((UINT8*)Image, weights, weights_table, radius, x, y, width, height);			           
			for(int i=0; i<window; i++)
			{
				weights[i] *= distance[i];
				weights[i] *= blkConf[i];
			}
			normalizeWeights(weights, radius);

			float Cost[255];
			int hypoMax = (int)ceil((Dmax-Dmin));

			float Cbest = MAXCost;
			float Dbest = Dnow; 
			int Hbest = -1;
            
			for(int hypo = 0; hypo <= hypoMax; hypo++)
			{ 
				float d = (float)Dmin + /*step**/hypo;
				float value = 0;
				for(int i=0; i<window; i++)
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
		
		}

		delete[] values;
	}

	delete[] minvals;

}