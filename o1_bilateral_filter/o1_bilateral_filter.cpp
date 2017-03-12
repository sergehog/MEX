/** O(1) bilateral filtering / aggregation
*	Updated version of fast_bilateral.  Uses new MexImage (2) class 
*	@file o1_bilateral_filter.cpp 
*	@date 13.04.2012
*	@author Sergey Smirnov <sergey.smirnov@tut.fi>
*/


#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}
#endif // __cplusplus

#include "../common/meximage.h"
//#include "../common/defines.h"
//#include "../common/matching.h"

#include <cmath>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <omp.h>

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")

typedef unsigned char uint8;
using namespace mymex;

void init_color_table(float *weights_table, float sigma_color, float range_max, float range_step)
{
	int i=0;
    for(float z=0; z<range_max; z+=range_step, i++)
    {
        weights_table[i] = exp(-((float)z)/sigma_color);
    }
}

void mexFunction (int nout, mxArray* output[], int in, const mxArray* input[])
{	
	omp_set_num_threads(std::max(2,omp_get_num_threads()));
	omp_set_dynamic(0);

	if(nout != 1 || in < 3 || in > 5 || mxGetClassID(input[0])!=mxUINT8_CLASS || mxGetClassID(input[1])!=mxSINGLE_CLASS)
	{
		mexErrMsgTxt("USAGE: Output = fast_bilateral(UINT8(Edges), single(Input), radius, <sigma_color , pbfics_num>);\n"); 
    } 

	MexImage<uint8> Edges(input[0]);
	MexImage<float> Input(input[1]);
	const int width = Edges.width;
	const int height = Edges.height;
	const int colors = Edges.layers;
	const long HW = width*height;

	if(Input.width != width || Input.height != height)
	{
		mexErrMsgTxt("ERROR: width and height of 'Input' and 'Edges' images must coincide!\n");
	}

	//if(colors != 1)
	//{
	//	mexErrMsgTxt("ERROR: fast_bilateral works with gray-scale 'Edge' images only!\n");
	//}
	
	const int layers = Input.layers;
	const int radius  = std::max(0, (int)mxGetScalar(input[2]));
	float const sigma_color = (in > 3) ? std::max(0.1f, (float)mxGetScalar(input[3])) : 30.f;	
	const int range_max = 255;
	const int PBFICs = (in > 4) ? std::min(range_max, std::max(4, (int)mxGetScalar(input[4]))) : 4;	
	const float PBFIC_step = (float)range_max/(PBFICs-1);

	//float *weights_table = new float [PBFICs + 1];
	//init_color_table(weights_table, sigma_color, range_max, PBFIC_step);	
	float *weights_table = new float [range_max];
	init_color_table(weights_table, sigma_color, range_max, 1);	
	
	size_t dimms[] = {height, width, layers};
	output[0] = mxCreateNumericArray(3, dimms, mxSINGLE_CLASS, mxREAL);
	MexImage<float> Output(output[0]);

	//float *Jk = new float[HW*layers*(unsigned long)colors];
	//float *Wk = new float[HW*colors];

	MexImage<float> Wk(width, height); //color-dependent part of weights
	MexImage<float> Jk(width, height, layers);  //content-dependent part of weights

	for(int c=0; c<colors; c++)
	{
		const long cHW = c*HW;
		// ! calculate PBFICs..
		int pbfic = 0; // index of current PBFIC, k = value of current PBFIC
		for(float k=0; k<=range_max; k+=PBFIC_step, pbfic++)
		{
			// calculate Wk and Jk
			#pragma omp parallel for
			for(int index=0; index<HW; index++)
			{
				int y = index % height;
				int x = index / height;
				
				int i = std::abs((float)Edges[index+cHW] - k);
				float w = weights_table[i];
				//float i = std::abs((float)Edges[index+cHW] - k);
				//float w = exp(-i/sigma_color);
				
				Wk.data[index] = w;
				for(int l=0; l<layers; l++)
				{
					Jk.data[index+l*HW] = w * (float)Input[index+l*HW];
				}
			}

			//integrate them
			Wk.IntegralImage(true);
			Jk.IntegralImage(true);

			// calculate approximate bilateral 
			#pragma omp parallel
			{
				float *Jkbs = new float[layers];
				#pragma omp for
				for(int index=0; index<HW; index++)
				{
					int y = index % height;
					int x = index / height;

					float k_next = k + PBFIC_step ;
					float k_previous = k - PBFIC_step;
					float image_value = (float)Edges[index+cHW];
					bool equal = (image_value == k);
					bool next = (image_value > k && image_value < k_next);
					bool previous = (image_value < k && image_value > k_previous);
					
					if(equal || next || previous)
					{		
						float factor = equal ? 1 : std::abs(image_value-k)/PBFIC_step;
						
						float Wkb = Wk.getIntegralAverage(x, y, radius);						
						Jk.getIntegralAverage(x, y, radius, Jkbs);

						for(int l=0; l<layers; l++)
						{
							float Jkb = Jkbs[l];
							Jkb /= Wkb;
							//if(colors > 1)
							//{
							//	Output.data[index+l*HW] += log(factor*Jkb);
							//}
							//else
							//{
							Output.data[index+l*HW] += factor*Jkb/colors;
							//}
						}
					}
				}
				delete[] Jkbs;
			}
		}

	}

	delete[] weights_table;

	//if(colors > 1)
	//{
	//	for(int index = 0; index < HW*layers; index ++)
	//	{
	//		Output.data[index] = (float)pow(exp(Output.data[index]), 1/3)/3;
	//	}
	//}

}
