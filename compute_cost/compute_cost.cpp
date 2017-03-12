/** Computes 3D Cost Volume for 2 (rectified) images. Uses SAD dissimilarity metric
* @author Sergey Smirnov
*/
#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}
#endif // __cplusplus
//#include "../common/common.h"
#include "../common/meximage.h"
#include <algorithm>
#ifndef _NDEBUG
#include <omp.h>
#endif

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")
#pragma warning(disable:4244)
#pragma warning(disable:4018)

using namespace mymex;

//class base_matcher
//{
//
//};

template<typename T, int colors>
void compute_in_parallel(MexImage<T> &Left, MexImage<T> &Right, MexImage<float> &CostL, MexImage<float> &CostR, const int mindisp, const int maxdisp, const float cost_threshold, const int y_offset)
{
	#pragma omp parallel sections
	{
		#pragma omp section
		{
			compute_cost<T,colors>(Left, Right, CostL, mindisp, maxdisp, 1, cost_threshold, y_offset);
		}
		#pragma omp section
		{
			compute_cost<T,colors>(Right, Left, CostR, mindisp, maxdisp, -1, cost_threshold, y_offset);
		}
	}
}

template<typename T>
void compute_costs(int colors, MexImage<T> &Left, MexImage<T> &Right, MexImage<float> &CostL, MexImage<float> &CostR, const int mindisp, const int maxdisp, const float cost_threshold, const int y_offset)
{
	if (colors == 1)
	{
		compute_in_parallel<T, 1>(Left, Right, CostL, CostR, mindisp, maxdisp, cost_threshold, y_offset);
	}
	else if (colors == 2)
	{
		compute_in_parallel<T, 2>(Left, Right, CostL, CostR, mindisp, maxdisp, cost_threshold, y_offset);
	}
	else if (colors == 3)
	{
		compute_in_parallel<T, 3>(Left, Right, CostL, CostR, mindisp, maxdisp, cost_threshold, y_offset);
	}
	else if (colors == 4)
	{
		compute_in_parallel<T, 4>(Left, Right, CostL, CostR, mindisp, maxdisp, cost_threshold, y_offset);
	}
	else if (colors == 5)
	{
		compute_in_parallel<T, 5>(Left, Right, CostL, CostR, mindisp, maxdisp, cost_threshold, y_offset);
	}
	else if (colors == 6)
	{
		compute_in_parallel<T, 6>(Left, Right, CostL, CostR, mindisp, maxdisp, cost_threshold, y_offset);
	}
	else if (colors == 7)
	{
		compute_in_parallel<T, 7>(Left, Right, CostL, CostR, mindisp, maxdisp, cost_threshold, y_offset);
	}
	else if (colors == 8)
	{
		compute_in_parallel<T, 8>(Left, Right, CostL, CostR, mindisp, maxdisp, cost_threshold, y_offset);
	}
	else if (colors == 9)
	{
		compute_in_parallel<T, 9>(Left, Right, CostL, CostR, mindisp, maxdisp, cost_threshold, y_offset);
	}
}


template<typename T, int colors>
void compute_cost(MexImage<T> &Ref, MexImage<T> &Tmpl, MexImage<float> &Cost, const int mindisp, const int maxdisp, const int step, const float cost_threshold, const int y_offset)
{
	const int height = Ref.height;
	const int width = Ref.width;
	const long HW = width*height;
	const int dispLayers = static_cast<size_t>(std::max(maxdisp - mindisp + 1, 1));
	const float nan = sqrt(-1);

	Cost.setval(cost_threshold);
	
	#pragma omp parallel for
	for (int d = 0; d<dispLayers; d++)
	{
		//#pragma omp parallel for    
		for(long i=0; i<HW; i++)
		{
			const int y = i % height;
			const int x = i / height;
				
			int xT = (step > 0) ? x - mindisp - d : x + mindisp + d;
			xT = (xT < 0) ? 0 : (xT >= width ? width-1 : xT);			
			
			float min_difference = nan;
			for(int yi = y-y_offset; yi<=y+y_offset; yi++)
			{
				float difference = 0;

				for(int c=0; c<colors; c++)
				{
					difference += abs(Ref(x,y,c)-Tmpl(xT,yi,c)); 
				}

				min_difference = _isnan(min_difference) ? difference : (difference < min_difference ? difference : min_difference);
			}
					            
			Cost(x, y, d) = std::min(min_difference, cost_threshold);            
		}   
	}
}

void mexFunction (int nout, mxArray* output[], int in, const mxArray* input[])
{
	#ifndef _NDEBUG
	//omp_set_num_threads(std::max<int>(8, omp_get_max_threads()));
	//omp_set_dynamic(std::max<int>(4, omp_get_max_threads() / 2));
	#endif
	omp_set_num_threads(omp_get_max_threads());
	omp_set_dynamic(omp_get_max_threads());

	if(in < 3 || in > 6 || nout != 2)
	{
		mexErrMsgTxt("USAGE: [CostL, CostR] = compute_cost(Left, Right, maxdisp, [mindisp = 0, cost_threshold, y_offset]);"); 
    } 

	const int maxdisp = static_cast<int>(mxGetScalar(input[2]));	
	const int mindisp = (in > 3) ?  static_cast<int>(mxGetScalar(input[3])) : 0;	
	const float cost_threshold = (in > 4) ?  static_cast<float>(mxGetScalar(input[4])) : 1000.f;	
	const int y_offset = (in > 5) ?  static_cast<int>(mxGetScalar(input[5])) : 0;	
	if(maxdisp <= mindisp)
	{
		mexErrMsgTxt("ERROR: 'maxdisp' must be larger than 'mindisp'!"); 
	}	

	const mxClassID type0 = mxGetClassID(input[0]);
	const mxClassID type1 = mxGetClassID(input[1]);
	const int width = mxGetWidth(input[0]);
	const int height = mxGetHeight(input[0]);
	const int colors = mxGetLayers(input[0]);

	if(height != mxGetHeight(input[1]) || width != mxGetWidth(input[1]) || colors != mxGetLayers(input[1]))
	{
		mexErrMsgTxt("ERROR: Sizes of parameters 'left', 'right' must be the same."); 
	}	

	if(colors < 1 || colors > 9)
	{
		mexErrMsgTxt("Too many colors in your images."); 
	}

	const size_t dispLayers = static_cast<size_t>(std::max(maxdisp - mindisp + 1, 1));
	const mwSize depthcost[] = {(size_t)height, (size_t)width, dispLayers};

	const bool bothFloat = (type0 == mxSINGLE_CLASS && type1 == mxSINGLE_CLASS);
	const bool bothByte = (type0 == mxUINT8_CLASS && type1 == mxUINT8_CLASS);
	if(!bothFloat && !bothByte) 
	{
		mexErrMsgTxt("Both input images must be either floats or uint8."); 
	}

	//Matlab-allocated variables
	output[0] = mxCreateNumericArray(3, depthcost, mxSINGLE_CLASS, mxREAL);
	output[1] = mxCreateNumericArray(3, depthcost, mxSINGLE_CLASS, mxREAL);	
	MexImage<float> CostL(output[0]);
	MexImage<float> CostR(output[1]);
	//CostL.setval(nan);
	//CostR.setval(nan);

	if(type0 == mxSINGLE_CLASS && type1 == mxSINGLE_CLASS)
	{
		MexImage<float> Left(input[0]);
		MexImage<float> Right(input[1]);
		compute_costs<float>(colors, Left, Right, CostL, CostR, mindisp, maxdisp, cost_threshold, y_offset);
	}
	else if(type0 == mxUINT8_CLASS && type1 == mxUINT8_CLASS)
	{
		MexImage<unsigned char> Left(input[0]);
		MexImage<unsigned char> Right(input[1]);
		compute_costs<unsigned char>(colors, Left, Right, CostL, CostR, mindisp, maxdisp, cost_threshold, y_offset);
	}

    
	
}