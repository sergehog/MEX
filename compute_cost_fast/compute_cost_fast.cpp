/** Computes 3D Cost Volume for left channel. low memory footprint 
* @date 25.04.2016
* @author Sergey Smirnov sergey.smirnov@tut.fi
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


template<int colors>
void compute_cost(MexImage<float> &Left, MexImage<float> &Right, MexImage<float> &Cost, const int mindisp, const int maxdisp, const float cost_threshold, const int y_offset)
{
	const int height = Left.height;
	const int width = Left.width;
	const long HW = width*height;
	const int dispLayers = std::max(maxdisp - mindisp + 1, 1);
	const float nan = sqrt(-1);

	Cost.setval(cost_threshold);

	#pragma omp parallel for
	for (int d = 0; d<dispLayers; d++)
	{
		//#pragma omp parallel for    
		for (long i = 0; i<HW; i++)
		{
			const int y = i % height;
			const int x = i / height;

			const int _xT = x - mindisp - d;
			const int xT = (_xT < 0) ? 0 : (_xT >= width ? width - 1 : _xT);

			float min_difference = cost_threshold;
			for (int yi = y - y_offset; yi <= y + y_offset; yi++)
			{
				float difference = 0;

				for (int c = 0; c<colors; c++)
				{
					float diff = (Left(x, y, c) - Right(xT, yi, c));					
					difference += diff*diff;
				}
				difference = sqrt(difference);
				min_difference = difference < min_difference ? difference : min_difference;
			}

			Cost(x, y, d) = std::min(min_difference, cost_threshold);
		}
	}
}

void mexFunction(int nout, mxArray* output[], int in, const mxArray* input[])
{
#ifndef _NDEBUG
	//omp_set_num_threads(std::max<int>(8, omp_get_max_threads()));
	//omp_set_dynamic(std::max<int>(8, omp_get_max_threads() / 2));
	omp_set_num_threads(omp_get_max_threads());
	omp_set_dynamic(omp_get_max_threads());
#endif
	

	if (in < 3 || in > 6 || nout != 1 || mxGetClassID(input[0]) != mxSINGLE_CLASS || mxGetClassID(input[1]) != mxSINGLE_CLASS)
	{
		mexErrMsgTxt("USAGE: [CostL] = compute_cost_fast(Left, Right, maxdisp, [mindisp = 0, cost_threshold, y_offset]);");
	}

	const int maxdisp = static_cast<int>(mxGetScalar(input[2]));
	const int mindisp = (in > 3) ? static_cast<int>(mxGetScalar(input[3])) : 0;
	const float cost_threshold = (in > 4) ? static_cast<float>(mxGetScalar(input[4])) : 1000.f;
	const int y_offset = (in > 5) ? static_cast<int>(mxGetScalar(input[5])) : 0;
	if (maxdisp <= mindisp)
	{
		mexErrMsgTxt("ERROR: 'maxdisp' must be larger than 'mindisp'!");
	}
	MexImage<float> Left(input[0]);
	MexImage<float> Right(input[1]);
	const int width = Left.width;
	const int height = Left.height;
	const int colors = Left.layers;
	if (Right.width != width || Right.height != height || Right.layers != colors)
	{
		mexErrMsgTxt("Input images must have same resolution!");
	}
	if (colors < 1 || colors > 9)
	{
		mexErrMsgTxt("Too many colors in your images.");
	}
	const size_t dispLayers = static_cast<size_t>(std::max(maxdisp - mindisp + 1, 1));
	const mwSize depthcost[] = { (size_t)height, (size_t)width, dispLayers };
	
	//Matlab-allocated variables
	output[0] = mxCreateNumericArray(3, depthcost, mxSINGLE_CLASS, mxREAL);	
	MexImage<float> CostL(output[0]);
	
	switch (colors)
	{
	case 2: compute_cost<2>(Left, Right, CostL, mindisp, maxdisp, cost_threshold, y_offset); break;
	case 3: compute_cost<3>(Left, Right, CostL, mindisp, maxdisp, cost_threshold, y_offset); break;
	case 4: compute_cost<4>(Left, Right, CostL, mindisp, maxdisp, cost_threshold, y_offset); break;
	case 5: compute_cost<5>(Left, Right, CostL, mindisp, maxdisp, cost_threshold, y_offset); break;
	case 6: compute_cost<6>(Left, Right, CostL, mindisp, maxdisp, cost_threshold, y_offset); break;
	case 7: compute_cost<7>(Left, Right, CostL, mindisp, maxdisp, cost_threshold, y_offset); break;
	case 8: compute_cost<8>(Left, Right, CostL, mindisp, maxdisp, cost_threshold, y_offset); break;
	case 9: compute_cost<9>(Left, Right, CostL, mindisp, maxdisp, cost_threshold, y_offset); break;
	default: compute_cost<1>(Left, Right, CostL, mindisp, maxdisp, cost_threshold, y_offset); break;
	}

}