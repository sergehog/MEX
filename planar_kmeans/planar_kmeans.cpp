/** planar_kmeans
*	@file planar_kmeans.cpp
*	@date 07.04.2015
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

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")

//typedef unsigned char uint8;
using namespace mymex;


void mexFunction(int nout, mxArray* output[], int in, const mxArray* input[])
{
	omp_set_num_threads(std::max(2, omp_get_max_threads()));
	omp_set_dynamic(std::max(1, omp_get_max_threads() / 2));

	if (in != 2 || nout != 2 || mxGetClassID(input[0]) != mxSINGLE_CLASS)
	{
		mexErrMsgTxt("USAGE: [Segments, KMeans] = planar_kmeans(single(ABCIn), K);");
	}

	MexImage<float> ABCIn(input[0]);

	const int height = ABCIn.height;
	const int width = ABCIn.width;	
	const int layers = ABCIn.layers;
	const long HW = ABCIn.layer_size;
	const float nan = sqrt(-1.f);	
	const unsigned K = std::max(2u, (unsigned)mxGetScalar(input[1]));

	if (layers != 3)
	{
		mexErrMsgTxt("ABCIn must have exactly 3 components: ABC, where d = Ax + By + C");
	}
	const size_t dims[] = { (size_t)height, (size_t)width, 1 };
	const size_t dimK[] = { (size_t)K, (size_t)4, 1 };

	output[0] = mxCreateNumericArray(3, dims, mxUINT32_CLASS, mxREAL);
	output[1] = mxCreateNumericArray(3, dimK, mxSINGLE_CLASS, mxREAL);

	MexImage<unsigned> Filtered(output[0]);
	MexImage<float> Kmeans(output[1]);	
	
	int k = 0; 
	while (k < K)
	{
		long i = k*HW / K - 1;
		while (!isnan(ABCIn[i]))
		{
			i = i++%HW;
		}
		const int x = i / height;
		const int y = i % height;
				
		Kmeans(0, k) = ABCIn(x, y, 0);
		Kmeans(1, k) = ABCIn(x, y, 1);
		Kmeans(2, k) = ABCIn(x, y, 2);
		Kmeans(4, k) = 0;
	}
	
	for (int iteration = 0; iteration < 100; iteration++)
	{
		#pragma omp parallel for
		for (long i = 0; i < HW; i++)
		{
			if (isnan(ABCIn[i]))
				continue;
			const int x = i / height;
			const int y = i % height;
			float best_distance = FLT_MAX;
			int best_cluster = -1;
			for (k = 0; k < K; k++)
			{
				const float d_cluster = x*Kmeans(0, k) + y*Kmeans(1, k) + Kmeans(2, k);
				const float d_actual = x*ABCIn(x, y, 0) + y*ABCIn(x, y, 1) + ABCIn(x, y, 2);

				//float distance = (ABCIn(x, y, 0) - Kmeans(0, k)) * (ABCIn(x, y, 0) - Kmeans(0, k));
				//distance += (ABCIn(x, y, 1) - Kmeans(1, k)) * (ABCIn(x, y, 1) - Kmeans(1, k));
				//distance += (ABCIn(x, y, 2) - Kmeans(2, k)) * (ABCIn(x, y, 2) - Kmeans(2, k));
				
				//float distance = sqrt((d_cluster - d_actual)*(d_cluster - d_actual));
				const float distance = abs(d_cluster - d_actual);
				if (distance < best_distance)
				{
					best_distance = distance;
					best_cluster = k;
				}
			}
			//const float d = ABCIn();
		}
	}

}