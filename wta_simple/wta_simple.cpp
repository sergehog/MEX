/** wta_simple - Winner takes All with simple interfacing
*	@file wta_simple.cpp
*	@date 28.01.2011
*	@author Sergey Smirnov
*/

#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}
#endif // __cplusplus
#include "../common/meximage.h"
#include <utility>
//#include "../common/defines.h"
//#include "../common/matching.h"
using namespace mymex;

#define GLM_FORCE_CXX11  
#include <glm/glm.hpp>

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")

template<typename T>
void wta_simple(MexImage<T> &Cost, MexImage<int> &Disparity, MexImage<float> &Confidence, const int mindisp)
{
	const int width = Cost.width;
	const int height = Cost.height;	
	const int layers = Cost.layers;
	const long HW = width*height;

	#pragma omp parallel for
	for (long i = 0; i<HW; i++)
	{
		const int x = i / height;
		const int y = i % height;
		T best_value = Cost(x, y, 0);
		T second_best = Cost(x, y, 0);
		Disparity(x, y) = mindisp;		
		for (int l = 1; l<layers; l++)
		{
			if (Cost(x, y, l) < best_value)
			{
				Disparity(x, y) = mindisp + l;
				second_best = best_value;
				best_value = Cost(x, y, l);
			}
			else if (Cost(x, y, l) < second_best)
			{
				second_best = Cost(x, y, l);
			}
		}
		Confidence(x, y) = float(second_best) > 0 ? (second_best - best_value) / float(second_best) : 0.f;
	}

}

template<typename T>
void wta_interpolate(MexImage<T> &Cost, MexImage<float> &Disparity, MexImage<float> &Confidence, const int mindisp)
{
	const int width = Cost.width;
	const int height = Cost.height;
	const int layers = Cost.layers;
	const long HW = width*height;

	#pragma omp parallel for
	for (long i = 0; i<HW; i++)
	{
		const int x = i / height;
		const int y = i % height;
		int best = 0, second_best = 0;
		T best_value = Cost(x, y, 0);
		T second_value = Cost(x, y, 0);
		T third_value = Cost(x, y, 0);
		
		for (int l = 1; l<layers; l++)
		{
			if (Cost(x, y, l) < best_value)
			{
				second_best = best;
				best = l;
				third_value = second_value;
				second_value = best_value;
				best_value = Cost(x, y, l);
			}
			else if (Cost(x, y, l) < second_best)
			{
				second_best = l;
				third_value = second_value;
				second_value = Cost(x, y, l);
			}
			else if (Cost(x, y, l) < third_value)
			{
				third_value = Cost(x, y, l);
			}
		}

		if (/*abs(best - second_best) == 1 &&*/ best > 0 && best < layers-1)
		{
			T cost_prev = Cost(x, y, best - 1);
			T cost_next = Cost(x, y, best + 1);
			T cost_optimal = (cost_next - float(cost_prev)) / (2 * (cost_next + cost_prev - 2 * float(best_value)));
			Disparity(x, y) = float(best) - float(cost_optimal);
			Disparity(x, y) = abs(Disparity(x, y) - best) >= 1 ? float(best) : Disparity(x, y);
			Disparity(x, y) += mindisp;
			
			//Confidence(x, y) = float(second_value) > 0 ? (second_value - best_value) / float(second_value) : 0.f;
			glm::vec3 abc = glm::mat3(0.5f, -0.5f, 0.f, -1.f, 0.f, 1.f, 0.5f, 0.5f, 0.f) * glm::vec3(cost_prev, best_value, cost_next);
			const float new_best = abc[2] - abc[1] * abc[1] / (4 * abc[0]);
			//Confidence(x, y) = float(second_value) > 0 ? (second_value - new_best) / float(second_value) : 0.f;
			Confidence(x, y) = float(best_value) > 0 ? (third_value - best_value) / float(third_value) : 0.f;
			
		}
		else
		{
			Disparity(x, y) = mindisp + float(best);
			Confidence(x, y) = float(second_value) > 0 ? (second_value - best_value) / float(second_value) : 0.f;
		}

		
	}
}

template<typename T>
void wta_interpolate2(MexImage<T> &Cost, MexImage<float> &Disparity, MexImage<float> &Confidence, const int mindisp)
{
	const int width = Cost.width;
	const int height = Cost.height;
	const int layers = Cost.layers;
	const long HW = width*height;

#pragma omp parallel for
	for (long i = 0; i < HW; i++)
	{
		const int x = i / height;
		const int y = i % height;
		int best = 0, second_best = 0;
		T best_value = Cost(x, y, 0);
		T second_value = Cost(x, y, 0);
		T third_value = Cost(x, y, 0);

		for (int l = 1; l < layers; l++)
		{
			if (Cost(x, y, l) < best_value)
			{
				second_best = best;
				best = l;
				third_value = second_value;
				second_value = best_value;
				best_value = Cost(x, y, l);
			}
			else if (Cost(x, y, l) < second_best)
			{
				second_best = l;
				third_value = second_value;
				second_value = Cost(x, y, l);
			}
			else if (Cost(x, y, l) < third_value)
			{
				third_value = Cost(x, y, l);
			}
		}

		if (/*abs(best - second_best) == 1 &&*/ best > 0 && best < layers - 1)
		{
			T cost_prev = Cost(x, y, best - 1);
			T cost_next = Cost(x, y, best + 1);
			T cost_optimal = (cost_next - float(cost_prev)) / (2 * (cost_next + cost_prev - 2 * float(best_value)));
			Disparity(x, y) = float(best) - float(cost_optimal);
			Disparity(x, y) = abs(Disparity(x, y) - best) >= 1 ? float(best) : Disparity(x, y);
			Disparity(x, y) += mindisp;
			Confidence(x, y) = float(best_value);

		}
		else
		{
			Disparity(x, y) = mindisp + float(best);
			Confidence(x, y) = float(best_value);
		}

	}
}

void mexFunction (int nout, mxArray* output[], int in, const mxArray* input[])
{
	if (nout < 1 || nout > 2 || in < 1 || in > 3)
	{
		mexErrMsgTxt("USAGE: [Disp, <Conf>] = wta_simple(Cost, <mindisp_to_be_added, interpolate>);"); 
    } 

	const int mindisp = (in > 1) ? (int) mxGetScalar(input[1]) : 0;
	const int interpolate_or_more = (in > 2) ? (int)mxGetScalar(input[2]) : false;

	if (mxGetClassID(input[0]) == mxUINT8_CLASS)
	{
		MexImage<unsigned char> Cost(input[0]);
		const mwSize depthdims[] = {Cost.height, Cost.width, 1};

		MexImage<float> *Confidence;
		if (nout > 1)
		{
			output[1] = mxCreateNumericArray(3, depthdims, mxSINGLE_CLASS, mxREAL);
			Confidence = new MexImage<float>(output[1]);
		}
		else
		{
			Confidence = new MexImage<float>(Cost.width, Cost.height);
		}
		
		if (interpolate_or_more)
		{
			output[0] = mxCreateNumericArray(3, depthdims, mxSINGLE_CLASS, mxREAL);
			MexImage<float> Disparity(output[0]);

			wta_interpolate<unsigned char>(Cost, Disparity, *Confidence, mindisp);
		}
		else
		{
			output[0] = mxCreateNumericArray(3, depthdims, mxINT32_CLASS, mxREAL);
			MexImage<int> Disparity(output[0]);

			wta_simple<unsigned char>(Cost, Disparity, *Confidence, mindisp);
		}

		delete Confidence;

	}
	else if (mxGetClassID(input[0]) == mxSINGLE_CLASS)
	{
		MexImage<float> Cost(input[0]);
		const mwSize depthdims[] = { Cost.height, Cost.width, 1 };

		MexImage<float> *Confidence;
		if (nout > 1)
		{
			output[1] = mxCreateNumericArray(3, depthdims, mxSINGLE_CLASS, mxREAL);
			Confidence = new MexImage<float>(output[1]);
		}
		else
		{
			Confidence = new MexImage<float>(Cost.width, Cost.height);
		}

		if (interpolate_or_more == 1)
		{
			output[0] = mxCreateNumericArray(3, depthdims, mxSINGLE_CLASS, mxREAL);
			MexImage<float> Disparity(output[0]);

			wta_interpolate<float>(Cost, Disparity, *Confidence, mindisp);
		}
		else if (interpolate_or_more == 2)
		{
			output[0] = mxCreateNumericArray(3, depthdims, mxSINGLE_CLASS, mxREAL);
			MexImage<float> Disparity(output[0]);

			wta_interpolate2<float>(Cost, Disparity, *Confidence, mindisp);
		}
		else
		{
			output[0] = mxCreateNumericArray(3, depthdims, mxINT32_CLASS, mxREAL);
			MexImage<int> Disparity(output[0]);

			wta_simple<float>(Cost, Disparity, *Confidence, mindisp);
		}

		delete Confidence;
	}
	else 
	{
		mexErrMsgTxt("Don't support this datatype!");
	}

}