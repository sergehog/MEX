/** Slanted PatchMatch with cost optimization
* @file patchmatch_slanted_propagation
* @date 16.05.2016
* @author Sergey Smirnov
*/
#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}
#endif // __cplusplus
//#include "../common/common.h"
#include "../common/meximage.h"
#include <cstdint>
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


template<int radius>
void prepare_weights(const MexImage<const float> &Guide, const MexImage<float> &Weights, float const sigma)
{
	const int width = Guide.width;
	const int height = Guide.height;
	const int colors = Guide.layers;
	const int64_t HW = Guide.layer_size;
	const int diameter = radius * 2 + 1;
	const int window = (radius * 2 + 1) * (radius * 2 + 1);

#pragma omp parallel for
	for (int64_t i = 0; i < HW; i++)
	{
		const int x = i / height;
		const int y = i % height;
		for (int w = 0; w < window; w++)
		{
			const int dx = w / diameter - radius;
			const int dy = w % diameter - radius;

			const int xx = x + dx;
			const int yy = y + dy;

			if (xx < 0 || xx >= width || yy < 0 || yy >= height)
			{
				continue;
			}

			float diff = 0;
			for (int c = 0; c < colors; c++)
			{
				diff += (Guide(x, y, c) - Guide(xx, yy, c)) * (Guide(x, y, c) - Guide(xx, yy, c));
			}

			Weights(x, y, w) = exp(-sqrt(diff) / sigma);
		}

	}
}

template<int radius>
void optimized_patchmatch(const MexImage<const float> & Left, const MexImage<const float> & Right, const MexImage<const float> &LeftGuide, const MexImage<const float> &RightGuide, const MexImage<float> &PlanesL, const MexImage<float> &PlanesR, const MexImage<float> & BestErrorL, const MexImage<float> & BestErrorR, const int mindisp, const int maxdisp, const float std_ab, const float std_c, const int iterations, const float forget_rate, const float sigma)
{
	const int width = Left.width;
	const int height = Left.height;
	const int diameter = radius * 2 + 1;
	const int window = (radius * 2 + 1) * (radius * 2 + 1);
	const MexImage<float> WeightsL(width, height, window);
	const MexImage<float> WeightsR(width, height, window);
#pragma omp parallel sections
	{
#pragma omp section
		{
			WeightsL.setval(0.f);
			prepare_weights<radius>(LeftGuide, WeightsL, sigma);
		}
#pragma omp section
		{
			WeightsR.setval(0.f);
			prepare_weights<radius>(RightGuide, WeightsR, sigma);
		}
	}

#pragma omp parallel sections
	{
#pragma omp section
		{
			single_patchmatch<radius>(false, Left, Right, WeightsL, PlanesL, BestErrorL, mindisp, maxdisp, std_ab, std_c, iterations, forget_rate);
		}
#pragma omp section
		{
			single_patchmatch<radius>(true, Right, Left, WeightsR, PlanesR, BestErrorR, mindisp, maxdisp, std_ab, std_c, iterations, forget_rate);
		}
	}
}

template<int radius>
void single_patchmatch(const bool direction, const MexImage<const float> &Reference, const MexImage<const float> &Template, const MexImage<float> &Weights, const MexImage<float> &Planes, const MexImage<float> & BestError, const int mindisp, const int maxdisp, const float ab_std, const float c_std, const float iterations, const float forget_rate)
{
	const int width = Reference.width;
	const int height = Reference.height;
	const int colors = Reference.layers;
	const int64_t HW = static_cast<int64_t>(Reference.layer_size);

	const int diameter = radius * 2 + 1;
	const int window = diameter*diameter;
	const int dir = direction ? +1 : -1;
	float ab_std_curr = ab_std;
	float c_std_curr = c_std;
	//mexPrintf("Random Init \n");
#pragma omp parallel for
	//random initialization
	for (int64_t i = 0; i < HW; i++)
	{
		const int x = i / height;
		const int y = i % height;
		const float d = Planes(x, y, 2) + c_std_curr  * (1 - 2 * float(rand()) / (RAND_MAX));
		const float nx = (1 - 2 * float(rand()) / (RAND_MAX));
		const float ny = (1 - 2 * float(rand()) / (RAND_MAX));
		const float nz = float(rand()) / (RAND_MAX)+0.0001;
		const float norm = sqrt(nx*nx + ny*ny + nz*nz);
		const float a = -nx / nz;
		const float b = -ny / nz;
		const float c = (nx*x + ny*y + nz*d) / nz;
		Planes(x, y, 0) = a;
		Planes(x, y, 1) = b;
		Planes(x, y, 2) = c;
		BestError(x, y) = check_error<radius>(direction, Reference, Template, Weights, x, y, a, b, c);
	}

	float normal_std = 1;
	for (int iter = 0; iter < iterations; iter++)
	{
		//mexPrintf("Iteration \n");
		//#pragma omp parallel for
		//for (int64_t i = 0; i < HW; i++)
		//{
		//	const int x = i / height;
		//	const int y = i % height;

		//ab_std_curr /= 2;
		//c_std_curr /= 2;

		//	// first attempt - try to change disparity level, keep orientation
		//	const float a1 = Planes(x, y, 0);
		//	const float b1 = Planes(x, y, 1);
		//	const float curr_d1 = a1*x + b1*y + Planes(x, y, 2);
		//	float new_d = curr_d1 + c_std_curr  * (1 - 2 * float(rand()) / (RAND_MAX));
		//	new_d = (new_d < mindisp) ? mindisp : new_d;
		//	new_d = (new_d > maxdisp) ? maxdisp : new_d;			
		//	const float c1 = new_d - a1*x - b1*y;

		//	float cost_value = check_error<radius>(direction, Reference, Template, Weights, x, y, a1, b1, c1);
		//	if (cost_value < BestError(x, y))
		//	{
		//		BestError(x, y) = cost_value;
		//		Planes(x, y, 0) = a1;
		//		Planes(x, y, 1) = b1;
		//		Planes(x, y, 2) = c1;
		//	}
		//	
		//}
		// left-to-right pass
#pragma omp parallel for
		for (int y = 0; y < height; y++)
		{
			float previous_error = BestError(0, y);
			int previous_pixels = 1;
			for (int x = 1; x < width; x++)
			{
				const float a = Planes(x - 1, y, 0);
				const float b = Planes(x - 1, y, 1);
				const float c = Planes(x - 1, y, 2);

				float d = a*x + b*y + c;
				if (d < mindisp || d > maxdisp)
				{
					continue;
				}
				const float cost_value = check_error<radius>(direction, Reference, Template, Weights, x, y, a, b, c);
				const float cumulative_error = (previous_error * previous_pixels + cost_value) / (previous_pixels + 1);

				if (cumulative_error < BestError(x, y))
				{
					BestError(x, y) = cumulative_error;
					previous_pixels++;
					previous_error = cumulative_error;
					Planes(x, y, 0) = a;
					Planes(x, y, 1) = b;
					Planes(x, y, 2) = c;
				}
				else
				{
					previous_pixels = 1;
					previous_error = BestError(x, y);
				}
			}
		}

		// right-to-left pass
#pragma omp parallel for
		for (int y = 0; y < height; y++)
		{
			float previous_error = BestError(width-1, y);
			int previous_pixels = 1;
			for (int x = width - 2; x >= 0; x--)
			{
				const float a = Planes(x + 1, y, 0);
				const float b = Planes(x + 1, y, 1);
				const float c = Planes(x + 1, y, 2);
				float d = a*x + b*y + c;
				if (d < mindisp || d > maxdisp)
				{
					continue;
				}
				const float cost_value = check_error<radius>(direction, Reference, Template, Weights, x, y, a, b, c);
				const float cumulative_error = (previous_error * previous_pixels + cost_value) / (previous_pixels + 1);

				if (cumulative_error < BestError(x, y))
				{
					BestError(x, y) = cumulative_error;
					previous_pixels++;
					previous_error = cumulative_error;
					Planes(x, y, 0) = a;
					Planes(x, y, 1) = b;
					Planes(x, y, 2) = c;
				}
				else
				{
					previous_pixels = 1;
					previous_error = BestError(x, y);
				}
			}
		}

		// up-to-bottom pass
#pragma omp parallel for
		for (int x = 0; x < width; x++)
		{
			float previous_error = BestError(x, 0);
			int previous_pixels = 1;
			for (int y = 1; y < height; y++)
			{
				const float a = Planes(x, y - 1, 0);
				const float b = Planes(x, y - 1, 1);
				const float c = Planes(x, y - 1, 2);
				float d = a*x + b*y + c;
				if (d < mindisp || d > maxdisp)
				{
					continue;
				}
				const float cost_value = check_error<radius>(direction, Reference, Template, Weights, x, y, a, b, c);
				const float cumulative_error = (previous_error * previous_pixels + cost_value) / (previous_pixels + 1);

				if (cumulative_error < BestError(x, y))
				{
					BestError(x, y) = cumulative_error;
					previous_pixels++;
					previous_error = cumulative_error;
					Planes(x, y, 0) = a;
					Planes(x, y, 1) = b;
					Planes(x, y, 2) = c;
				}
				else
				{
					previous_pixels = 1;
					previous_error = BestError(x, y);
				}
			}
		}

		// bottom-to-up pass
#pragma omp parallel for
		for (int x = 0; x < width; x++)
		{
			float previous_error = BestError(x, height-1);
			int previous_pixels = 1;

			for (int y = height - 2; y >= 0; y--)
			{
				const float a = Planes(x, y + 1, 0);
				const float b = Planes(x, y + 1, 1);
				const float c = Planes(x, y + 1, 2);
				float d = a*x + b*y + c;
				if (d < mindisp || d > maxdisp)
				{
					continue;
				}
				const float cost_value = check_error<radius>(direction, Reference, Template, Weights, x, y, a, b, c);
				const float cumulative_error = (previous_error * previous_pixels + cost_value) / (previous_pixels + 1);

				if (cumulative_error < BestError(x, y))
				{
					BestError(x, y) = cumulative_error;
					previous_pixels++;
					previous_error = cumulative_error;
					Planes(x, y, 0) = a;
					Planes(x, y, 1) = b;
					Planes(x, y, 2) = c;
				}
				else
				{
					previous_pixels = 1;
					previous_error = BestError(x, y);
				}
			}
		}

		// refinement
#pragma omp parallel for
		for (int64_t i = 0; i < HW; i++)
		{
			const int x = i / height;
			const int y = i % height;
			const float a = Planes(x, y, 0);// +ab_std_curr * (1 - 2 * float(rand()) / (RAND_MAX));
			const float b = Planes(x, y, 1);// +ab_std_curr * (1 - 2 * float(rand()) / (RAND_MAX));
			const float c = Planes(x, y, 2);//  curr_d2 - a2*x - b2*y;
			const float d = (a*x + b*y + c);
			float nz = sqrt(1.f / (1 + a*a + b*b));
			float nx = -a * nz;
			float ny = -b * nz;
			nx += normal_std * (1 - 2 * float(rand()) / (RAND_MAX));
			ny += normal_std * (1 - 2 * float(rand()) / (RAND_MAX));
			nz += normal_std * (1 - 2 * float(rand()) / (RAND_MAX));
			nz = (nz < 0.001) ? 0.001 : nz;
			const float d2 = d + c_std_curr * (1 - 2 * float(rand()) / (RAND_MAX));
			const float norm = sqrt(nx*nx + ny*ny + nz*nz);

			const float a2 = -nx / nz;
			const float b2 = -ny / nz;
			const float c2 = (nx*x + ny*y + nz*d2) / nz;

			const float cost_value = check_error<radius>(direction, Reference, Template, Weights, x, y, a2, b2, c2);

			if (cost_value < BestError(x, y))
			{
				BestError(x, y) = cost_value;
				Planes(x, y, 0) = a2;
				Planes(x, y, 1) = b2;
				Planes(x, y, 2) = c2;
			}
		}
		normal_std /= 2;
		c_std_curr /= 2;
	}

}

template<int radius>
float check_error(const bool direction, const MexImage<const float> &Reference, const MexImage<const float> &Template, const MexImage<float> &Weights, const int x, const int y, const float a, const float b, const float c)
{
	const int width = Reference.width;
	const int height = Reference.height;
	const int colors = Reference.layers;

	const int diameter = radius * 2 + 1;
	const int window = diameter*diameter;
	const int dir = direction ? +1 : -1;

	float cost_value = 0;
	float pixels = 0;
	for (int w = 0; w < window; w++)
	{
		const int dx = w / diameter - radius;
		const int dy = w % diameter - radius;

		const int xx = x + dx;
		const int yy = y + dy;
		if (xx < 0 || xx >= width || yy < 0 || y >= height)
		{
			continue;
		}

		float d = a * xx + b * yy + c;
		const float xr = xx + d*dir;
		if (xr < 0.f || xr >= width - 1.f)
		{
			continue;
		}
		const int xr_f = int(floor(xr));
		const int xr_c = int(ceil(xr));
		pixels++;
		float diff = 0;
		for (int c = 0; c < colors; c++)
		{
			float R_color = Template(xr_f, yy, c) * (xr_c - xr) + Template(xr_c, yy, c) * (xr - xr_f);
			diff += abs(Reference(xx, yy, c) - R_color);
		}

		cost_value += diff * Weights(x, y, w);
	}
	//mexPrintf("(%d, %d): [%5.2f %5.2f %5.2f] = %5.2f \n", x, y, a, b, c, cost_value);
	return pixels >(window / 3) ? cost_value : FLT_MAX;
}


void mexFunction(int nout, mxArray* output[], int in, const mxArray* input[])
{
#ifndef _NDEBUG
	omp_set_num_threads(std::max(8, omp_get_max_threads()));
	omp_set_dynamic(std::max(7, omp_get_max_threads() - 1));
	//omp_set_dynamic(omp_get_max_threads());
#endif	

	if (in < 6 || in > 10 || nout != 4 || mxGetClassID(input[0]) != mxSINGLE_CLASS || mxGetClassID(input[1]) != mxSINGLE_CLASS || mxGetClassID(input[2]) != mxSINGLE_CLASS || mxGetClassID(input[3]) != mxSINGLE_CLASS)
	{
		mexErrMsgTxt("USAGE: [Al, Ar, ErrL, ErrR] = patchmatch_slanted_propagation(single(Left), single(Right), single(LeftGuide), single(RightGuide), radius, maxdisp, [mindisp = 0, iterations=10, forget_rate=0.9, sigma]);");
	}

	const int radius = static_cast<int>(mxGetScalar(input[4]));
	const int maxdisp = static_cast<int>(mxGetScalar(input[5]));
	const int mindisp = (in > 6) ? static_cast<int>(mxGetScalar(input[6])) : 0;
	const int iterations = (in > 7) ? static_cast<int>(mxGetScalar(input[7])) : 10;
	const float forget_rate = (in > 8) ? static_cast<int>(mxGetScalar(input[8])) : 0.9f;
	const float sigma = (in > 9) ? static_cast<int>(mxGetScalar(input[9])) : 10.f;
	//const float cost_threshold = (in > 5) ? static_cast<float>(mxGetScalar(input[5])) : 1000.f;
	//co0nst int y_offset = (in > 6) ? static_cast<int>(mxGetScalar(input[6])) : 0;
	if (maxdisp <= mindisp)
	{
		mexErrMsgTxt("ERROR: 'maxdisp' must be larger than 'mindisp'!");
	}

	const MexImage<const float> Left(input[0]);
	const MexImage<const float> Right(input[1]);
	const MexImage<const float> LeftGuide(input[2]);
	const MexImage<const float> RightGuide(input[3]);
	const int width = Left.width;
	const int height = Left.height;
	const int layers = Left.layers;
	const int colors = LeftGuide.layers;
	const int64_t HW = static_cast<int64_t>(Left.layer_size);

	if (height != Right.height || width != Right.width || layers != Right.layers)
	{
		mexErrMsgTxt("ERROR: Sizes of parameters 'Left', 'Right' must be the same.");
	}
	if (height != LeftGuide.height || width != LeftGuide.width || colors != LeftGuide.layers)
	{
		mexErrMsgTxt("ERROR: Sizes of parameters 'Left', 'Right' must be the same.");
	}
	if (height != RightGuide.height || width != RightGuide.width || colors != RightGuide.layers)
	{
		mexErrMsgTxt("ERROR: Sizes of parameters 'Left', 'Right' must be the same.");
	}

	if (colors < 1 || colors > 9)
	{
		mexErrMsgTxt("Too many colors in your images.");
	}

	const mwSize planeDims[] = { (size_t)height, (size_t)width, 3 };
	const mwSize errDims[] = { (size_t)height, (size_t)width, 1 };

	output[0] = mxCreateNumericArray(3, planeDims, mxSINGLE_CLASS, mxREAL);
	output[1] = mxCreateNumericArray(3, planeDims, mxSINGLE_CLASS, mxREAL);
	output[2] = mxCreateNumericArray(3, errDims, mxSINGLE_CLASS, mxREAL);
	output[3] = mxCreateNumericArray(3, errDims, mxSINGLE_CLASS, mxREAL);
	const MexImage<float> PlanesL(output[0]);
	const MexImage<float> PlanesR(output[1]);
	const MexImage<float> BestErrorL(output[2]);
	const MexImage<float> BestErrorR(output[3]);
	//const MexImage<float> BestErrorL(width, height);
	//const MexImage<float> BestErrorR(width, height);
	PlanesL.setval(0);
	PlanesR.setval(0);
	BestErrorL.setval(FLT_MAX);
	BestErrorR.setval(FLT_MAX);

	const float c_init = (maxdisp - mindisp) / 2.f + mindisp;
	const float c_std = (maxdisp - mindisp);
	const float ab_std = 10.f;
	MexImage<float>(PlanesL, 2).setval(c_init);
	MexImage<float>(PlanesR, 2).setval(c_init);

	switch (radius)
	{
	case 2: optimized_patchmatch<2>(Left, Right, LeftGuide, RightGuide, PlanesL, PlanesR, BestErrorL, BestErrorR, mindisp, maxdisp, ab_std, c_std, iterations, forget_rate, sigma); break;
	case 3: optimized_patchmatch<3>(Left, Right, LeftGuide, RightGuide, PlanesL, PlanesR, BestErrorL, BestErrorR, mindisp, maxdisp, ab_std, c_std, iterations, forget_rate, sigma); break;
	case 4: optimized_patchmatch<4>(Left, Right, LeftGuide, RightGuide, PlanesL, PlanesR, BestErrorL, BestErrorR, mindisp, maxdisp, ab_std, c_std, iterations, forget_rate, sigma); break;
	case 5: optimized_patchmatch<5>(Left, Right, LeftGuide, RightGuide, PlanesL, PlanesR, BestErrorL, BestErrorR, mindisp, maxdisp, ab_std, c_std, iterations, forget_rate, sigma); break;
	case 6: optimized_patchmatch<6>(Left, Right, LeftGuide, RightGuide, PlanesL, PlanesR, BestErrorL, BestErrorR, mindisp, maxdisp, ab_std, c_std, iterations, forget_rate, sigma); break;
	case 7: optimized_patchmatch<7>(Left, Right, LeftGuide, RightGuide, PlanesL, PlanesR, BestErrorL, BestErrorR, mindisp, maxdisp, ab_std, c_std, iterations, forget_rate, sigma); break;
	case 8: optimized_patchmatch<8>(Left, Right, LeftGuide, RightGuide, PlanesL, PlanesR, BestErrorL, BestErrorR, mindisp, maxdisp, ab_std, c_std, iterations, forget_rate, sigma); break;
	case 9: optimized_patchmatch<9>(Left, Right, LeftGuide, RightGuide, PlanesL, PlanesR, BestErrorL, BestErrorR, mindisp, maxdisp, ab_std, c_std, iterations, forget_rate, sigma); break;
	case 10: optimized_patchmatch<10>(Left, Right, LeftGuide, RightGuide, PlanesL, PlanesR, BestErrorL, BestErrorR, mindisp, maxdisp, ab_std, c_std, iterations, forget_rate, sigma); break;
	default: optimized_patchmatch<1>(Left, Right, LeftGuide, RightGuide, PlanesL, PlanesR, BestErrorL, BestErrorR, mindisp, maxdisp, ab_std, c_std, iterations, forget_rate, sigma); break;
	}




}