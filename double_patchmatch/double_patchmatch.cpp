/** Caculates Double-planesweep Slanted PatchMatch
* @file double_patchmatch
* @date 21.05.2016
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
void optimized_patchmatch(const MexImage<const float> & Left, const MexImage<const float> & Right, const MexImage<float> &PlanesL, const MexImage<float> &PlanesR, const MexImage<float> & BestErrorL, const MexImage<float> & BestErrorR, const int mindisp, const int maxdisp, const int iterations, const float sigma)
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
			prepare_weights<radius>(Left, WeightsL, sigma);
		}
#pragma omp section
		{
			WeightsR.setval(0.f);
			prepare_weights<radius>(Right, WeightsR, sigma);
		}
	}
#pragma omp parallel sections
	{
#pragma omp section
		{
			single_patchmatch<radius>(false, Left, Right, PlanesL, BestErrorL, WeightsL, mindisp, maxdisp, iterations);
		}
#pragma omp section
		{
			single_patchmatch<radius>(true, Right, Left, PlanesR, BestErrorR, WeightsR, mindisp, maxdisp, iterations);
		}
	}
}

//class plane
//{
//public:
//	plane getRandom(const float _d, const float _nx, const float _ny, const float d_std, const float n_std)
//	{
//
//	}
//
//private:
//	plane();
//
//	float a, b, c;
//	float x, y, d;
//	float nx, ny;
//
//};

struct cost_triple
{	
	float cost, cost_fg, cost_bg;
	cost_triple(float a, float b, float c) : cost(a), cost_fg(b), cost_bg(c)
	{
	}

	cost_triple() : cost(0.f), cost_fg(0.f), cost_bg(0.f)
	{}

	cost_triple(const cost_triple &cost) : cost(cost.cost), cost_fg(cost.cost_fg), cost_bg(cost.cost_fg)
	{
	}
	
};

template<int radius>
void single_patchmatch(const bool direction, const MexImage<const float> &Reference, const MexImage<const float> &Template, const MexImage<float> &Planes, const MexImage<float> & BestError, const MexImage<float> & Weights, const int mindisp, const int maxdisp, const float iterations)
{
	const int width = Reference.width;
	const int height = Reference.height;
	const int colors = Reference.layers;
	const int64_t HW = static_cast<int64_t>(Reference.layer_size);
	const MexImage<float> PlanesF(width, height, 3);
	const MexImage<float> PlanesB(width, height, 3);
	const MexImage<float> CostF(width, height);
	const MexImage<float> CostB(width, height);

	const int diameter = radius * 2 + 1;
	const int window = diameter*diameter;
	const int dir = direction ? +1 : -1;	
	float normal_std = 3;
	float d_mean = (maxdisp+mindisp)/2.f;
	float d_std = maxdisp - d_mean;
	//mexPrintf("Random Init \n");
	#pragma omp parallel for

	//random initialization
	for (int64_t i = 0; i < HW; i++)
	{
		const int x = i / height;
		const int y = i % height;

		const float d_fg = d_mean + d_std  * (1 - 2 * float(rand()) / (RAND_MAX));
		const float nx = (1 - 2 * float(rand()) / (RAND_MAX));
		const float ny = (1 - 2 * float(rand()) / (RAND_MAX));
		const float nz = float(rand()) / (RAND_MAX)+0.0001;
		const float norm = sqrt(nx*nx + ny*ny + nz*nz);
		const float af = -nx / nz;
		const float bf = -ny / nz;
		const float cf = (nx*x + ny*y + nz*d_fg) / nz;
		PlanesF(x, y, 0) = af;
		PlanesF(x, y, 1) = bf;
		PlanesF(x, y, 2) = cf;

		float d_bg = maxdisp*2.f;
		while (d_bg >= mindisp && d_bg <= d_fg)
		{
			d_bg = d_mean + d_std  * (1 - 2 * float(rand()) / (RAND_MAX));
		}

		//float d_bg = c_std_curr  * (0.5 - float(rand()) / (RAND_MAX)) + (Planes(x, y, 2) - c_std + d_fg) / 2;
		//d_bg = d_bg > d_fg ? d_fg : d_bg;
		const float nxb = (1 - 2 * float(rand()) / (RAND_MAX));
		const float nyb = (1 - 2 * float(rand()) / (RAND_MAX));
		const float nzb = float(rand()) / (RAND_MAX)+0.0001;
		const float normb = sqrt(nxb*nxb + nyb*nyb + nzb*nzb);
		const float ab = -nxb / nzb;
		const float bb = -nyb / nzb;
		const float cb = (nxb*x + nyb*y + nzb*d_bg) / nzb;
		PlanesB(x, y, 0) = ab;
		PlanesB(x, y, 1) = bb;
		PlanesB(x, y, 2) = cb;
		
		cost_triple cost = check_error<radius>(direction, Reference, Template, Weights, x, y, af, bf, cf, ab, bb, cb);

		BestError(x, y) = cost.cost;
		CostF(x, y) = cost.cost_fg;
		CostB(x, y) = cost.cost_bg;
	}

	
	for (int iter = 0; iter < iterations; iter++)
	{
		//mexPrintf("Iteration \n");		

		// left-to-right foreground pass
		#pragma omp parallel for
		for (int y = 0; y < height; y++)
		{
			for (int x = 1; x < width; x++)
			{
				const float af = PlanesF(x - 1, y, 0);
				const float bf = PlanesF(x - 1, y, 1);
				const float cf = PlanesF(x - 1, y, 2);
				const float ab = PlanesB(x, y, 0);
				const float bb = PlanesB(x, y, 1);
				const float cb = PlanesB(x, y, 2);

				const float df = af*x + bf*y + cf;
				if (df < mindisp || df > maxdisp)
				{
					continue;
				}

				cost_triple cost = check_error<radius>(direction, Reference, Template, Weights, x, y, af, bf, cf, ab, bb, cb);

				if (cost.cost < BestError(x, y))
				{
					BestError(x, y) = cost.cost;
					CostF(x, y) = cost.cost_fg;
					CostB(x, y) = cost.cost_bg;

					PlanesF(x, y, 0) = af;
					PlanesF(x, y, 1) = bf;
					PlanesF(x, y, 2) = cf;
				}
			}
		}

		// left-to-right background pass
#pragma omp parallel for
		for (int y = 0; y < height; y++)
		{
			for (int x = 1; x < width; x++)
			{
				const float af = PlanesF(x, y, 0);
				const float bf = PlanesF(x, y, 1);
				const float cf = PlanesF(x, y, 2);
				const float ab = PlanesB(x - 1, y, 0);
				const float bb = PlanesB(x - 1, y, 1);
				const float cb = PlanesB(x - 1, y, 2);

				const float db = ab*x + bb*y + cb;
				if (db < mindisp || db > maxdisp)
				{
					continue;
				}

				cost_triple cost = check_error<radius>(direction, Reference, Template, Weights, x, y, af, bf, cf, ab, bb, cb);

				if (cost.cost < BestError(x, y))
				{
					BestError(x, y) = cost.cost;
					CostF(x, y) = cost.cost_fg;
					CostB(x, y) = cost.cost_bg;

					PlanesB(x, y, 0) = ab;
					PlanesB(x, y, 1) = bb;
					PlanesB(x, y, 2) = cb;
				}
			}
		}

		// right-to-left foreground pass
#pragma omp parallel for
		for (int y = 0; y < height; y++)
		{
			for (int x = width-2; x >= 0; x--)
			{
				const float af = PlanesF(x + 1, y, 0);
				const float bf = PlanesF(x + 1, y, 1);
				const float cf = PlanesF(x + 1, y, 2);
				const float ab = PlanesB(x, y, 0);
				const float bb = PlanesB(x, y, 1);
				const float cb = PlanesB(x, y, 2);			

				const float df = af*x + bf*y + cf;
				if (df < mindisp || df > maxdisp)
				{
					continue;
				}

				cost_triple cost = check_error<radius>(direction, Reference, Template, Weights, x, y, af, bf, cf, ab, bb, cb);

				if (cost.cost < BestError(x, y))
				{
					BestError(x, y) = cost.cost;
					CostF(x, y) = cost.cost_fg;
					CostB(x, y) = cost.cost_bg;

					PlanesF(x, y, 0) = af;
					PlanesF(x, y, 1) = bf;
					PlanesF(x, y, 2) = cf;
				}
			}
		}

		// right-to-left background pass
#pragma omp parallel for
		for (int y = 0; y < height; y++)
		{
			for (int x = width - 2; x >= 0; x--)
			{
				const float af = PlanesF(x, y, 0);
				const float bf = PlanesF(x, y, 1);
				const float cf = PlanesF(x, y, 2);
				const float ab = PlanesB(x + 1, y, 0);
				const float bb = PlanesB(x + 1, y, 1);
				const float cb = PlanesB(x + 1, y, 2);

				const float db = ab*x + bb*y + cb;
				if (db < mindisp || db > maxdisp)
				{
					continue;
				}

				cost_triple cost = check_error<radius>(direction, Reference, Template, Weights, x, y, af, bf, cf, ab, bb, cb);

				if (cost.cost < BestError(x, y))
				{
					BestError(x, y) = cost.cost;
					CostF(x, y) = cost.cost_fg;
					CostB(x, y) = cost.cost_bg;

					PlanesB(x, y, 0) = ab;
					PlanesB(x, y, 1) = bb;
					PlanesB(x, y, 2) = cb;
				}
			}
		}

		// up-to-bottom foreground pass
		#pragma omp parallel for
		for (int x = 0; x < width; x++)
		{
			for (int y = 1; y < height; y++)
			{
				const float af = PlanesF(x, y-1, 0);
				const float bf = PlanesF(x, y-1, 1);
				const float cf = PlanesF(x, y-1, 2);
				const float ab = PlanesB(x, y, 0);
				const float bb = PlanesB(x, y, 1);
				const float cb = PlanesB(x, y, 2);

				const float df = af*x + bf*y + cf;
				if (df < mindisp || df > maxdisp)
				{
					continue;
				}

				cost_triple cost = check_error<radius>(direction, Reference, Template, Weights, x, y, af, bf, cf, ab, bb, cb);

				if (cost.cost < BestError(x, y))
				{
					BestError(x, y) = cost.cost;
					CostF(x, y) = cost.cost_fg;
					CostB(x, y) = cost.cost_bg;

					PlanesF(x, y, 0) = af;
					PlanesF(x, y, 1) = bf;
					PlanesF(x, y, 2) = cf;
				}				
			}
		}

		// up-to-bottom background pass
		#pragma omp parallel for
		for (int x = 0; x < width; x++)
		{
			for (int y = 1; y < height; y++)
			{
				const float af = PlanesF(x, y, 0);
				const float bf = PlanesF(x, y, 1);
				const float cf = PlanesF(x, y, 2);
				const float ab = PlanesB(x, y - 1, 0);
				const float bb = PlanesB(x, y - 1, 1);
				const float cb = PlanesB(x, y - 1, 2);

				const float db = ab*x + bb*y + cb;
				if (db < mindisp || db > maxdisp)
				{
					continue;
				}

				cost_triple cost = check_error<radius>(direction, Reference, Template, Weights, x, y, af, bf, cf, ab, bb, cb);

				if (cost.cost < BestError(x, y))
				{
					BestError(x, y) = cost.cost;
					CostF(x, y) = cost.cost_fg;
					CostB(x, y) = cost.cost_bg;

					PlanesB(x, y, 0) = ab;
					PlanesB(x, y, 1) = bb;
					PlanesB(x, y, 2) = cb;
				}
			}
		}

		// bottom-to-up foreground pass
		#pragma omp parallel for
		for (int x = 0; x < width; x++)
		{
			for (int y = height - 2; y >= 0; y--)
			{
				const float af = PlanesF(x, y + 1, 0);
				const float bf = PlanesF(x, y + 1, 1);
				const float cf = PlanesF(x, y + 1, 2);
				const float ab = PlanesB(x, y, 0);
				const float bb = PlanesB(x, y, 1);
				const float cb = PlanesB(x, y, 2);
				
				const float df = af*x + bf*y + cf;
				if (df < mindisp || df > maxdisp)
				{
					continue;
				}				

				cost_triple cost = check_error<radius>(direction, Reference, Template, Weights, x, y, af, bf, cf, ab, bb, cb);

				if (cost.cost < BestError(x, y))
				{
					BestError(x, y) = cost.cost;
					CostF(x, y) = cost.cost_fg;
					CostB(x, y) = cost.cost_bg;

					PlanesF(x, y, 0) = af;
					PlanesF(x, y, 1) = bf;
					PlanesF(x, y, 2) = cf;
				}				
			}
		}

		// bottom-to-up background pass
		#pragma omp parallel for
		for (int x = 0; x < width; x++)
		{
			for (int y = height - 2; y >= 0; y--)
			{
				const float af = PlanesF(x, y, 0);
				const float bf = PlanesF(x, y, 1);
				const float cf = PlanesF(x, y, 2);
				const float ab = PlanesB(x, y + 1, 0);
				const float bb = PlanesB(x, y + 1, 1);
				const float cb = PlanesB(x, y + 1, 2);

				const float db = ab*x + bb*y + cb;
				if (db < mindisp || db > maxdisp)
				{
					continue;
				}

				cost_triple cost = check_error<radius>(direction, Reference, Template, Weights, x, y, af, bf, cf, ab, bb, cb);

				if (cost.cost < BestError(x, y))
				{
					BestError(x, y) = cost.cost;
					CostF(x, y) = cost.cost_fg;
					CostB(x, y) = cost.cost_bg;

					PlanesB(x, y, 0) = ab;
					PlanesB(x, y, 1) = bb;
					PlanesB(x, y, 2) = cb;
				}
			}
		}

		// foreground refinement
		#pragma omp parallel for
		for (int64_t i = 0; i < HW; i++)
		{
			const int x = i / height;
			const int y = i % height;
			const float af = PlanesF(x, y, 0);// +ab_std_curr * (1 - 2 * float(rand()) / (RAND_MAX));
			const float bf = PlanesF(x, y, 1);// +ab_std_curr * (1 - 2 * float(rand()) / (RAND_MAX));
			const float cf = PlanesF(x, y, 2);//  curr_d2 - a2*x - b2*y;
			const float df = (af*x + bf*y + cf);

			const float ab = PlanesB(x, y, 0);// +ab_std_curr * (1 - 2 * float(rand()) / (RAND_MAX));
			const float bb = PlanesB(x, y, 1);// +ab_std_curr * (1 - 2 * float(rand()) / (RAND_MAX));
			const float cb = PlanesB(x, y, 2);//  curr_d2 - a2*x - b2*y;


			float nz = sqrt(1.f / (1 + af*af + bf*bf));
			float nx = -af * nz;
			float ny = -bf * nz;
			nx += normal_std * (1 - 2 * float(rand()) / (RAND_MAX));
			ny += normal_std * (1 - 2 * float(rand()) / (RAND_MAX));
			nz += normal_std * (1 - 2 * float(rand()) / (RAND_MAX));
			nz = (nz < 0.001) ? 0.001 : nz;
			const float d2 = df + d_std * (1 - 2 * float(rand()) / (RAND_MAX));
			const float norm = sqrt(nx*nx + ny*ny + nz*nz);

			const float a2 = -nx / nz;
			const float b2 = -ny / nz;
			const float c2 = (nx*x + ny*y + nz*d2) / nz;

			cost_triple cost = check_error<radius>(direction, Reference, Template, Weights, x, y, a2, b2, c2, ab, bb, cb);

			if (cost.cost < BestError(x, y))
			{
				BestError(x, y) = cost.cost;
				CostF(x, y) = cost.cost_fg;
				CostB(x, y) = cost.cost_bg;

				PlanesF(x, y, 0) = a2;
				PlanesF(x, y, 1) = b2;
				PlanesF(x, y, 2) = c2;
			}
		}

		// background refinement
		#pragma omp parallel for
		for (int64_t i = 0; i < HW; i++)
		{
			const int x = i / height;
			const int y = i % height;
			const float af = PlanesF(x, y, 0);// +ab_std_curr * (1 - 2 * float(rand()) / (RAND_MAX));
			const float bf = PlanesF(x, y, 1);// +ab_std_curr * (1 - 2 * float(rand()) / (RAND_MAX));
			const float cf = PlanesF(x, y, 2);//  curr_d2 - a2*x - b2*y;			

			const float ab = PlanesB(x, y, 0);// +ab_std_curr * (1 - 2 * float(rand()) / (RAND_MAX));
			const float bb = PlanesB(x, y, 1);// +ab_std_curr * (1 - 2 * float(rand()) / (RAND_MAX));
			const float cb = PlanesB(x, y, 2);//  curr_d2 - a2*x - b2*y;
			const float db = (ab*x + bb*y + cb);

			float nz = sqrt(1.f / (1 + ab*ab + bb*bb));
			float nx = -ab * nz;
			float ny = -bb * nz;
			nx += normal_std * (1 - 2 * float(rand()) / (RAND_MAX));
			ny += normal_std * (1 - 2 * float(rand()) / (RAND_MAX));
			nz += normal_std * (1 - 2 * float(rand()) / (RAND_MAX));
			nz = (nz < 0.001) ? 0.001 : nz;
			const float d2 = db + d_std * (1 - 2 * float(rand()) / (RAND_MAX));
			const float norm = sqrt(nx*nx + ny*ny + nz*nz);

			const float a2 = -nx / nz;
			const float b2 = -ny / nz;
			const float c2 = (nx*x + ny*y + nz*d2) / nz;

			cost_triple cost = check_error<radius>(direction, Reference, Template, Weights, x, y, af, bf, cf, a2, b2, c2);

			if (cost.cost < BestError(x, y))
			{
				BestError(x, y) = cost.cost;
				CostF(x, y) = cost.cost_fg;
				CostB(x, y) = cost.cost_bg;

				PlanesB(x, y, 0) = a2;
				PlanesB(x, y, 1) = b2;
				PlanesB(x, y, 2) = c2;
			}
		}

		normal_std /= 2;
		d_std /= 2;
	}

	#pragma omp parallel for
	for (int64_t i = 0; i < HW; i++)
	{
		const int x = i / height;
		const int y = i % height;
		
		if (CostF(x, y) < CostB(x, y))
		{
			Planes(x, y, 0) = PlanesF(x, y, 0);
			Planes(x, y, 1) = PlanesF(x, y, 1);
			Planes(x, y, 2) = PlanesF(x, y, 2);
		}
		else
		{
			Planes(x, y, 0) = PlanesB(x, y, 0);
			Planes(x, y, 1) = PlanesB(x, y, 1);
			Planes(x, y, 2) = PlanesB(x, y, 2);
		}
	}

}

template<int radius>
cost_triple check_error(const bool direction, const MexImage<const float> &Reference, const MexImage<const float> &Template, const MexImage<float> &Weights, const int x, const int y, const float af, const float bf, const float cf, const float ab, const float bb, const float cb)
{
	
	const int width = Reference.width;
	const int height = Reference.height;
	const int colors = Reference.layers;

	const int diameter = radius * 2 + 1;
	const int window = diameter*diameter;
	const int dir = direction ? +1 : -1;

	float cost_value = 0;
	float cost_fg = 0.f, cost_bg = 0.f;
	int pixels = 0;
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

		const float df = af * xx + bf * yy + cf;
		const float db = ab * xx + bb * yy + cb;

		const float _xfr = xx + df*dir;
		const float _xbr = xx + db*dir;
		const float xfr = (_xfr < 0.f) ? 0.f : ((_xfr >= (width - 1.f)) ? width - 1.f : _xfr);
		const float xbr = (_xbr < 0.f) ? 0.f : ((_xbr >= (width - 1.f)) ? width - 1.f : _xbr);

		float diffF = 0, diffB = 0;
		//if (xfr < 0.f || xfr >= (width - 1.f) || xbr < 0.f || xbr >= (width - 1.f))
		//{
		//	continue;
		//	//diff = cost_thr;
		//}
		//else
		//{			
			const int xfr_f = int(floor(xfr));
			const int xfr_c = int(ceil(xfr));

			const int xbr_f = int(floor(xbr));
			const int xbr_c = int(ceil(xbr));

			pixels++;

			for (int c = 0; c < colors; c++)
			{
				float Rf_color = Template(xfr_f, yy, c) * (xfr_c - xfr) + Template(xfr_c, yy, c) * (xfr - xfr_f);
				diffF += abs(Reference(xx, yy, c) - Rf_color);

				float Rb_color = Template(xbr_f, yy, c) * (xbr_c - xbr) + Template(xbr_c, yy, c) * (xbr - xbr_f);
				diffB += abs(Reference(xx, yy, c) - Rb_color);
			}

			cost_fg += diffF*Weights(x, y, w);
			cost_bg += diffB*Weights(x, y, w);			
		//}

		cost_value += std::min(diffF, diffB);
	}
	//mexPrintf("(%d, %d): [%5.2f %5.2f %5.2f] = %5.2f \n", x, y, a, b, c, cost_value);
	cost_value = pixels > diameter ? cost_value / pixels : FLT_MAX;
	
	return cost_triple(cost_value, cost_fg, cost_bg);
}


void mexFunction(int nout, mxArray* output[], int in, const mxArray* input[])
{
#ifndef _NDEBUG
	omp_set_num_threads(std::max(8, omp_get_max_threads()));
	omp_set_dynamic(std::max(7, omp_get_max_threads() - 1));
	//omp_set_dynamic(omp_get_max_threads());
#endif	

	if (in < 4 || in > 7 || nout != 4 || mxGetClassID(input[0]) != mxSINGLE_CLASS || mxGetClassID(input[1]) != mxSINGLE_CLASS)
	{
		mexErrMsgTxt("USAGE: [AL, AR, ErrL, ErrR] = double_patchmatch(single(Left), single(Right), radius, maxdisp, [mindisp = 0, iterations=4]);");
	}

	const int radius = static_cast<int>(mxGetScalar(input[2]));
	const int maxdisp = static_cast<int>(mxGetScalar(input[3]));
	const int mindisp = (in > 4) ? static_cast<int>(mxGetScalar(input[4])) : 0;
	const int iterations = (in > 5) ? static_cast<int>(mxGetScalar(input[5])) : 4;
	const float sigma = (in > 6) ? static_cast<float>(mxGetScalar(input[6])) : 10.f;
	
	if (maxdisp <= mindisp)
	{
		mexErrMsgTxt("ERROR: 'maxdisp' must be larger than 'mindisp'!");
	}

	const MexImage<const float> Left(input[0]);
	const MexImage<const float> Right(input[1]);
	const int width = Left.width;
	const int height = Left.height;
	const int colors = Left.layers;
	const int64_t HW = static_cast<int64_t>(Left.layer_size);

	if (height != Right.height || width != Right.width || colors != Right.layers)
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

	PlanesL.setval(0);
	PlanesR.setval(0);
	BestErrorL.setval(FLT_MAX);
	BestErrorR.setval(FLT_MAX);

	switch (radius)
	{
	case 2: optimized_patchmatch<2>(Left, Right, PlanesL, PlanesR, BestErrorL, BestErrorR, mindisp, maxdisp, iterations, sigma); break;
	case 3: optimized_patchmatch<3>(Left, Right, PlanesL, PlanesR, BestErrorL, BestErrorR, mindisp, maxdisp, iterations, sigma); break;
	case 4: optimized_patchmatch<4>(Left, Right, PlanesL, PlanesR, BestErrorL, BestErrorR, mindisp, maxdisp, iterations, sigma); break;
	case 5: optimized_patchmatch<5>(Left, Right, PlanesL, PlanesR, BestErrorL, BestErrorR, mindisp, maxdisp, iterations, sigma); break;
	case 6: optimized_patchmatch<6>(Left, Right, PlanesL, PlanesR, BestErrorL, BestErrorR, mindisp, maxdisp, iterations, sigma); break;
	case 7: optimized_patchmatch<7>(Left, Right, PlanesL, PlanesR, BestErrorL, BestErrorR, mindisp, maxdisp, iterations, sigma); break;
	case 8: optimized_patchmatch<8>(Left, Right, PlanesL, PlanesR, BestErrorL, BestErrorR, mindisp, maxdisp, iterations, sigma); break;
	case 9: optimized_patchmatch<9>(Left, Right, PlanesL, PlanesR, BestErrorL, BestErrorR, mindisp, maxdisp,  iterations, sigma); break;
	case 10: optimized_patchmatch<10>(Left, Right, PlanesL, PlanesR, BestErrorL, BestErrorR, mindisp, maxdisp, iterations, sigma); break;
	default: optimized_patchmatch<1>(Left, Right, PlanesL, PlanesR, BestErrorL, BestErrorR, mindisp, maxdisp, iterations, sigma); break;
	}


}