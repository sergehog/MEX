/** Optimized color_range_filter with complexity O(HW*Q*N)
*	@file color_range_filter_new.cpp
*	@date 04.05.2015
*	@author Sergey Smirnov <sergey.smirnov@tut.fi>
*/

#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}
#endif // __cplusplus

#include "../common/meximage.h"
#include <memory>
#ifndef _DEBUG
#include <omp.h>
#endif
#include <algorithm>

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")

using namespace mymex;
#define gaussian_filter 0.99f

typedef long long size_i;
void fast_filtering(const MexImage<float> &Signal, const MexImage<float> &Temporal, const float sigma);


//// helper class to calculate overall number of levels (quantization ^ colors) in compile time
//template<unsigned levels, unsigned colors>
//class layers_counter
//{
//public:
//	static const size_i count = layers_counter<levels, colors - 1>::count * levels;
//	static const size_i weight = layers_counter<levels, colors - 1>::weight * 2;
//};
//
//template<unsigned levels>
//class layers_counter<levels, 1>
//{
//public:
//	static const size_i count = levels;
//	static const size_i weight = 2;
//};

//! Base class for polymorfic call
struct color_filtering
{
	virtual void process_signal(const MexImage<float> &Signal, const MexImage<uint8_t> &Image, MexImage<float> &Filtered, MexImage<float> &Buffer, const float sigma_spatial, const float sigma_color) {}
};

//! levels - quantization levels, colors - number of color components
template<unsigned levels, unsigned colors>
struct cost_matcher : base_matcher
{
	// step between different layers
	static const float step = 255.f / (levels - 1);

	// length of main cycle 
	static const size_i count = cost_matcher<levels, colors - 1>::count * levels;
	
	// buffer size
	static const size_i buffer_layers = colors * levels;

	void process_signal(const MexImage<float> &Signal, const MexImage<uint8_t> &Image, MexImage<float> &Filtered, MexImage<float> &Buffer, const float sigma_spatial, const float sigma_color)
	{
		const long HW = Signal.layer_size;
		const int width = Signal.width;
		const int height = Signal.height;

		// prepare a buffer with the pre-filtered signals 
		#pragma omp parallel
		{
			MexImage<float> Temporal(width, height);
			#pragma omp for
			for (int bl = 0; bl < buffer_layers; bl++)
			{
				const int color = bl / levels;
				const int level = bl % levels;

				MexImage<float> BufferLayer(Buffer, bl);
				MexImage<uint8_t> Color(Image, color);

				#pragma omp parallel for
				for (long index = 0; index < HW; index++)
				{
					BufferLayer[index] = Signal[index] * abs(float(Color[index]) - step*level);
				}

				fast_filtering(BufferLayer, Temporal, sigma_spatial);
			}
		}

		// calculate indexes to access +
		size_i indexes[colors];
		indexes[0] = 1;
		for (unsigned c = 1; c<colors; c++)
		{
			indexes[c] = indexes[c - 1] * levels;
		}


		// main cycle over all possible color-level variants
		#pragma omp parallel
		{
			float color[colors];
			#pragma omp for
			for (size_i slice = 0; slice < count; slice++)
			{
				size_i remainder = slice;
				for (int c = colors - 1; c >= 0; c--)
				{
					color[c] = (remainder / indexes[c]) * step;
					remainder = remainder % indexes[c];
				}
			}
		}
	
	}
};

// lower level implementation
template<unsigned levels>
struct cost_matcher<levels, 0> : base_matcher
{
	static const float two_sq = sqrt(2);
	static const float step = 255.f / (levels - 1);
	static const size_i count = levels;

	void process_color(const MexImage<float> &Signal, const MexImage<uint8_t> &Image, MexImage<float> &Filtered, const float sigma_spatial, const float sigma_color)
	{}
};



template<unsigned levels, unsigned colors>
void process_color(const MexImage<float> &Signal, const MexImage<uint8_t> &Image, MexImage<float> &Filtered, const float sigma_spatial, const float sigma_color)
{
	static const float two_sq = sqrt(2);
	static const float step = 255.f / (levels - 1);
	const int width = Image.width;
	const int height = Image.height;
	const int layers = Signal.layers;
	const long HW = width * height;

	size_i indexes[colors];
	//float const sigma_distance = 1.f - pow<float, int>(0.1f, radius/10.f);

	indexes[0] = 1;
	for (unsigned c = 1; c<colors; c++)
	{
		indexes[c] = indexes[c - 1] * levels;
	}
	Filtered.setval(0.f);
	MexImage<float> AggWeight(width, height);
	AggWeight.setval(0.f);

#pragma omp parallel 
	{
		MexImage<float> Weight(width, height);
		MexImage<float> SignalS(width, height, layers);
#ifdef gaussian_filter
		MexImage<float> TemporalBuff(width, height);
#endif
		float color[colors];

#pragma omp for schedule(dynamic)
		for (size_i slice = 0; slice < layers_counter<levels, colors>::count; slice++)
		{
			//Weight.setval(0.f);
			//SignalS.setval(0.f);
			size_i remainder = slice;

			for (int c = colors - 1; c >= 0; c--)
			{
				color[c] = (remainder / indexes[c]) * step;
				remainder = remainder % indexes[c];
			}

			//for(long i=0; i<HW; i++)
			//{
			//	const int x = i / height;
			//	const int y = i % height;
			//	
			//	int maxd = 0; 
			//	float dist_weight = 0;
			//	for(unsigned c=0; c<colors; c++)
			//	{
			//		int diff = abs(Image(x,y,c)-color[c]);
			//		maxd = diff > maxd ? diff : maxd;
			//		dist_weight += (Image(x,y,c)-color[c]) * (Image(x,y,c)-color[c]);
			//	}
			//	if(maxd > step)
			//	{
			//		continue;
			//	}
			//	dist_weight = two_sq-sqrt(dist_weight/(step*step));
			//	
			//	#pragma omp atomic
			//	AggWeight(x,y) += dist_weight;
			//	for(unsigned c=0; c<colors; c++)
			//	{
			//		//const float value = color[c]/layers_counter<levels, colors>::weight;
			//		const float value = color[c] * dist_weight;
			//		#pragma omp atomic
			//		Filtered(x,y,c) += value;
			//	}
			//}
			long zeros = 0;
#pragma omp parallel for reduction(+:zeros)
			for (long i = 0; i<HW; i++)
			{
				const int x = i / height;
				const int y = i % height;
				float diff = 0;
#pragma parallel
				for (unsigned c = 0; c<colors; c++)
				{
					diff += abs(float(Image(x, y, c)) - float(color[c]));
				}

				zeros += diff <= step ? 1 : 0;
				Weight(x, y) = exp(-diff / (255 * /*colors **/ sigma_color));
#pragma loop
				for (int l = 0; l<layers; l++)
				{
					SignalS(x, y, l) = Signal(x, y, l) * Weight(x, y);
				}
			}

			if (zeros < 100)
			{
				continue;
			}

#ifdef gaussian_filter
			fast_filtering(SignalS, TemporalBuff, sigma_spatial);
			fast_filtering(Weight, TemporalBuff, sigma_spatial);
#else
			Weight.IntegralImage(true);
			SignalS.IntegralImage(true);
#endif			

#pragma omp parallel for
			for (long i = 0; i<HW; i++)
			{
				const int x = i / height;
				const int y = i % height;

				int maxd = 0;
				float dist_weight = 0;
				for (unsigned c = 0; c<colors; c++)
				{
					const int diff = abs(Image(x, y, c) - color[c]);
					maxd = diff > maxd ? diff : maxd;
					dist_weight += (Image(x, y, c) - color[c]) * (Image(x, y, c) - color[c]);
				}
				if (maxd > step)
				{
					continue;
				}
				dist_weight = two_sq - sqrt(dist_weight / (step*step));
#pragma omp atomic
				AggWeight(x, y) += dist_weight;

#ifdef gaussian_filter
				const float weight = Weight(x, y);
#else
				const float weight = Weight.getIntegralAverage(x, y, radius);
#endif

#pragma loop
				for (int l = 0; l<layers; l++)
				{
#ifdef gaussian_filter
					const float value = dist_weight * SignalS(x, y, l) / weight;
#else
					const float value = dist_weight * SignalS.getIntegralAverage(x, y, radius, l) / weight;
#endif

#pragma omp atomic
					Filtered(x, y, l) += value;
				}
			}

		}
	}

#pragma omp parallel for	
	for (long i = 0; i<HW; i++)
	{
		const int x = i / height;
		const int y = i % height;
#pragma loop
		for (int l = 0; l<layers; l++)
		{
			Filtered(x, y, l) /= AggWeight(x, y);
		}
	}
}


template<unsigned levels>
void process(const MexImage<float> &Signal, const MexImage<uint8_t> &Image, MexImage<float> &Filtered, const float sigma_spatial, const float sigma_color)
{
	if (Image.layers == 1)
	{
		process_color<levels, 1>(Signal, Image, Filtered, sigma_spatial, sigma_color);
	}
	else if (Image.layers == 2)
	{
		process_color<levels, 2>(Signal, Image, Filtered, sigma_spatial, sigma_color);
	}
	else if (Image.layers == 3)
	{
		process_color<levels, 3>(Signal, Image, Filtered, sigma_spatial, sigma_color);
	}
	else if (Image.layers == 4)
	{
		process_color<levels, 4>(Signal, Image, Filtered, sigma_spatial, sigma_color);
	}
	else if (Image.layers == 5)
	{
		process_color<levels, 5>(Signal, Image, Filtered, sigma_spatial, sigma_color);
	}
	else if (Image.layers == 6)
	{
		process_color<levels, 6>(Signal, Image, Filtered, sigma_spatial, sigma_color);
	}
	else if (Image.layers == 7)
	{
		process_color<levels, 7>(Signal, Image, Filtered, sigma_spatial, sigma_color);
	}
	else if (Image.layers == 8)
	{
		process_color<levels, 8>(Signal, Image, Filtered, sigma_spatial, sigma_color);
	}
	else if (Image.layers == 9)
	{
		process_color<levels, 9>(Signal, Image, Filtered, sigma_spatial, sigma_color);
	}
	else
	{
		mexErrMsgTxt("Not supported range decimation!");
	}

}


void mexFunction(int nout, mxArray* output[], int in, const mxArray* input[])
{
#ifndef _DEBUG
	omp_set_num_threads(std::max<int>(4, omp_get_max_threads()));
	omp_set_dynamic(std::max<int>(2, omp_get_max_threads() / 2));
#endif

	if (in < 3 || in > 5 || nout != 1 || mxGetClassID(input[0]) != mxSINGLE_CLASS || mxGetClassID(input[1]) != mxUINT8_CLASS)
	{
		mexErrMsgTxt("USAGE: [Filtered] = color_range_filter(single(Signal), uint8(Image), <sigma_color, sigma_spatial, quantization_levels=8>);");
	}

	MexImage<float> Signal(input[0]);
	MexImage<uint8_t> Image(input[1]);

	const float sigma_color = (in > 2) ? std::max<float>(0.f, std::min<float>(1.f, mxGetScalar(input[2]))) : 0.1;
	const float sigma_spatial = (in > 3) ? std::max<float>(0.f, std::min<float>(1.f, mxGetScalar(input[3]))) : 0.1;

	//const int radius = std::max(1, (int)mxGetScalar(input[2])); 
	//const float sigma_color = (in > 3) ? static_cast<float>(mxGetScalar(input[3])) : 1.0; 		
	const unsigned levels = (in > 4) ? std::min<unsigned>(32, static_cast<unsigned>(mxGetScalar(input[4]))) : 8;

	const int width = Signal.width;
	const int height = Signal.height;
	const int layers = Signal.layers;
	const int colors = Image.layers;

	if (width != Image.width || height != Image.height)
	{
		mexErrMsgTxt("Resolution of Image and Signal must coincide!");
	}

	const size_t dims[] = { (size_t)height, (size_t)width, (size_t)layers };
	output[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
	MexImage<float> Filtered(output[0]);
	if (levels == 4)
	{
		process<4>(Signal, Image, Filtered, sigma_spatial, sigma_color);
	}
	else if (levels == 5)
	{
		process<5>(Signal, Image, Filtered, sigma_spatial, sigma_color);
	}
	else if (levels == 6)
	{
		process<6>(Signal, Image, Filtered, sigma_spatial, sigma_color);
	}
	else if (levels == 7)
	{
		process<7>(Signal, Image, Filtered, sigma_spatial, sigma_color);
	}
	else if (levels == 8)
	{
		process<8>(Signal, Image, Filtered, sigma_spatial, sigma_color);
	}
	else if (levels == 9)
	{
		process<9>(Signal, Image, Filtered, sigma_spatial, sigma_color);
	}
	else if (levels == 10)
	{
		process<10>(Signal, Image, Filtered, sigma_spatial, sigma_color);
	}
	else if (levels == 11)
	{
		process<11>(Signal, Image, Filtered, sigma_spatial, sigma_color);
	}
	else if (levels == 12)
	{
		process<12>(Signal, Image, Filtered, sigma_spatial, sigma_color);
	}
	else if (levels == 13)
	{
		process<13>(Signal, Image, Filtered, sigma_spatial, sigma_color);
	}
	else if (levels == 14)
	{
		process<14>(Signal, Image, Filtered, sigma_spatial, sigma_color);
	}
	else if (levels == 15)
	{
		process<15>(Signal, Image, Filtered, sigma_spatial, sigma_color);
	}
	else if (levels == 16)
	{
		process<16>(Signal, Image, Filtered, sigma_spatial, sigma_color);
	}
	else if (levels == 32)
	{
		process<32>(Signal, Image, Filtered, sigma_spatial, sigma_color);
	}
	else
	{
		mexErrMsgTxt("Not supported quantization!");
	}
}


void fast_filtering(const MexImage<float> &Signal, const MexImage<float> &Temporal, const float sigma)
{
	const int width = Signal.width;
	const int height = Signal.height;
	const int layers = Signal.layers;
	const long HW = Signal.layer_size;
	std::unique_ptr<float> temporal1_ptr = std::unique_ptr<float>(new float[std::max(width, height)]);
	std::unique_ptr<float> temporal2_ptr = std::unique_ptr<float>(new float[std::max(width, height)]);
	float* temporal1 = temporal1_ptr.get();
	float* temporal2 = temporal2_ptr.get();

	for (int l = 0; l<layers; l++)
	{
		// Temporal.setval(0.f);
		// Horisontal (left-to-right & right-to-left) pass
		for (int y = 0; y<height; y++)
		{
			temporal1[0] = Signal(0, y, l);
			temporal2[width - 1] = Signal(width - 1, y, l);
			for (int x1 = 1; x1<width; x1++)
			{
				const int x2 = width - x1 - 1;
				temporal1[x1] = (Signal(x1, y, l) + temporal1[x1 - 1] * sigma);
				temporal2[x2] = (Signal(x2, y, l) + temporal2[x2 + 1] * sigma);
			}
			for (int x = 0; x<width; x++)
			{
				Temporal(x, y) = (temporal1[x] + temporal2[x]) - Signal(x, y, l);
			}
		}

		// Vertical (up-to-down & down-to-up) pass
		for (int x = 0; x<width; x++)
		{
			temporal1[0] = Temporal(x, 0);
			temporal2[height - 1] = Temporal(x, height - 1);
			for (int y1 = 1; y1<height; y1++)
			{
				const int y2 = height - y1 - 1;
				temporal1[y1] = (Temporal(x, y1) + temporal1[y1 - 1] * sigma);
				temporal2[y2] = (Temporal(x, y2) + temporal2[y2 + 1] * sigma);
			}
			for (int y = 0; y<height; y++)
			{
				Signal(x, y, l) = (temporal1[y] + temporal2[y]) - Temporal(x, y);
			}
		}
	}
}