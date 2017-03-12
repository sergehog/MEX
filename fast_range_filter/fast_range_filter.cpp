/** fast_range_filter O(1) complexity range filter
*	@file fast_range_filter.cpp
*	@date 04.09.2014
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

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")

using namespace mymex;


template <unsigned levels, unsigned colors>
class histogram_size
{
public: 
	const static size_t size = histogram_size<levels, colors-1>::size * levels;
	//const static size_t index = histogram_size<levels, colors-1>::index * levels;
};


template <unsigned levels>
class histogram_size<levels, 1>
{
public: 	
	const static size_t size = levels;
	//const static size_t index = 1;
};


template <unsigned levels, unsigned colors>
class histogram
{
	static_assert(colors>0, "number of colors must be positive");
	static_assert(levels>0, "number of levels must be positive");	

	public: 
	size_t indexes[colors];	
	uint8_t histo[histogram_size<levels, colors>::size];
	

	histogram()
	{
		size_t index_value = 1;
		for(unsigned i=0; i<colors; i++)
		{
			indexes[i] = index_value;
			index_value *= levels;
		}
	}


	size_t index(const uint8_t color[colors])
	{
		size_t index_value = 0;
		for(unsigned i=0; i<colors; i++)
		{
			index_value += static_cast<size_t>(levels*color[i]/256.f) * (indexes[i]);
		}
		return index_value;
	}
	

	uint8_t &value(const uint8_t color[colors])
	{
		return histo[index(color)];
	}

	void increment(const uint8_t color[colors])
	{
		size_t indexx = index(color);
		if(histo[indexx] < 255)
		{
			histo[indexx] ++;
		}
	}

	void decrement(const uint8_t color[colors])
	{
		size_t indexx = index(color);
		if(histo[indexx] > 0)
		{
			histo[indexx] --;
		}
	}

	void operator=(histogram<levels, colors> &hist)
	{
		for(size_t i = 0; i<sizeof(histogram_size<levels, colors>::size); i++)
		{
			histo[i] = hist.histo[i];
		}
	}

	void operator+=(histogram<levels, colors> &hist)
	{
		for(size_t i = 0; i<sizeof(histogram_size<levels, colors>::size); i++)
		{
			histo[i] += hist.histo[i];
		}
	}

	void operator-=(histogram<levels, colors> &hist)
	{
		for(size_t i = 0; i<sizeof(histogram_size<levels, colors>::size); i++)
		{
			histo[i] -= hist.histo[i];
		}
	}

private:
	histogram(histogram &a)
	{
	}

};



template <unsigned levels, unsigned colors>
class histogram_list
{	
public:
	
	histogram<levels, colors> * const histograms;

	histogram_list(const int length) : histograms(new histogram<levels, colors>[length])
	{

	}

	histogram<levels, colors>& operator[](const int i)
	{
		return histograms[i];
	}

	~histogram_list()
	{
		delete histograms;
	}

private: 
	histogram_list()
	{
		delete[] histograms;
	}

};


template<unsigned colors>
static void get_color(const MexImage<uint8_t> &Image, const int x, const int y, uint8_t color[colors])
{
	for(int c=0; c<colors; c++)
	{
		color[c] = Image(x,y,c);
	}
}


template<unsigned levels, unsigned colors>
void process_color(const MexImage<float> &Signal, const MexImage<uint8_t> &Image, const MexImage<float> &Filtered, const int radius, const float sigma_distance)
{
	typedef histogram<levels, colors> histo;
	const int width = Signal.width;
	const int height = Signal.height;
	const int layers = Signal.layers;
	
	std::unique_ptr<histo> histograms_ptr = std::unique_ptr<histo> (new histo [height+1]);
	histo* histograms = histograms_ptr.get();
		
	for(int x=0; x<=radius; x++)
	{
		for(int y=0; y<=radius; y++)
		{		
			uint8_t color[colors];
			get_color<colors>(Image, x, y, color);
			histograms[0].increment(color);
		}
	}

	for(int y=1; y<height; y++)
	{
		histograms[y] = histograms[y-1];
		for(int x=0; x<=radius; x++)
		{
			if(y-radius >= 0)
			{
				uint8_t color[colors];
				get_color<colors>(Image, x, y-radius, color);
				histograms[y].decrement(color);
			}
			if(y+radius <= height-1)
			{
				uint8_t color[colors];
				get_color<colors>(Image, x, y+radius, color);
				histograms[y].increment(color);
			}
		}
	}

	
}

template<unsigned levels>
void process(const MexImage<float> &Signal, const MexImage<uint8_t> &Image, const MexImage<float> &Filtered, const int radius, const float sigma_distance)
{
	static_assert(levels>2, "number of levels must be positive");

	if(Image.layers == 1)
	{
		process_color<levels, 1>(Signal, Image, Filtered, radius, sigma_distance);
	}
	else if(Image.layers == 2)
	{
		process_color<levels, 2>(Signal, Image, Filtered, radius, sigma_distance);
	}
	else if(Image.layers == 3)
	{
		process_color<levels, 3>(Signal, Image, Filtered, radius, sigma_distance);
	}
	else if(Image.layers == 4)
	{
		process_color<levels, 4>(Signal, Image, Filtered, radius, sigma_distance);
	}
	else if(Image.layers == 5)
	{
		process_color<levels, 5>(Signal, Image, Filtered, radius, sigma_distance);
	}
	else if(Image.layers == 6)
	{
		process_color<levels, 6>(Signal, Image, Filtered, radius, sigma_distance);
	}
	else if(Image.layers == 7)
	{
		process_color<levels, 7>(Signal, Image, Filtered, radius, sigma_distance);
	}
	else if(Image.layers == 8)
	{
		process_color<levels, 8>(Signal, Image, Filtered, radius, sigma_distance);
	}
	else if(Image.layers == 9)
	{
		process_color<levels, 9>(Signal, Image, Filtered, radius, sigma_distance);
	}
	else
	{
		mexErrMsgTxt("Not supported number of colors!");
	}
}

void mexFunction (int nout, mxArray* output[], int in, const mxArray* input[])
{
#ifndef _DEBUG
	omp_set_num_threads(std::max<int>(4, omp_get_max_threads()));
	omp_set_dynamic(std::max<int>(2, omp_get_max_threads()/2));
#endif
	
	if(in < 3 || in > 5 || nout != 1 || mxGetClassID(input[0]) != mxSINGLE_CLASS && mxGetClassID(input[1]) != mxUINT8_CLASS)
	{
		mexErrMsgTxt("USAGE: [Filtered] = fast_range_filter(single(Signal), uint8(Image), radius, <sigma, quantization_levels=64>);");
    }
	 
	MexImage<float> Signal(input[0]);
	MexImage<uint8_t> Image(input[1]);

	const int radius = std::max(1, (int)mxGetScalar(input[2])); 
	const float sigma_distance = (in > 3) ? static_cast<float>(mxGetScalar(input[3])) : 1.0; 		
	const unsigned levels = (in > 4) ? std::max<unsigned>(64, static_cast<unsigned>(mxGetScalar(input[4]))) : 64; 		

	const int diameter = radius*2+1;
	const int window = diameter*diameter;
	
	const int width = Signal.width;
	const int height = Signal.height;
	const int layers = Signal.layers;
	const int colors = Image.layers;

	if(width != Image.width || height != Image.height)
	{
		mexErrMsgTxt("Resolution of Image and Signal must coincide!");
	}

	const size_t dims[] = {(size_t)height, (size_t)width, (size_t)layers};
	output[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL); 	
	MexImage<float> Filtered(output[0]);
	//if(levels == 128)
	//{
	//	process<128>(Signal, Image, Filtered, radius, sigma_distance);
	//}
	//else 
	if(levels == 64)
	{
		process<64>(Signal, Image, Filtered, radius, sigma_distance);
	}
	else if(levels == 32)
	{
		process<32>(Signal, Image, Filtered, radius, sigma_distance);
	}
	else if(levels == 16)
	{
		process<16>(Signal, Image, Filtered, radius, sigma_distance);
	}	
	else
	{
		mexErrMsgTxt("Not supported range decimation!");				
	}	
}

