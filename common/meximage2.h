/** Intrface to simplify work with Matlab MEX image format (mxArray)
*	@file meximage.h
*	@date 17.03.2011
*	@author Sergey Smirnov <sergey.smirnov@tut.fi>
*/

#pragma once

#include <exception>
#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}
#endif // __cplusplus
#include <cmath>


namespace mymex
{

	typedef size_t matlab_size;

	struct mymex_window
	{
		int x_low;
		int x_up;
		int y_low;
		int y_up;
		int size;
		void recalculate() { size = (x_up - x_low)*(y_up - y_low); };
	};

	struct mymex_rectangle
	{
		long indexA;
		long indexB;
		long indexC;
		long indexD;
	};

	static int mxGetLayers(const mxArray* image)
	{
		return (int)mxGetNumberOfDimensions(image) > 2 ? (int)(mxGetDimensions(image))[2] : 1;
	}

	static int mxGetWidth(const mxArray* image)
	{
		return (int)(mxGetDimensions(image))[1];
	}

	static int mxGetHeight(const mxArray* image)
	{
		return (int)(mxGetDimensions(image))[0];
	}

	static int mxGetHW(const mxArray* image)
	{
		return ((int)(mxGetDimensions(image))[0])*((int)(mxGetDimensions(image))[1]);
	}

#define LIM(a,w) ((a) < 0 ? 0 : (a) >= (w) ? (w)-1 : (a))

	//! Main class to work with mxArray structures
	template<typename type>
	class MexImage
	{
	private:
		const bool freeOnDestroy;
		type* means;

		MexImage()
		{};

		MexImage(MexImage&)
		{};

		type * const data;

	public:
		const int width;
		const int height;
		const int layers;
		//const long layer_size;
		const long layer_length;
		const long stride;
		
		inline MexImage(type* const _data, const int _width, const int _height, const int _layers = 1) :
			height(_height),
			width(_width),
			layers(_layers),			
			stride(_height),
			layer_length(_height*_width),
			data(_data),
			means(new type[_layers]),
			freeOnDestroy(false)
		{

		};

		//! Initialize MexImage from given mxArray data
		inline MexImage(const mxArray* image) :
			height(mxGetHeight(image)),
			width(mxGetWidth(image)),
			stride(mxGetHeight(image)),
			layer_length(mxGetHW(image)),
			layers(mxGetLayers(image)),
			data(static_cast<type*>(mxGetData(image))),
			means(new type[mxGetLayers(image)]),
			freeOnDestroy(false)
		{

		};

		//! Initialize MexImage with internally allocated memory
		inline MexImage(const int _width, const int _height, const int _layers = 1) :
			width(_width),
			height(_height),
			stride(_height),
			layer_length(_width*_height),
			layers(_layers),
			freeOnDestroy(true),
			data(new type[_width*_height*_layers]),
			means(new type[_layers])
		{
		};

		//! Initialize MexImage from given mxArray data
		inline MexImage(const MexImage& image, int layer) :
			height(image.height),
			width(image.width),
			stride(image.stride),
			layer_length(image.layer_length),
			layers(1),
			data(image.data + layer*image.layer_size),
			means(new type[1]),
			freeOnDestroy(false)
		{

		};

		inline MexImage(const MexImage& image, const int x0, const int y0, const int x1, const int y1) :
			height(LIM(y1, image.height) - LIM(y0, image.height)),
			width(LIM(x1, image.width) - LIM(x0, image.width)),
			layer_length(image.layer_length),
			stride(image.stride),
			layers(image.layers),
			data(image.data + LIM(x0, image.width)*image.stride + LIM(y0, image.height)),
			means(new type[1]),
			freeOnDestroy(false)
		{

		};

		inline MexImage(const MexImage& image, const int x0, const int y0, const int x1, const int y1, const int layer) :
			height(LIM(y1, image.height) - LIM(y0, image.height)),
			width(LIM(x1, image.width) - LIM(x0, image.width)),
			layer_length(image.layer_length),
			stride(image.stride),
			layers(1),
			data(image.data + layer*image.layer_length + LIM(x0, image.width)*image.stride + LIM(y0, image.height)),
			means(new type[1]),
			freeOnDestroy(false)
		{

		};

		~MexImage()
		{
			if (freeOnDestroy)
			{
				delete[] data;
			}

			delete[] means;
		}


		/*T& operator[](long i)
		{
		return data[i];
		}*/

		type &operator()(int x, int y, int c = 0) const
		{
			return data[y + long(x)*stride + c*layer_length];
		}

		bool TryLoad(const mxArray* image)
		{
			if (width == mxGetWidth(image) && height == mxGetHeight(image) && layers == mxGetLayers(image))
			{
				type* imdata = static_cast<type*>(mxGetData(image));

				for (long i = 0; i<layer_size*layers; i++)
				{
					data[i] = imdata[i];
				}
				return true;
			}
			else
			{
				return false;
			}
		};

		bool copyFrom(MexImage& a)
		{
			if (a.width == width && a.height == height && a.layers == layers)
			{
#pragma omp parallel for
				for (long i = 0; i < layer_size*layers; i++)
				{
					data[i] = a[i];
				}
			}

		}


		type getIntegralAverage(int x, int y, int radius, int layer = 0)
		{
			mymex_window wnd = get_window(x, y, radius);
			mymex_rectangle rect = get_rectangle(wnd);
			long cHW = layer*layer_size;

			type avg = (float)data[rect.indexA + cHW];
			avg += data[rect.indexD + cHW];
			avg -= data[rect.indexB + cHW];
			avg -= data[rect.indexC + cHW];
			avg /= wnd.size;

			return avg + means[layer];
		}

		// works for variable number of layers
		void getIntegralAverage(int x, int y, int radius, type *buffer)
		{
			for (int c = 0; c<layers; c++)
			{
				buffer[c] = getIntegralAverage(x, y, radius, c);
			}
		}


		//! works for first layer only
		type getIntegralSum(int x, int y, int radius, int layer = 0)
		{
			mymex_window wnd = get_window(x, y, radius);
			mymex_rectangle rect = get_rectangle(wnd);
			long cHW = layer*layer_size;

			type avg = (float)data[rect.indexA + cHW];
			avg += data[rect.indexD + cHW];
			avg -= data[rect.indexB + cHW];
			avg -= data[rect.indexC + cHW];
			//avg /= wnd.size;
			return avg + wnd.size*means[layer];
		}

		// works for variable number of layers
		type getIntegralAverage(mymex_window wnd, int layer = 0)
		{
			mymex_rectangle rect = get_rectangle(wnd);
			long cHW = layer*layer_size;

			type avg = (float)data[rect.indexA];
			avg += data[rect.indexD];
			avg -= data[rect.indexB];
			avg -= data[rect.indexC];
			avg /= wnd.size;

			return avg + means[layer];
		}


		//! bounded average 
		type getBoundedIntegralAverage(int x, int y, int radius, int x_low, int x_up, int y_low, int y_up, type wrong_value)
		{
			mymex_window wnd = get_window(x, y, radius);
			wnd.x_low = (wnd.x_low < x_low) ? x_low : wnd.x_low;
			wnd.x_up = (wnd.x_up > x_up) ? x_up : wnd.x_up;
			wnd.y_low = (wnd.y_low < y_low) ? y_low : wnd.y_low;
			wnd.y_up = (wnd.y_up > y_up) ? y_up : wnd.y_up;
			wnd.recalculate();

			if (wnd.size <= 0)
			{
				return wrong_value;
			}

			mymex_rectangle rect = get_rectangle(wnd);

			type avg = (float)data[rect.indexA];
			avg += data[rect.indexD];
			avg -= data[rect.indexB];
			avg -= data[rect.indexC];
			avg /= wnd.size;

			return avg;
		}


		void set(MexImage<type>& source)
		{
			if (source.width != width || source.height != height || source.layers != layers)
			{
				throw std::exception("Cannot copy data!");
			}
#pragma omp parallel for
			for (long i = 0; i < layer_size*layers; i++)
			{
				data[i] = source[i];
			}
		}

		//! Calcualte Buffer-less Integral Image
		void IntegralImage(bool compensate_mean = true)
		{
			IntegralFrom(*this, compensate_mean);
		}

		template<typename T>
		void IntegralFrom(MexImage<T>& given, bool compensate_mean = true)
		{
			if (layer_size != given.layer_size || layers != given.layers)
			{
				//throw new std::exception("Wrong MexImage::IntegralImage dimestions");
				mexErrMsgTxt("Wrong MexImage::IntegralImage dimestions \n");
			}

			if (compensate_mean)
			{
				for (size_t c = 0; c<layers; c++)
				{
					type data_mean = 0;

					for (long i = 0; i<layer_size; i++)
					{
						data_mean += given.data[i];
					}
					data_mean /= layer_size;
					means[c] = data_mean;
				}
			}
			else
			{
				for (size_t c = 0; c<layers; c++)
				{
					means[c] = (type)0;
				}
			}

#pragma omp parallel for //independent on layers
			for (int c = 0; c<layers; c++)
			{
				long cHW = c*layer_size;
				type data_mean = means[c];

				for (long index = 0; index<layer_size; index++)
				{
					int y = index % height;
					int x = index / height;

					if (x == y && y == 0)
					{
						given[cHW] = given[cHW] - data_mean;
					}
					else if (x == 0)
					{
						int prev = Index(0, y - 1);
						data[index + cHW] = given[index + cHW] + data[prev + cHW] - data_mean;
					}
					else if (y == 0)
					{
						int prev = Index(x - 1, 0);
						data[index + cHW] = given[index + cHW] + data[prev + cHW] - data_mean;
					}
					else
					{
						int prevX = Index(x - 1, y);
						int prevY = Index(x, y - 1);
						int prev = Index(x - 1, y - 1);
						data[index + cHW] = given[index + cHW] + data[prevX + cHW] + data[prevY + cHW] - data[prev + cHW] - data_mean;
					}
				}
			}
		}


		// Practical, but not 100% correct way to check validity of datatypes :( 
		bool static isValidType(const mxArray* image)
		{
			if (mxGetClassID(image) == mxUINT8_CLASS || mxGetClassID(image) == mxINT8_CLASS)
			{
				return sizeof(type) == 1;
			}
			else if (mxGetClassID(image) == mxUINT16_CLASS || mxGetClassID(image) == mxINT16_CLASS)
			{
				return sizeof(type) == 2;
			}
			else if (mxGetClassID(image) == mxUINT32_CLASS || mxGetClassID(image) == mxINT32_CLASS)
			{
				return sizeof(type) == 4;
			}
			else if (mxGetClassID(image) == mxUINT64_CLASS || mxGetClassID(image) == mxINT64_CLASS)
			{
				return sizeof(type) == 8;
			}
			else if (mxGetClassID(image) == mxSINGLE_CLASS)
			{
				return sizeof(type) == 4;
			}
			else if (mxGetClassID(image) == mxDOUBLE_CLASS)
			{
				return sizeof(type) == 8;
			}
			return true;
		}

		//type& operator[](long index) const
		//{
		//	return data[index];
		//}

		//type& at(long index)  const
		//{
		//	return data[index];
		//}


		void setval(type value) const
		{
			for (int i = 0; i<width*height; i++)
			{
				const int x = i / height;
				const int y = i % height;
				for (int l = 0; l < layers; l++)
				{
					data[y + x*stride + l*layer_length] = value;
				}				
			}
		}

		//inline long Index(int x, int y) const
		//{
		//	return (long)x*stride + y;
		//};

		//inline long Index(int x, int y, int c) const
		//{
		//	return (long)x*stride + y + layer_size*c;
		//};

		inline mxArray* getMxArray()
		{
			return image;
		}

		bool hasNans()
		{
			for (long i = 0; i<layer_size; i++)
			{
				if (_isnan(data[i]))
					return true;
			}

			return false;
		}

	private:

		mymex_window get_window(int x, int y, int radius)
		{
			mymex_window wnd;

			wnd.x_low = x - radius - 1;
			wnd.x_up = x + radius;
			wnd.y_low = y - radius - 1;
			wnd.y_up = y + radius;

			wnd.x_low = (wnd.x_low<0) ? 0 : wnd.x_low;
			wnd.y_low = (wnd.y_low<0) ? 0 : wnd.y_low;
			wnd.x_up = (wnd.x_up >= width) ? width - 1 : wnd.x_up;
			wnd.y_up = (wnd.y_up >= height) ? height - 1 : wnd.y_up;
			//wnd.size = (wnd.x_up-wnd.x_low)*(wnd.y_up-wnd.y_low);
			wnd.recalculate();
			return wnd;
		}

		mymex_rectangle get_rectangle(mymex_window wnd, int layer = 0)
		{
			mymex_rectangle rect;
			rect.indexA = Index(wnd.x_low, wnd.y_low, layer);
			rect.indexB = Index(wnd.x_up, wnd.y_low, layer);
			rect.indexC = Index(wnd.x_low, wnd.y_up, layer);
			rect.indexD = Index(wnd.x_up, wnd.y_up, layer);

			return rect;
		}


	};


	int round(float a)
	{
		return int(a + 0.5);
	}

}
