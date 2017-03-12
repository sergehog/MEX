/** Segments image using recursive-superpixels approach
* @file super_pixels.cpp
* @author Sergey Smirnov <sergey.smirnov@tut.fi>
* @date 08.09.2015
* @copyright 3D Media Group / Tampere University of Technology
*/


#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}
#endif // __cplusplus

#include "../common/meximage2.h"
//#include "../common/defines.h"
#include <vector>
#define _USE_MATH_DEFINES
#include <cmath>
#include <algorithm>
#include <memory>
#include <array>
#ifndef _DEBUG
#include <omp.h>
#endif

//#define M_PI       3.14159265358979323846

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")

//typedef unsigned char uint8;
using namespace mymex;

float getWeight(const MexImage<uint8_t> &Color, const int x1, const int y1, const int x2, const int y2, const float * const weights_lookup)
{	
	if (x1 < 0 || x1 >= Color.width || x2 < 0 || x2 >= Color.width)
	{
		return 0.f;
	}
	if (y1 < 0 || y1 >= Color.height || y2 < 0 || y2 >= Color.height)
	{
		return 0.f;
	}

	int diff = 0;
	
	for (int c = 0; c < Color.layers; c++)
	{
		diff += (abs(int(Color(x1, y1, c)) - Color(x2, y2, c)));
	}	
	return weights_lookup[diff];
}

void recursive_bilateral(const MexImage<uint8_t> &Image, MexImage<float> &Signal, MexImage<float> &Temporal, MexImage<float> &Weights, MexImage<float> &Weights2, float const * const weights_lookup, float const sigma_spatial)
{
	const int width = Image.width;
	const int height = Image.height;
	const long HW = width*height;

	//const float alpha = exp(-sqrt(2.0) / (sigma_spatial*std::min(height, width)));
	const float alpha = sigma_spatial;

	Weights.setval(0.f);
	Temporal.setval(0.f);

	// horisontal passes
#pragma omp parallel for
	for (int y = 0; y<height; y++)
	{
		float t1 = Signal(0, y);
		float t2 = Signal(width - 1, y);
		float w1 = 1.f;
		float w2 = 1.f;

		Temporal(0, y) = t1;
		Temporal(width - 1, y) = t2;
		Weights(0, y) = w1;
		Weights(width - 1, y) = w2;

		for (int x1 = 1; x1<width; x1++)
		{			
			
			const float weight1 = getWeight(Image , x1, y, x1 - 1, y, weights_lookup);
			t1 = Signal(x1, y) + t1 * alpha*weight1;
			w1 = (1 + w1 * alpha*weight1);
			Temporal(x1, y) += t1;
			Weights(x1, y) += w1;
				
				
			const int x2 = width - x1 - 1;
			const float weight2 = getWeight(Image, x2, y, x2 + 1, y, weights_lookup);
			t2 = Signal(x2, y) + t2 * alpha*weight2;
			w2 = (1 + w2 * alpha*weight2);
			Temporal(x2, y) += t2;
			Weights(x2, y) += w2;			
		}

		#pragma omp parallel for
		for (int x = 0; x<width; x++)
		{
			Weights(x, y) -= 1;
			Temporal(x, y) -= Signal(x, y);
		}
	}

	Signal.setval(0.f);
	Weights2.setval(0.f);

	//vertical passes		
#pragma omp parallel for
	for (int x = 0; x<width; x++)
	{
		float t1 = Temporal(x, 0);
		float t2 = Temporal(x, height - 1);
		float w1 = Weights(x, 0);
		float w2 = Weights(x, height - 1);

		Signal(x, 0) = t1;
		Signal(x, height - 1) = t2;
		Weights2(x, 0) = w1;
		Weights2(x, height - 1) = w2;

		for (int y1 = 1; y1<height; y1++)
		{
			const float weight1 = getWeight(Image, x, y1, x, y1 - 1, weights_lookup);
			t1 = Temporal(x, y1) + t1 * alpha*weight1;
			w1 = (Weights(x, y1) + w1 * alpha*weight1);

			Signal(x, y1) += t1;
			Weights2(x, y1) += w1;
			const int y2 = height - y1 - 1;
			const float weight2 = getWeight(Image, x, y2, x, y2 + 1, weights_lookup);
			t2 = Temporal(x, y2) + t2 * alpha*weight2;
			w2 = (Weights(x, y2) + w2 * alpha*weight2);
			Signal(x, y2) += t2;
			Weights2(x, y2) += w2;
		}

		#pragma omp parallel for
		for (int y = 0; y<height; y++)
		{
			Weights2(x, y) -= Weights(x, y);
			Signal(x, y) -= Temporal(x, y);
		}
	}

	// final normalization
	#pragma omp parallel for
	for (long i = 0; i<HW; i++)
	{
		const int x = i / height;
		const int y = i % height;
		Signal(x, y) /= Weights2(x, y);
	}
}


void mexFunction(const int nout, mxArray* output[], const int in, const mxArray* input[])
{
#ifndef _DEBUG
	omp_set_num_threads(std::max<int>(4, omp_get_max_threads()));
	omp_set_dynamic(std::max<int>(2, omp_get_max_threads() / 2));
#endif

	if (in < 2 || in > 5 || nout != 2 || mxGetClassID(input[0]) != mxUINT8_CLASS)
	{
		mexErrMsgTxt("USAGE: [SuperPixels, Buffer] = super_pixels(uint8(image), density_factor, <sigma_color, sigma_distance, iterations>);");
	}

	MexImage<uint8_t> Image(input[0]);
	
	const int density = std::max<int>(2, (int)mxGetScalar(input[1]));
	const float sigma_color = in > 2 ? std::min<float>(1.f, std::max<float>(0.f, (float)mxGetScalar(input[2]))) : 0.2f;
	const float sigma_distance = in > 3 ? std::min<float>(1.f, std::max<float>(0.f, (float)mxGetScalar(input[3]))) : 0.8f;
	const int iterations = in > 4 ? std::max<int>(1, (int)mxGetScalar(input[4])) : 1;
	const int diameter = density * 2 + 1;
	const int window = diameter * diameter;
	const int width = Image.width;
	const int height = Image.height;
	const int colors = Image.layers;
	const long HW = width*height;
	const float nan = sqrt(-1.f);

	const int pix_height = height / density;
	const int pix_width = width / density;

	const size_t dims1[] = { (size_t)height, (size_t)width, 1};
	const size_t dims4[] = { (size_t)height, (size_t)width, 4};

	output[0] = mxCreateNumericArray(3, dims1, mxINT32_CLASS, mxREAL);
	output[1] = mxCreateNumericArray(3, dims4, mxSINGLE_CLASS, mxREAL);
		
	MexImage<int32_t> SuperPixels(output[0]);		
	//MexImage<float> Buffer1(width, height, 4);
	MexImage<float> Buffer1(output[1]);	
	Buffer1.setval(0.f);	

	std::unique_ptr<float[]> weights_lookup(new float[256 * colors + 1]);
	for (int i = 0; i < 256 * colors; i++)
	{
		weights_lookup[i] = exp(-float(i) / (255.f * sigma_color * colors));
	}

#pragma omp parallel	
	{		
		MexImage<float> Temporal(diameter, diameter);
		MexImage<float> Weights(diameter, diameter);
		MexImage<float> Weights2(diameter, diameter);

		#pragma omp for
		for (int32_t pixel = 0; pixel < pix_height*pix_width; pixel++)
		{
			const int pix_x = pixel / pix_height;
			const int pix_y = pixel % pix_height;
			const int x = pix_x*density;
			const int y = pix_y*density;
			const int buffer = pix_y % 2 + 2 * (pix_x % 2);
			Buffer1(x, y, buffer) = float(window);
			MexImage<float> SubBuffer(Buffer1, x - density, y - density, x + density, y + density, buffer);
			MexImage<uint8_t> SubImage(Image, x - density, y - density, x + density, y + density);
			for (int i = 0; i < iterations; i++)
			{
				recursive_bilateral(SubImage, SubBuffer, Temporal, Weights, Weights2, weights_lookup.get(), sigma_distance);
			}
		}
	}

	for (long i = 0; i < HW; i++)
	{
		const int x = i / height;
		const int y = i % height;
		const int pix_x = x / density;
		const int pix_y = y / density;		
		const int buffer = pix_y % 2 + 2 * (pix_x % 2);
		const int bufferR = pix_x < pix_width ? pix_y % 2 + 2 * ((pix_x + 1) % 2) : -1;
		const int bufferD = pix_y < pix_height ? (pix_y + 1) % 2 + 2 * (pix_x % 2) : -1;
		const int bufferRD = pix_y < pix_height && pix_x < pix_width ? (pix_y + 1) % 2 + 2 * ((pix_x+1) % 2) : -1;

		const int32_t pixel = pix_x * pix_height + pix_y;
		const int32_t pixelR = (pix_x + 1) * pix_height + pix_y;
		const int32_t pixelD = pix_x * pix_height + pix_y  + 1;
		const int32_t pixelRD = (pix_x+1)* pix_height + pix_y + 1;

		if (bufferR < 0 || bufferD < 0 || bufferRD < 0)
		{
			SuperPixels(x, y) = pixel;
			continue;
		}
		if (Buffer1(x, y, buffer) > Buffer1(x, y, bufferR))
		{
			if (Buffer1(x, y, buffer) > Buffer1(x, y, bufferD))
			{
				SuperPixels(x, y) = (Buffer1(x, y, buffer) > Buffer1(x, y, bufferRD)) ? pixel : pixelRD;
			}
			else
			{
				SuperPixels(x, y) = (Buffer1(x, y, bufferD) > Buffer1(x, y, bufferRD)) ? pixelD : pixelRD;
			}
		}
		else
		{
			if (Buffer1(x, y, bufferR) > Buffer1(x, y, bufferD))
			{
				SuperPixels(x, y) = (Buffer1(x, y, bufferR) > Buffer1(x, y, bufferRD)) ? pixelR : pixelRD;
			}
			else
			{
				SuperPixels(x, y) = (Buffer1(x, y, bufferD) > Buffer1(x, y, bufferRD)) ? pixelD : pixelRD;
			}
		}

	}
	return;

	/*#pragma omp parallel	
	{
		//std::unique_ptr<float[]> weigths(new float[window]);
		//std::unique_ptr<float[]> buffer1(new float[window]);
		//std::unique_ptr<float[]> buffer2(new float[window]);
		//MexImage<float> buffer1(diameter, diameter);
		//MexImage<float> buffer2(diameter, diameter);
		//MexImage<float> weights(diameter, diameter);

		#pragma omp for
		for (int32_t pixel = 0; pixel < pix_height*pix_width; pixel++)
		{
			//buffer1.setval(0.f);
			//buffer2.setval(0.f);
			const int pix_x = pixel / pix_height;
			const int pix_y = pixel % pix_height;
			const int x = pix_x*density;
			const int y = pix_y*density;
			const int buffer = pix_y % 2 + 2 * (pix_x % 2);
			Buffer1(x, y, buffer) = 1.f;


			for (int w = 0; w < window; w++)
			{
				const int dx = w / diameter - density;
				const int dy = w % diameter - density;
				const int xw = x + dx;
				const int yw = x + dx;
				if (xw < 0 || xw >= width || yw < 0 || yw >= height)
				{
					continue;
				}
				buffer1(dx, dy) = Buffer1(xw, yw, buffer);
				//buffer1[w] = Buffer1(x, y, buffer);
			}
			
			// horisontal passes
			#pragma omp parallel for
			for (int dy = 0; dy < diameter; dy++)
			{
				if (y + dy < 0 || y + dy >= height)
				{
					continue;
				}
				float t1 = buffer1(0, dy);
				float t2 = (diameter-1, dy);
				float w1 = 1.f;
				float w2 = 1.f;

				buffer2(0, dy) = t1;
				buffer2(diameter - 1, dy) = t2;
				weights(0, dy) = w1;
				weights(diameter - 1, dy) = w2;

				for (int dx = 1; dx < diameter; dx++)
				{
					if (x + dx >= width-1)
					{
						continue;
					}
					int dist = 0;
					for (int c = 0; c < colors; c++)
					{
						dist += abs(int(Image(x+dx, y+dy, c)) - Image(x - 1, y, c));
					}

					const float weight = weights_lookup[dist] * sigma_distance;


					//const float weight1 = getWeight2(Color1, Color2, x1, y, x1 - 1, y, weights_lookup);
					const float weight1 = sigma_distance;
					t1 = buffer1(dx, dy) + t1 * weight1;
					w1 = (1 + w1 * weight1);

					buffer2(dx, dy) += t1;
					weights(dx, dy) += w1;
				}
			}


		}
	}



	Buffer2.setval(0.f);
	for (int y = 0; y < height; y++)	
	{		
		float buffers[4] = { Buffer1(0, y, 0), Buffer1(0, y, 1), Buffer1(0, y, 2), Buffer1(0, y, 3)};

		Buffer2(0, y, 0) = buffers[0];
		Buffer2(0, y, 1) = buffers[1];
		Buffer2(0, y, 2) = buffers[2];
		Buffer2(0, y, 3) = buffers[3];

		float weights[4] = {1.f, 1.f, 1.f, 1.f};


		// Left-to-Right pass
		for (int x = 1; x < width; x ++)
		{			
			const long index = x*height + y;
			const int pix_x = (x / density);
			const int pix_y = (y / density);
			const uint32_t pixel = pix_x*pix_height + pix_y;
			const int buffer = pix_y % 2 + 2 * (pix_x % 2);			
			const int bufferR = pix_x < pix_width ? pix_y % 2 + 2 * ((pix_x + 1) % 2) : -1;
			const int bufferD = pix_y < pix_height ? (pix_y + 1) % 2 + 2 * (pix_x % 2) : -1;			

			int dist = 0;
			for (int c = 0; c < colors; c++)
			{				
				dist += abs(int(Image(x, y, c)) - Image(x - 1, y, c));
			}
			
			const float weight = weights_lookup[dist] * sigma_distance;


			if (pix_x*density == x && pix_y*density == y)
			{
				Buffer2(x, y, buffer) += Buffer1(x, y, buffer) + Buffer1(x - 1, y, buffer) * weight;
			}			
			else if (pix_x*density == x)
			{
				Buffer2(x, y, buffer) += Buffer1(x, y, buffer) + Buffer1(x - 1, y, buffer) * weight;
				if (bufferR >= 0)
				{
					Buffer2(x, y, bufferR) += Buffer1(x, y, bufferR) + Buffer1(x - 1, y, bufferR) * weight;
				}
			}
			else if (pix_y*density == y)
			{

				Buffer2(x, y, buffer) += Buffer1(x, y, buffer) + Buffer1(x - 1, y, buffer) * weight;
				if (bufferD >= 0)
				{
					Buffer2(x, y, bufferD) += Buffer1(x, y, bufferD) + Buffer1(x - 1, y, bufferD) * weight;
				}
			}
			else
			{
				Buffer2(x, y, 0) += Buffer1(x, y, 0) + Buffer1(x - 1, y, 0) * weight;
				Buffer2(x, y, 1) += Buffer1(x, y, 1) + Buffer1(x - 1, y, 1) * weight;
				Buffer2(x, y, 2) += Buffer1(x, y, 2) + Buffer1(x - 1, y, 2) * weight;
				Buffer2(x, y, 3) += Buffer1(x, y, 3) + Buffer1(x - 1, y, 3) * weight;
			}
		}

		// Right-to-Left pass
		Buffer2(width - 1, y, 0) += Buffer1(width - 1, y, 0);
		Buffer2(width - 1, y, 1) += Buffer1(width - 1, y, 1);
		Buffer2(width - 1, y, 2) += Buffer1(width - 1, y, 2);
		Buffer2(width - 1, y, 3) += Buffer1(width - 1, y, 3);

		for (int x = width - 2; x >= 0; x--)
		{
			const long index = x*height + y;
			const int pix_x = (x / density);
			const int pix_y = (y / density);
			const uint32_t pixel = pix_x*pix_height + pix_y;
			const int buffer = pix_y % 2 + 2 * (pix_x % 2);
			const int bufferR = pix_x < pix_width ? pix_y % 2 + 2 * ((pix_x + 1) % 2) : -1;
			const int bufferD = pix_y < pix_height ? (pix_y + 1) % 2 + 2 * (pix_x % 2) : -1;

			int dist = 0;
			for (int c = 0; c < colors; c++)
			{
				dist += abs(int(Image(x, y, c)) - Image(x + 1, y, c));
			}

			const float weight = weights_lookup[dist] * sigma_distance;

			if (pix_x*density == x && pix_y*density == y)
			{
				Buffer2(x, y, buffer) += Buffer1(x, y, buffer) + Buffer1(x + 1, y, buffer) * weight;
			}
			else if (pix_x*density == x)
			{
				Buffer2(x, y, buffer) += Buffer1(x, y, buffer) + Buffer1(x + 1, y, buffer) * weight;
				if (bufferR >= 0)
				{
					Buffer2(x, y, bufferR) += Buffer1(x, y, bufferR) + Buffer1(x + 1, y, bufferR) * weight;
				}
			}
			else if (pix_y*density == y)
			{

				Buffer2(x, y, buffer) += Buffer1(x, y, buffer) + Buffer1(x + 1, y, buffer) * weight;
				if (bufferD >= 0)
				{
					Buffer2(x, y, bufferD) += Buffer1(x, y, bufferD) + Buffer1(x + 1, y, bufferD) * weight;
				}
			}
			else
			{
				Buffer2(x, y, 0) += Buffer1(x, y, 0) + Buffer1(x + 1, y, 0) * weight;
				Buffer2(x, y, 1) += Buffer1(x, y, 1) + Buffer1(x + 1, y, 1) * weight;
				Buffer2(x, y, 2) += Buffer1(x, y, 2) + Buffer1(x + 1, y, 2) * weight;
				Buffer2(x, y, 3) += Buffer1(x, y, 3) + Buffer1(x + 1, y, 3) * weight;
			}
		}
	}

	//for (long i = 0; i < HW; i++)
	//{
	//	const int x = i / height;
	//	const int y = i % height;
	//	Buffer2(x, y, 0) -= Buffer1(x, y, 0);
	//	Buffer2(x, y, 1) -= Buffer1(x, y, 1);
	//	Buffer2(x, y, 2) -= Buffer1(x, y, 2);
	//	Buffer2(x, y, 3) -= Buffer1(x, y, 3);
	//}


	
	*/
	
}
