/** hash_weights
*	@file hash_weights.cpp
*	@date 26.01.2014
*	@author Sergey Smirnov <sergey.smirnov@tut.fi>
*/

#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}
#endif // __cplusplus

#include "../common/meximage.h"
//#include "../common/defines.h"
#include <vector>
#define _USE_MATH_DEFINES
#include <cmath>
#include <algorithm>
#ifndef _DEBUG
#include <omp.h>
#endif

#define M_PI   3.14159265358979323846

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")

typedef unsigned char uint8;
using namespace mymex;


template<typename IT>
void prepareHashImage(MexImage<IT> &Image,  MexImage<IT> &HashImage, const int radius)
{
	const int width = Image.width;
	const int height = Image.height;
	const int colors = Image.layers;
	const long HW = Image.layer_size;
	const int diameter = radius*2+1;
	const int patch_window = diameter*diameter;
	//const int pseudo_colors = colors * patch_window;
	for(int c=0; c<colors; c++)
	{
		for(int w=0; w<patch_window; w++)
		{
			const int dx = w / diameter - radius;
			const int dy = w % diameter - radius;

			#pragma omp parallel for
			for(long i=0; i<HW; i++)
			{
				const int x = i / height;
				const int y = i % height;		
		
				const int xx = (x + dx) < 0 ? 0 : (x + dx) >= width ? width-1 : (x + dx);
				const int yy = (y + dy) < 0 ? 0 : (y + dy) >= height? height-1 : (y + dy);
				HashImage(x,y,w+c*patch_window) = Image(xx,yy,c);
			}
		}			
	}	

}

void prepare_zigzag_ordering(const int radius, const int colors, const int coefficients, int * const zigzag_order)
{
	const int diameter = radius*2+1;
	float * const zigzag_distance = new float[coefficients];
	float const maxdist = sqrt((float)diameter*diameter*2 + colors*colors)+1;

	for(int i=0; i<coefficients; i++)
	{
		zigzag_distance[i] = maxdist;
		//zigzag_order[i][0] = {0,0,0};
	}

	for(int z=0; z<colors; z++)
	{
		for(int y=0; y<diameter; y++)
		{
			for(int x=0; x<diameter; x++)
			{
				float dist = sqrt((float)x*x + y*y + z*z);
				int coords[3] = {x,y,z};
				//coords[0] = x;
				//coords[1] = y;
				//coords[2] = z;
				
				// in-place sorting
				for(int i=0; i<coefficients; i++) 
				{
					if(dist < zigzag_distance[i]) 
					{
						float tem_dist = zigzag_distance[i];
						int tmp_coords[3];
						tmp_coords[0] = zigzag_order[i*3];
						tmp_coords[1] = zigzag_order[i*3+1];
						tmp_coords[2] = zigzag_order[i*3+2];
						
						zigzag_distance[i] = dist;
						zigzag_order[i*3] = coords[0];
						zigzag_order[i*3+1] = coords[1];
						zigzag_order[i*3+2] = coords[2];

						dist = tem_dist;
						//coords = tmp_coords;
						coords[0] = tmp_coords[0];
						coords[1] = tmp_coords[1];
						coords[2] = tmp_coords[2];
						
					}
				}
			}
		}
	}

	delete[] zigzag_distance;
}

template<typename IT, typename HT>
void prepareCompressedHashImage(MexImage<IT> &Image,  MexImage<HT> &HashImage, const int radius, const int coefficients)
{
	const int width = Image.width;
	const int height = Image.height;
	const int colors = Image.layers;
	const long HW = Image.layer_size;
	const int diameter = radius*2+1;
	const int window = diameter*diameter;
	
	HT * const dct_buffer = new HT[window * colors* coefficients];
	const HT norm = sqrt((HT)2/diameter);
	//const HT norm = 1/window;

	int * const zigzag_order = new int[coefficients*3];
	prepare_zigzag_ordering(radius, colors, coefficients, zigzag_order);

	
	#pragma omp parallel for
	for(int k=0; k<coefficients; k++)
	{
		const int k1 = zigzag_order[k*3]; 
		const int k2 = zigzag_order[k*3+1]; 
		const int k3 = zigzag_order[k*3+2]; 

		HT * const element_buffer = dct_buffer + k*window*colors;
		
		HT norm1 = (k1==0) ? 1/sqrt((HT)2) : 1;
		HT norm2 = (k2==0) ? 1/sqrt((HT)2) : 1;
		HT norm3 = (k3==0) ? 1/sqrt((HT)2) : 1;

		for(int n3=0; n3<colors; n3++)
		{
			for(int w=0; w<window; w++)
			{
				const int n1 = w / diameter;
				const int n2 = w % diameter;
				
				element_buffer[w+n3*window] = norm * norm1 * norm2 * norm3 * cos((M_PI/diameter)*(n1+0.5)*k1) * cos((M_PI/diameter)*(n2+0.5)*k2) * cos((M_PI/colors)*(n3+0.5)*k3);
			}
		}
	}

	#pragma omp parallel for
	for(long i=0; i<HW; i++)
	{
		const int x = i / height;
		const int y = i % height;		
		
		
		for(int k=0; k<coefficients; k++)
		{
			HT * const element_buffer = dct_buffer + k*window;
			HT value = 0;
			for(int c=0; c<colors; c++)
			{
				for(int w=0; w<window; w++)
				{
					const int dx = w / diameter - radius;
					const int dy = w % diameter - radius;

					const int xx = (x + dx) < 0 ? 0 : (x + dx) >= width ? width-1 : (x + dx);
					const int yy = (y + dy) < 0 ? 0 : (y + dy) >= height? height-1 : (y + dy);
						
					//value += (HT(Image(xx,yy,c))-127) * element_buffer[w];
					value += HT(Image(xx,yy,c)) * element_buffer[w+c*window];
				}
			}
			//value = k==0 ? value : value/2+127;
			HashImage(x,y,k) = value/window;
		}
					
	}	

	delete[] dct_buffer;
	delete[] zigzag_order;
}

//#define EXPERIMENTAL_WEIGHT

void mexFunction (int nout, mxArray* output[], int in, const mxArray* input[])
{
#ifndef _DEBUG
	omp_set_num_threads(std::max<int>(4, omp_get_max_threads()));
	omp_set_dynamic(std::max<int>(2, omp_get_max_threads()/2));
#endif
	
	if(in < 2 || in > 3 || nout != 1)
	{
		mexErrMsgTxt("USAGE: [Hash] = hash_image(Image, radius, <zigzag_coefficients>);");
    }
	
	const int radius = std::max(1, (int)mxGetScalar(input[1])); 	
	const int diameter = radius*2+1;
	const int window = diameter*diameter;
	const int coefficients = in > 2  ? (int)mxGetScalar(input[2]) : 0;
	const bool compression = in > 2 && coefficients > 0;

	if(compression)
	{
		if(mxGetClassID(input[0]) == mxUINT8_CLASS)
		{
			MexImage<unsigned char> Image(input[0]);
			if(coefficients >= window*Image.layers)
			{
				mexErrMsgTxt("Too large number of coefficients!");
			}
			const size_t dims[] = {(size_t)Image.height, (size_t)Image.width, (size_t)coefficients};
			output[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL); 	
			MexImage<float> HashImage(output[0]);

			prepareCompressedHashImage<unsigned char, float>(Image, HashImage, radius, coefficients);
		}
		else if(mxGetClassID(input[0]) == mxUINT16_CLASS)
		{
			MexImage<unsigned short> Image(input[0]);
			if(coefficients >= window*Image.layers)
			{
				mexErrMsgTxt("Too large number of coefficients!");
			}
			const size_t dims[] = {(size_t)Image.height, (size_t)Image.width, (size_t)coefficients};
			output[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL); 	
			MexImage<float> HashImage(output[0]);

			prepareCompressedHashImage<unsigned short, float>(Image, HashImage, radius, coefficients);		
		}
		else if(mxGetClassID(input[0]) == mxSINGLE_CLASS)
		{
			MexImage<float> Image(input[0]);
			if(coefficients >= window*Image.layers)
			{
				mexErrMsgTxt("Too large number of coefficients!");
			}
			const size_t dims[] = {(size_t)Image.height, (size_t)Image.width, (size_t)coefficients};
			output[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL); 	
			MexImage<float> HashImage(output[0]);

			prepareCompressedHashImage<float, float>(Image, HashImage, radius, coefficients);		
		}
		else if(mxGetClassID(input[0]) == mxDOUBLE_CLASS)
		{
			MexImage<double> Image(input[0]);
			if(coefficients >= window*Image.layers)
			{
				mexErrMsgTxt("Too large number of coefficients!");
			}
			const size_t dims[] = {(size_t)Image.height, (size_t)Image.width, (size_t)coefficients};
			output[0] = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL); 	
			MexImage<double> HashImage(output[0]);

			prepareCompressedHashImage<double, double>(Image, HashImage, radius, coefficients);		
		}
		else
		{
			mexErrMsgTxt("Unsupported Image datatype."); 
		}
	}
	else
	{
		if(mxGetClassID(input[0]) == mxUINT8_CLASS)
		{
			MexImage<unsigned char> Image(input[0]);
			const size_t dims[] = {(size_t)Image.height, (size_t)Image.width, (size_t)Image.layers*window};
			output[0] = mxCreateNumericArray(3, dims, mxUINT8_CLASS, mxREAL); 	
			MexImage<unsigned char> HashImage(output[0]);

			prepareHashImage<unsigned char>(Image, HashImage, radius);
		}
		else if(mxGetClassID(input[0]) == mxUINT16_CLASS)
		{
			MexImage<unsigned short> Image(input[0]);
			const size_t dims[] = {(size_t)Image.height, (size_t)Image.width, (size_t)Image.layers*window};
			output[0] = mxCreateNumericArray(3, dims, mxUINT16_CLASS, mxREAL); 	
			MexImage<unsigned short> HashImage(output[0]);

			prepareHashImage<unsigned short>(Image, HashImage, radius);		
		}
		else if(mxGetClassID(input[0]) == mxSINGLE_CLASS)
		{
			MexImage<float> Image(input[0]);
			const size_t dims[] = {(size_t)Image.height, (size_t)Image.width, (size_t)Image.layers*window};
			output[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL); 	
			MexImage<float> HashImage(output[0]);

			prepareHashImage<float>(Image, HashImage, radius);		
		}
		else if(mxGetClassID(input[0]) == mxDOUBLE_CLASS)
		{
			MexImage<double> Image(input[0]);
			const size_t dims[] = {(size_t)Image.height, (size_t)Image.width, (size_t)Image.layers*window};
			output[0] = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL); 	
			MexImage<double> HashImage(output[0]);

			prepareHashImage<double>(Image, HashImage, radius);		
		}
		else
		{
			mexErrMsgTxt("Unsupported Image datatype."); 
		}
	}

}