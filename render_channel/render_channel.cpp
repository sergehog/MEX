/** DIBR implementation
*	@file render_channel.cpp
*	@date 01.06.2012
*	@author Sergey Smirnov <sergey.smirnov@tut.fi>
*/


#ifdef __cplusplus 
extern "C" 
{
#include "mex.h"
}
#endif // __cplusplus

#include "../common/meximage.h"
#include "../common/mymath.h"
#include <float.h>
//#include <cmath>
//#include <armadillo>

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")

using namespace mymex;
using namespace mymath;

static long HW=0;
static int colors=0;

void replaceColorValue(MexImage<float> &Virtual, long index_to, long index_from)
{
	for(int c=0; c<colors; c++)
	{
		Virtual.data[index_to + HW*c] = Virtual[index_from + HW*c];
	}
}

void clearColorValue(MexImage<float> &Virtual, long index_to)
{
	for(int c=0; c<colors; c++)
	{
		Virtual.data[index_to + HW*c] = 0;
	}
}

void aggregateColorValue(MexImage<float> &Virtual, long index_to, long index_from, float weight)
{
	for(int c=0; c<colors; c++)
	{
		Virtual.data[index_to + HW*c] += Virtual[index_from + HW*c]*weight;
	}
}

void normalizeColorValue(MexImage<float> &Virtual, long index_to, float weights)
{
	for(int c=0; c<colors; c++)
	{
		Virtual.data[index_to + HW*c] /= weights;
	}
}



void mexFunction (int nout, mxArray* output[], int in, const mxArray* input[])
{	
	if(in < 8 || in > 8 && (in-8)%3!=0 || nout < 1 || nout > 2 || mxGetClassID(input[0]) != mxSINGLE_CLASS)
	{
		mexErrMsgTxt("USAGE: [single(Virtual), Z_virtual] = render_channel(single(P_virtual), radius, sigma, z_thr, fill_in, single(Image1), single(Z_image1), single(P_image1), <I2, Z2, P2, ...>); \n"); 
    }
	
	Matrix<4, 4> P_virtual (input[0]);
	const int radius = (int)mxGetScalar(input[1]);
	const float sigma = (float)mxGetScalar(input[2]);
	const float zthr = (float)mxGetScalar(input[3]);
	const int fill_in = (int)mxGetScalar(input[4]);
	const float nan = sqrt(-1.f);

	MexImage<float> Image1(input[5]);
	//MexImage<float> Zimage1(input[6]);
	//Matrix<4, 4> P_camera (input[7]);
	//Matrix<4, 4> P_invert = P_camera.invert();
	const int width = Image1.width;
	const int height = Image1.height;
	/*const int */colors = Image1.layers;
	/*const long */HW = Image1.layer_size;	
	

	
	size_t dims3d[] = {height, width, colors};
	size_t dims2d[] = {height, width, 1};

	output[0] = mxCreateNumericArray(3, dims3d, mxSINGLE_CLASS, mxREAL);
	output[1] = mxCreateNumericArray(3, dims2d, mxSINGLE_CLASS, mxREAL);
	MexImage<float> Virtual(output[0]);
	MexImage<float> Zvirt(output[1]);
	MexImage<float> Weights(Zvirt.width, Zvirt.height);
	Weights.setval(0.f);
	Zvirt.setval(nan);    	

	Matrix<4, 1> camera_coords(1.f);
	const int cameras = (int)floor((in-5)/3.f);
	mexPrintf("number of cameras: %d\n", cameras);	 

	for(int camera=0; camera<cameras; camera++)
	{
		int inputN = camera*3 + 5;		

		mexPrintf("camera %d, inputN=%d\n", camera, inputN);
		
		if(mxGetClassID(input[inputN]) != mxSINGLE_CLASS || mxGetClassID(input[inputN+1]) != mxSINGLE_CLASS || mxGetClassID(input[inputN+2]) != mxSINGLE_CLASS )
		{
			mexErrMsgTxt("Both input parameters must be of type SINGLE"); 
		}

		MexImage<float> Image(input[inputN]);
		MexImage<float> Zimage(input[inputN+1]);
		Matrix<4, 4> P_camera (input[inputN+2]);

		if(Zimage.width != width || Zimage.height != height || Image.width != width || Image.height != height)
		{
			mexErrMsgTxt("All input Images and Z-maps must have same sizes!");
		}

	}

	for(int camera=0; camera<cameras; camera++)
	{
		int inputN = camera*3 + 5;		

		MexImage<float> Image(input[inputN]);
		MexImage<float> Zimage(input[inputN+1]);
		Matrix<4, 4> P_camera (input[inputN+2]);
		Matrix<4, 4> P_invert(input[inputN+2]);// = P_camera.invert();

		for(long i=0; i<HW; i++)
		{
			int x = i / height;
			int y = i % height;

			if(_isnan(Zimage[i]))
			{
				continue;
			}

			float zi = Zimage[i];
			camera_coords.data[0][0] = float(x)*zi;
			camera_coords.data[1][0] = float(y)*zi;
			camera_coords.data[2][0] = zi;
			camera_coords.data[3][0] = 1.f;
			
			Matrix<4, 1> world_coords =  P_invert.multiply<1>(camera_coords);
			Matrix<4, 1> virtual_coords = P_virtual.multiply<1>(world_coords);
		
			//if(i == 99)
			//{
			//	mexPrintf("x==0 & y==0\n");
			//	mexPrintf("camera_coords\n");
			//	camera_coords.mexPrint();
			//	mexPrintf("world_coords\n");
			//	world_coords.mexPrint();
			//	mexPrintf("virtual_coords\n");
			//	virtual_coords.mexPrint();
			//}

			float zv = virtual_coords.data[2][0];
			if(zv > 0)
			{
				float xv = virtual_coords.data[0][0]/zv;
				float yv = virtual_coords.data[1][0]/zv;

				int xvv = round(xv);
				int yvv = round(yv);

				for(int xx=xvv-radius; xx<=xvv+radius; xx++)
				{
					if(xx<0 || xx>=width)
						continue;
					for(int yy=yvv-radius; yy<=yvv+radius; yy++)
					{
						if(yy<0 || yy>=height)
							continue;
						long index = Zvirt.Index(xx, yy);

						float distance = sqrt((xx-xv)*(xx-xv) + (yy-yv)*(yy-yv));
						float weight = exp(-distance/sigma);
						float currentW = Weights[index];
						//float currentZ = currentW > 0.f ? Zvirt[index]/currentW : nan;
						float currentZ = Zvirt[index];

						//if((_isnan(currentZ) || zv < (currentZ-zthr)) && weight > 0.0001)
						if(_isnan(currentZ) || zv < (currentZ-zthr))
						{
							//Zvirt.data[index] = zv*weight;
							Zvirt.data[index] = zv;
							Weights.data[index] = weight;
							for(int c=0; c<colors; c++)
							{
								Virtual.data[index  + HW*c] = Image[i + HW*c]*weight;
							}
						}
						else if((currentZ-zthr) <= zv && zv <= (currentZ+zthr))
						{
							//Zvirt.data[index] += zv*weight;
							Weights.data[index] += weight;
							for(int c=0; c<colors; c++)
							{
								Virtual.data[index  + HW*c] += Image[i + HW*c]*weight;
							}
						}
					}

				}
			}	
		}


	}


	float z_min = Zvirt[0];
	float z_max = Zvirt[0];

	for(long i=0; i<HW; i++)
	{
		if(!_isnan(Zvirt[i]))
		{
			float weight = Weights[i];
			//Zvirt.data[i] /= weight;
			for(int c=0; c<colors; c++)
			{
				Virtual.data[i  + HW*c] /= weight;
			}
		}
		else
		{
			z_min = (Zvirt[i] < z_min) ? Zvirt[i] : z_min;
			z_max = (Zvirt[i] > z_max) ? Zvirt[i] : z_max;
		}
	}
	//float z_diff = z_max-z_min;

	// fill_in == 0   - no hole filling

	if(fill_in == 1) // simple hole filling (4-points averaging)
	{
		for(long i=0; i<HW; i++)
		{
			if(_isnan(Zvirt[i]))
			{
				int x = i / height;
				int y = i % height;
				clearColorValue(Virtual, i);

				float weights = 0;

				for(int x_left=x-1; x_left>=0; x_left--)
				{
					long index_from = Zvirt.Index(x_left, y);
					if(!_isnan(Zvirt[index_from]))
					{
						float weight = 1.0/(x-x_left);
						aggregateColorValue(Virtual, i, index_from, weight);
						weights += weight;
						break;
					}
				}
				for(int x_right=x+1; x_right<width; x_right++)
				{
					long index_from = Zvirt.Index(x_right, y);
					if(!_isnan(Zvirt[index_from]))
					{
						float weight = 1.0/(x_right-x);
						aggregateColorValue(Virtual, i, index_from, weight);
						weights += weight;
						break;
					}
				}
				for(int y_up=y-1; y_up>=0; y_up--)
				{
					long index_from = Zvirt.Index(x, y_up);
					if(!_isnan(Zvirt[index_from]))
					{
						float weight = 1.0/(y-y_up);
						aggregateColorValue(Virtual, i, index_from, weight);
						weights += weight;
						break;
					}
				}
				for(int y_down=y+1; y_down<height; y_down++)
				{
					long index_from = Zvirt.Index(x, y_down);
					if(!_isnan(Zvirt[index_from]))
					{
						float weight = 1.0/(y_down-y);
						aggregateColorValue(Virtual, i, index_from, weight);
						weights += weight;
						break;
					}
				}

				if(weights > 0)
				{
					normalizeColorValue(Virtual, i, weights);
				}
			}
		}
	}
	else if(fill_in == 2) // experimental filling
	{
		for(long i=0; i<HW; i++)
		{
			if(_isnan(Zvirt[i]))
			{
				int x = i / height;
				int y = i % height;
				clearColorValue(Virtual, i);

				float weights = 0;

				for(int x_left=x-1; x_left>=0; x_left--)
				{
					long index_from = Zvirt.Index(x_left, y);
					if(!_isnan(Zvirt[index_from]))
					{
						float weight = Zvirt[index_from]/(x-x_left);
						aggregateColorValue(Virtual, i, index_from, weight);
						weights += weight;
						break;
					}
				}
				for(int x_right=x+1; x_right<width; x_right++)
				{
					long index_from = Zvirt.Index(x_right, y);
					if(!_isnan(Zvirt[index_from]))
					{
						float weight = Zvirt[index_from]/(x_right-x);
						aggregateColorValue(Virtual, i, index_from, weight);
						weights += weight;
						break;
					}
				}
				for(int y_up=y-1; y_up>=0; y_up--)
				{
					long index_from = Zvirt.Index(x, y_up);
					if(!_isnan(Zvirt[index_from]))
					{
						float weight = Zvirt[index_from]/(y-y_up);
						aggregateColorValue(Virtual, i, index_from, weight);
						weights += weight;
						break;
					}
				}
				for(int y_down=y+1; y_down<height; y_down++)
				{
					long index_from = Zvirt.Index(x, y_down);
					if(!_isnan(Zvirt[index_from]))
					{
						float weight = Zvirt[index_from]/(y_down-y);
						aggregateColorValue(Virtual, i, index_from, weight);
						weights += weight;
						break;
					}
				}

				if(weights > 0)
				{
					normalizeColorValue(Virtual, i, weights);
				}
			}
		}
	}
	else if(fill_in == 3)
	{
		for(long i=0; i<HW; i++)
		{
			if(_isnan(Zvirt[i]))
			{
				int x = i / height;
				int y = i % height;
				clearColorValue(Virtual, i);

				float weights = 0;

				for(int x_left=x-1; x_left>=0; x_left--)
				{
					long index_from = Zvirt.Index(x_left, y);
					if(!_isnan(Zvirt[index_from]))
					{
						float weight = Zvirt[index_from]/(x-x_left);
						aggregateColorValue(Virtual, i, index_from, weight);
						weights += weight;
						break;
					}
				}
				for(int x_right=x+1; x_right<width; x_right++)
				{
					long index_from = Zvirt.Index(x_right, y);
					if(!_isnan(Zvirt[index_from]))
					{
						float weight = Zvirt[index_from]/(x_right-x);
						aggregateColorValue(Virtual, i, index_from, weight);
						weights += weight;
						break;
					}
				}
				for(int y_up=y-1; y_up>=0; y_up--)
				{
					long index_from = Zvirt.Index(x, y_up);
					if(!_isnan(Zvirt[index_from]))
					{
						float weight = Zvirt[index_from]/(y-y_up);
						aggregateColorValue(Virtual, i, index_from, weight);
						weights += weight;
						break;
					}
				}
				for(int y_down=y+1; y_down<height; y_down++)
				{
					long index_from = Zvirt.Index(x, y_down);
					if(!_isnan(Zvirt[index_from]))
					{
						float weight = Zvirt[index_from]/(y_down-y);
						aggregateColorValue(Virtual, i, index_from, weight);
						weights += weight;
						break;
					}
				}

				if(weights > 0)
				{
					normalizeColorValue(Virtual, i, weights);
				}
			}
		}
	}	
}



