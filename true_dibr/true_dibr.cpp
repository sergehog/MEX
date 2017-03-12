/** True DIBR implementation (Gaussian Clouds Method)
*	@file true_dibr.cpp
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
#include <vector>
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
	if(in < 4 || in > 7 || nout != 3 || mxGetClassID(input[0]) != mxSINGLE_CLASS)
	{
		mexPrintf("True Depth Image Based Rendering (DIBR), using Gaussian Clouds method.\n");
		mexPrintf("USAGE: [I_virtual, Z_virtual] = true_dibr(P_virtual, cell(I_input), cell(Z_input), cell(P_input), radius, sigma_sqr, z_thr); \n");
		mexPrintf(", where: \n");
		mexPrintf("P_virtual - 4-by-4 matrix of virtual camera (of type SINGLE), \n");
		mexPrintf("cell(I_input) - image or a stack of images used as input views (of type SINGLE), \n");
		mexPrintf("cell(Z_input) - corresponding Depth Maps or a stack of them (of type SINGLE), \n");
		mexPrintf("cell(P_input) - corresponding input camera matrix or a stack of them (of type SINGLE), \n");
		mexPrintf("radius - radius of Gaussian Cloud (recomended value: 1 or 0) \n");
		mexPrintf("sigma_sqr - std. deviation of Gaussian Cloud (recomended value: 0.1 or lower) \n");
		mexPrintf("z_thr - Depth threshold value (0.0 or any positive number),  \n");
		//mexPrintf("fill_in - applies hole filling (0 - none, 1 - simple, 2 - experimental),  \n");
		mexErrMsgTxt("ERROR: wrong parameters!\n");
    }
	
	Matrix<4, 4> P_virtual (input[0]);
	std::vector<MexImage<float>*> Images;
	std::vector<MexImage<float>*> Depths;
	std::vector<Matrix<4, 4> *> P_mats;

	const size_t cameras = mxIsCell(input[1]) ? mxGetNumberOfElements (input[1]) : 1;
	const int width = mxIsCell(input[1]) ?  mxGetWidth(mxGetCell(input[1], 0)) : mxGetWidth(input[1]);
	const int height = mxIsCell(input[1]) ? mxGetHeight(mxGetCell(input[1], 0)) : mxGetHeight(input[1]);
	colors = mxIsCell(input[1]) ? mxGetLayers(mxGetCell(input[1], 0)) : mxGetLayers(input[1]);
	HW = width*height;
	
	// Validate user inputs:
	if(mxIsCell(input[1]))
	{
		if(!mxIsCell(input[2]) || mxGetNumberOfElements (input[2]) != cameras)
		{
			mexErrMsgTxt("cell(Z_input) must have same number of elements as cell(I_input) !\n");
		}

		if(!mxIsCell(input[3]) || mxGetNumberOfElements (input[3]) != cameras)
		{
			mexErrMsgTxt("cell(P_input) must have same number of elements as cell(I_input) !\n");
		}

		for(int i=0; i<cameras; i++)
		{
			if(mxGetClassID(mxGetCell(input[2], i)) != mxSINGLE_CLASS || mxGetClassID(mxGetCell(input[3], i)) != mxSINGLE_CLASS)
			{
				mexErrMsgTxt("All members of cells Z_input and P_input must be SINGLE-valued! \n");
			}

			MexImage<float> *Image;
			if(mxGetClassID(mxGetCell(input[1], i)) == mxSINGLE_CLASS)
			{
				Image = new MexImage<float>(mxGetCell(input[1], i));
			}
			else if(mxGetClassID(mxGetCell(input[1], i)) == mxUINT8_CLASS)
			{
				Image = new MexImage<float>(width, height, colors);
				unsigned char *data = (unsigned char *)mxGetData(mxGetCell(input[1], i));
				for(long i=0; i<Image->layer_size*colors; i++)
				{
					Image->data[i] = (float)data[i];
				}
			}
			else
			{
				mexErrMsgTxt("Members of cell I_input must be either SINGLE or UINT8 valued! \n");
			}


			
			MexImage<float> *Depth = new MexImage<float>(mxGetCell(input[2], i));

			//MexImage<float> *Image = new MexImage<float>(mxDuplicateArray(mxGetCell(input[1], i)));
			//MexImage<float> *Depth = new MexImage<float>(mxDuplicateArray(mxGetCell(input[2], i)));

			//MexImage<float> *Image = new MexImage<float>(width, height, colors);
			//Image->TryLoad(mxGetCell(input[1], i));

			//MexImage<float> *Depth = new MexImage<float>(width, height, 1);
			//Depth->TryLoad(mxGetCell(input[2], i));
			
			Matrix<4, 4> *P_input = new Matrix<4, 4> (mxGetCell(input[3], i));

			if(Image->width != width || Image->height != height || Image->layers != colors)
			{
				mexErrMsgTxt("All images in cell(I_input) must have same sizes!\n");
			}

			if(Depth->width != width || Depth->height != height || Depth->layers != 1)
			{
				mexErrMsgTxt("All images in cell(Z_input) must have same sizes!\n");
			}

			Images.push_back(Image);
			Depths.push_back(Depth);
			P_mats.push_back(P_input);
		}
	}
	else
	{
		MexImage<float> *Image;
		if(mxGetClassID(input[2]) != mxSINGLE_CLASS || mxGetClassID(input[3]) != mxSINGLE_CLASS)
		{
			mexErrMsgTxt("If cells are not used Z_input and P_input must be images of SINGLE type!\n");
		}
		
		if(mxGetClassID(input[1]) == mxSINGLE_CLASS)
		{
			Image = new MexImage<float>(input[1]);
		}
		else if(mxGetClassID(input[1]) == mxUINT8_CLASS)
		{
			Image = new MexImage<float>(width, height, colors);
			unsigned char *data = (unsigned char *)mxGetData(input[1]);
			for(long i=0; i<Image->layer_size*colors; i++)
			{
				Image->data[i] = (float)data[i];
			}
		}
		else
		{
			mexErrMsgTxt("I_input must be either SINGLE or UINT8 valued! \n");
		}

		MexImage<float> *Depth = new MexImage<float>(input[2]);
		
		if(Depth->height != height || Depth->width != width)
		{
			delete Image;
			delete Depth;

			mexErrMsgTxt("I_input and Z_input must be of the same size!");
		}

		Matrix<4, 4> *P_input = new Matrix<4, 4> (input[3]);		
		
		Images.push_back(Image);
		Depths.push_back(Depth);
		P_mats.push_back(P_input);
	}

	// optional function parameters
	const int radius = (in > 4) ? (int)mxGetScalar(input[4]) : 0;
	const float sigma = (in > 5) ? (float)mxGetScalar(input[5]) : 0.1f;
	const float zthr = (in > 6) ? (float)mxGetScalar(input[6]) : 0.f;
	//const int fill_in = (in > 7) ? (int)mxGetScalar(input[7]) : 0;
	const float nan = sqrt(-1.f);
	
	mexPrintf("cameras = %d \n", cameras);
	mexPrintf("width = %d, height = %d, colors = %d\n", width, height, colors);
	mexPrintf("radius = %d \n", radius);
	mexPrintf("sigma = %f \n", sigma);
	mexPrintf("zthr = %f \n", zthr);
	//mexPrintf("fill_in = %d \n", fill_in);

	size_t dims3d[] = {(size_t)height, (size_t)width, (size_t)colors};
	size_t dims2d[] = {(size_t)height, (size_t)width, 1};

	output[0] = mxCreateNumericArray(3, dims3d, mxSINGLE_CLASS, mxREAL);
	output[1] = mxCreateNumericArray(3, dims2d, mxSINGLE_CLASS, mxREAL);
	output[2] = mxCreateNumericArray(3, dims2d, mxSINGLE_CLASS, mxREAL);
	MexImage<float> Virtual(output[0]);
	MexImage<float> Zvirt(output[1]);
	MexImage<float> Weights(output[2]);
	//MexImage<float> Weights(Zvirt.width, Zvirt.height);
	Weights.setval(0.f);
	Zvirt.setval(nan);
	Virtual.setval(nan);

	// invert input camera matrixes
	for(size_t camera = 0; camera<cameras; camera++)
	{
		//mexPrintf("Camera Matrix %d: \n", camera);
		//P_mats.at(camera)->mexPrint();		
		invertMatrix4x4(P_mats.at(camera), P_mats.at(camera));
		//mexPrintf("Inverted Camera Matrix %d: \n", camera);
		//P_mats.at(camera)->mexPrint();
	}	

	Matrix<4, 1> camera_coords(1.f);
	
	//mexPrintf("3. \n");
	for(size_t camera = 0; camera<cameras; camera++)
	{

		//mexPrintf("3.%d \n", camera);

		MexImage<float> *Image = Images.at(camera);
		MexImage<float> *Zimage = Depths.at(camera);
		Matrix<4, 4> *P_invert = P_mats.at(camera);
		
		//mexPrintf("Image: width = %d, height = %d, colors = %d\n", Image->width, Image->height, Image->layers);
		//mexPrintf("Depth: width = %d, height = %d, colors = %d\n", Zimage->width, Zimage->height, Zimage->layers);
		//P_invert->mexPrint();
		//mexPrintf("colors = %f %f %f\n", Image->data[0], Image->data[HW], Image->data[HW*2]);
		//mexPrintf("z = %f \n", Zimage->data[0]);
		//#pragma omp parallel for 
		for(long i=0; i<HW; i++)
		{
			int x = i / height;
			int y = i % height;

			// ignore holes in input frame
			if(_isnan(Zimage->data[i]) || _isnan(Image->data[i]))
			{
				continue;
			}
			//mexPrintf("a");
			float zi = Zimage->data[i];
			camera_coords.data[0][0] = float(x)*zi;
			camera_coords.data[1][0] = float(y)*zi;
			camera_coords.data[2][0] = zi;
			camera_coords.data[3][0] = 1.f;
			
			Matrix<4, 1> world_coords =  P_invert->multiply<1>(camera_coords);
			Matrix<4, 1> virtual_coords = P_virtual.multiply<1>(world_coords);
			//mexPrintf("b");
			//if(i == 99)
			//{
			//	mexPrintf("x=%d & y=%d\n", x, y);
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
				
				if(xvv < 0 || xvv >= width || yvv < 0 || yvv >= height)
				{
					continue;
				}

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
								Virtual.data[index  + HW*c] = Image->data[i + HW*c]*weight;
							}
						}
						else if((currentZ-zthr) <= zv && zv <= (currentZ+zthr))
						{
							//Zvirt.data[index] += zv*weight;
							Weights.data[index] += weight;
							for(int c=0; c<colors; c++)
							{
								Virtual.data[index  + HW*c] += Image->data[i + HW*c]*weight;
							}
						}
					}

				}
			}	
		}


	}

	//mexPrintf("4. \n");

	//float z_min = Zvirt[0];
	//float z_max = Zvirt[0];

	for(long i=0; i<HW; i++)
	{
		float weight = Weights[i];
		if(!_isnan(Zvirt[i]) && weight > 0)
		{
			
			//Zvirt.data[i] /= weight;
			for(int c=0; c<colors; c++)
			{
				Virtual.data[i  + HW*c] /= weight;
			}
		}
		else
		{
			for(int c=0; c<colors; c++)
			{
				Virtual.data[i  + HW*c]  = nan;
			}
		}
		//else
		//{
		//	z_min = (Zvirt[i] < z_min) ? Zvirt[i] : z_min;
		//	z_max = (Zvirt[i] > z_max) ? Zvirt[i] : z_max;
		//}
	}

	//mexPrintf("5. \n");
	
	/*
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
	*/


	for(int i=0; i<cameras; i++)
	{
		delete Images.at(i);
		delete Depths.at(i);
		delete P_mats.at(i);
	}

	Images.clear();
	Depths.clear();
	P_mats.clear();

}



