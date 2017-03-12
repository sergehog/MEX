/** Simple 1D-Translation DIBR implementation (Gaussian Clouds Method)
*	@file translate_dibr.cpp
*	@date 01.06.2012
*	@author Sergey Smirnov <sergey.smirnov@tut.fi>
*/

#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}
#endif // __cplusplus
//#include "../common/defines.h"
#include "../common/meximage.h"
#include <vector>
#include <cmath>
#include <algorithm>

typedef unsigned char uint8;
using namespace mymex;

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")


//template<typename T> void render(T*, float*, T*, float*, bool*, size_t, size_t, size_t);

const float skip_delta = 3;
const unsigned skip_numb = 1;

void mexFunction (int nout, mxArray* output[], int in, const mxArray* input[])
{	
	if(in < 3 || in > 8 || nout != 2)
	{
		mexErrMsgTxt("USAGE: [I_v, Disp_v] = translate_dibr(position_v, cell(I_input), cell(Disp_input), positions, <radius, sigma_weight, z_thr>); ");
    }
		 
	const double position_v = (float) mxGetScalar(input[0]);

	std::vector<MexImage<float> *> Images;
	std::vector<MexImage<float> *> Depths;

	const int cameras = mxIsCell(input[1]) ? mxGetNumberOfElements (input[1]) : 1;
	const int width = mxIsCell(input[1]) ?  mxGetWidth(mxGetCell(input[1], 0)) : mxGetWidth(input[1]);
	const int height = mxIsCell(input[1]) ? mxGetHeight(mxGetCell(input[1], 0)) : mxGetHeight(input[1]);
	const int colors = mxIsCell(input[1]) ? mxGetLayers(mxGetCell(input[1], 0)) : mxGetLayers(input[1]);
	const long HW = width*height;
	

	double *positions;
	
	if(in > 3)
	{
		if(mxGetNumberOfElements (input[3]) != cameras)
		{
			mexErrMsgTxt("'positions' must contain input camera potitions relative to disparity scale!\n");
		}

		positions = (double*) mxGetData(input[3]);
	}
	else
	{
		positions = new double[cameras];
		for(int i=0; i<cameras; i++)
			positions[i] = (double) i;
	}
	
	// Validate user inputs:
	if(mxIsCell(input[1]))
	{
		if(!mxIsCell(input[2]) || mxGetNumberOfElements (input[2]) != cameras)
		{
			mexErrMsgTxt("cell(Disp_input) must have same number of elements as cell(I_input) !\n");
		}
		

		for(int i=0; i<cameras; i++)
		{			
			if(mxGetClassID(mxGetCell(input[2], i)) != mxSINGLE_CLASS)
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
		}
	}
	else
	{
		MexImage<float> *Image;
		if(mxGetClassID(input[2]) != mxSINGLE_CLASS )
		{
			mexErrMsgTxt("If cells are not used, Z_input must be of SINGLE type!\n");
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
		
		Images.push_back(Image);
		Depths.push_back(Depth);
		
	}

	// optional function parameters
	const int radius = (in > 4) ? (int)mxGetScalar(input[4]) : 0;
	const float sigma_distance  = (in > 5) ? (float)mxGetScalar(input[5]) : 0.1f;
	const float zthr  = (in > 6) ? (float)mxGetScalar(input[6]) : 1.f;	
	const bool background  = (in > 7) ? (bool)mxGetScalar(input[7]) : false;	
	const float nan = sqrt(-1.f);


	size_t dims[] = {(size_t)height, (size_t)width, (size_t)colors};
	size_t dims2d[] = {(size_t)height, (size_t)width, 1};

	output[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
	output[1] = mxCreateNumericArray(3, dims2d, mxSINGLE_CLASS, mxREAL);	

	MexImage<float> Virtual(output[0]);
	MexImage<float> DispOut(output[1]);
	MexImage<float> Weights(width, height);	
	
	Virtual.setval(nan);
	DispOut.setval(nan);
	Weights.setval(0.f);

	for(int camera=0; camera<cameras; camera++)
	{
		MexImage<float> *Image = Images.at(camera);
		MexImage<float> *Disparity = Depths.at(camera);
		// disaprity scaling factor to obtain desired view 
		float const rfactor = positions[camera] - position_v;  

		if(std::abs(rfactor) < 0.00001f)
		{
			for(long j=0; j<HW; j++)
			{
				//if(_isnan(DispOut[j]))
				{
					DispOut.data[j] = Disparity->data[j];
					Weights.data[j] = 1.f;
					for(int c=0; c<colors; c++)
					{
						Virtual.data[j + c*HW] = Image->data[j + c*HW];						
					}
				}
				//else
				//{

				//}
				
			}
			continue;
		}

		float mindisp = Disparity->data[0];	
		float maxdisp = Disparity->data[0];

		for(int i=1; i<HW; i++)
		{
			float disp = Disparity->data[i];
			mindisp = disp < mindisp ? disp : mindisp;
			maxdisp = disp > maxdisp ? disp : maxdisp;
		}

		//#pragma omp parallel for
		for(int y=0; y<height; y++)
		{
			for(int xt=0; xt<width; xt++)
			{
				int x = rfactor >= 0 ? xt : width-xt-1;

				int index = Disparity->Index(x,y);
				float disp = Disparity->data[index] * rfactor;

				if(_isnan(Disparity->data[index]))
					continue;

				float xrf = (float)x + disp;
				int xr = mymex::round(xrf);
			
				for(int dx=-radius; dx<=radius; dx++)
				{
					int xrr = xr + dx;					

					if(xrr >= 0 && xrr < width)
					{
						int indexRR = Disparity->Index(xrr, y);
						float distance = std::abs(xrf - xrr);
						float weight = std::exp(- (float)distance/sigma_distance);
						float currentW = Weights[indexRR];
						float currentZ = DispOut[indexRR];
						float zv = disp/rfactor;

						
						if(_isnan(currentZ) || (!background && zv > (currentZ + zthr))  || (background && zv < (currentZ - zthr)))
						{							
							DispOut.data[indexRR] = zv;
							Weights.data[indexRR] = weight;
							for(int c=0; c<colors; c++)
							{
								Virtual.data[indexRR  + HW*c] = Image->data[index + HW*c]*weight;
							}
						}
						else if((currentZ-zthr) <= zv && zv <= (currentZ+zthr))
						{
							//Zvirt.data[index] += zv*weight;
							Weights.data[indexRR] += weight;
							for(int c=0; c<colors; c++)
							{
								Virtual.data[indexRR  + HW*c] += Image->data[index + HW*c]*weight;
							}
						}
						

					}
				}
			}

		}
	}

	
	for(long i=0; i<HW; i++)
	{
		float weight = Weights[i];
		if(!_isnan(DispOut[i]) && weight > 0)
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

	if(in <= 3)
	{
		delete[] positions;
	}

}

