/** Simple directionsl hole filling algorithm. 
*	@file hole_filling.cpp
*	@date 02.07.2012
*	@author Sergey Smirnov <sergey.smirnov@tut.fi>
*/

#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}
#endif // __cplusplus

#include "../common/meximage.h"
#include <float.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <omp.h>

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")

using namespace mymex;

#define isnan _isnan

void clearColorValue(MexImage<float> &Virtual, long index_to)
{
	const int colors = Virtual.layers;
	const long HW = Virtual.layer_size;
	for(int c=0; c<colors; c++)
	{
		Virtual.data[index_to + HW*c] = 0;
	}
}

void aggregateColorValue(MexImage<float> &Virtual, long index_to, long index_from, float weight)
{
	const int colors = Virtual.layers;
	const long HW = Virtual.layer_size;
	for(int c=0; c<colors; c++)
	{
		Virtual.data[index_to + HW*c] += Virtual[index_from + HW*c]*weight;
	}
}

void normalizeColorValue(MexImage<float> &Virtual, long index_to, float weights)
{
	const int colors = Virtual.layers;
	const long HW = Virtual.layer_size;
	for(int c=0; c<colors; c++)
	{
		Virtual.data[index_to + HW*c] /= weights;
	}
}

void mexFunction (int nout, mxArray* output[], int in, const mxArray* input[])
{	
	omp_set_num_threads(std::max(4, omp_get_num_threads())/2);
	omp_set_dynamic(0);

	if(in < 2 || in > 6 || nout != 2 || mxGetClassID(input[0]) != mxSINGLE_CLASS || mxGetClassID(input[1]) != mxSINGLE_CLASS )
	{
		mexPrintf("USAGE: [Image_hat, Disp_hat] = hole_filling(Image, Disparity, <algorithm, direction, radius, maxdisp>);\n");
		mexPrintf("Image - SINGLE-valued (possibly color) image with NaN values, threated as holes \n");
		mexPrintf("Disparity - SINGLE-valued 1-layer image with NaN values, threated as holes \n");
		mexPrintf("algorithm - 1 (default): averaging, 2: median, 3: multi-sample, ... \n");
		mexPrintf("direction - 0 (default): no direction, 1: left-to-right, -1: right-to-left \n");
		mexPrintf("radius - radius of processing window (default: 2)\n");
		mexPrintf("maxdisp - if direction != 0, process last N roows in opposite direction (def: 0) \n");		
		mexErrMsgTxt("Wrong input parameters!");
    }

	MexImage<float> Signal(input[0]);
	MexImage<float> Disparity(input[1]);
	const int height = Signal.height;
	const int width = Signal.width;
	const int layers = Signal.layers;
	const long HW = Signal.layer_size;	
	const float nan = sqrt(-1.f);	
		
	const unsigned algorithm = (in > 2) ? (unsigned)mxGetScalar(input[2]) : 0;	
	const int direction = (in > 3) ? (int)mxGetScalar(input[3]) : 0; 
	const int radius = (in > 4) ? (int)mxGetScalar(input[4]) : 2;	
	const int offset = (in > 5) ? std::abs((int)mxGetScalar(input[5])) : 0;	
	
	size_t dimsI[] = {(size_t)height, (size_t)width, (size_t)layers};
	size_t dims[] = {(size_t)height, (size_t)width, 1};

	output[0] = mxCreateNumericArray(3, dimsI, mxSINGLE_CLASS, mxREAL);
	output[1] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
	MexImage<float> Filled(output[0]);
	MexImage<float> FilledDisp(output[1]);

	
	float maxdisp = Disparity[0];
	float mindisp = Disparity[0];

	// pre-fill output image
	#pragma omp parallel for 
	for(long i=0; i<HW; i++)
	{
		bool ishole = isnan(Disparity[i]);

		for(int c=0; c<layers; c++)
		{
			ishole |= isnan(Signal[i+c*HW]);
		}

		for(int c=0; c<layers; c++)
		{
			Filled.data[i+c*HW] = ishole ? nan : Signal[i+c*HW];
		}
		
		FilledDisp.data[i] = ishole ? nan : Disparity[i];
		
		if(!ishole)
		{
			#pragma omp critical
			maxdisp = isnan(maxdisp) ? Disparity[i] : (Disparity[i] > maxdisp ? Disparity[i] : maxdisp);
		
			#pragma omp critical
			mindisp = isnan(mindisp) ? Disparity[i] : (Disparity[i] < mindisp? Disparity[i] : mindisp);
		}

	}

	maxdisp = ceil(maxdisp);
	mindisp = floor(mindisp);	
	
	if(algorithm == 0) // no filling (left just for clarity :-))
	{

	}
	else if(algorithm == 1) // directional prediction with depth-weighted average 
	{		
		float *values = new float[layers];
				
		for(long i=0; i<HW; i++)
		{		
			int xk = i / height;
			int yk = i % height;				
			int x, y, offset_direction = 0;

			if(direction > 0)
			{
				x = xk < offset ? offset-xk-1 : xk;
				offset_direction = xk < offset ? -direction : direction;
				y = xk % 2 ? yk : height - yk - 1; // snake-like trajectory by y-axis
			}
			else if(direction < 0)
			{
				x = xk < offset ? width-offset+xk : width - xk - 1;
				offset_direction = xk < offset ? -direction : direction;
				y = xk % 2 ? yk : height - yk - 1; // snake-like trajectory by y-axis
			}
			else
			{
				x = i % width; 
				y = i / width; 
				x = y % 2 ? x : width - x - 1;// snake-like trajectory by x-axis
				offset_direction = 0;
				//x = yk % 2 ? xk : width - xk - 1; 
				//y = yk; // up-to-down
			}
						
			long index = Filled.Index(x, y);

			if(_isnan(FilledDisp[index]))
			{
				float weights = 0;
				float disparity = 0;
				for(int c=0; c<layers; c++)
				{
					values[c] = 0.f;
				}
			
				for(int dx=-radius; dx<=radius; dx++)
				{
					int xd = x + dx;
				
					if(offset_direction > 1 && dx > 0 || offset_direction < -1 && dx < 0)
						continue;

					if(xd<0 || xd>=width)
						continue;

					for(int dy=-radius; dy<=radius; dy++)
					{
						int yd = y + dy;

						if(yd<0 || yd>=height)
							continue;
						long indexd = Filled.Index(xd, yd);

						if(!_isnan(FilledDisp[indexd]))
						{
							//float weightT = _isnan(Disparity[indexd]) ? 1 : 5;
							float weightZ = 1-abs(FilledDisp[indexd]-mindisp+1)/(maxdisp-mindisp);
							//float weightS = 1./(abs(dx) + abs(dy) + 1);
							//float weight = weightT * weightZ * weightS;

							float weight = weightZ;

							weights += weight;
							disparity += FilledDisp[indexd]*weight;
							#pragma omp parallel for
							for(int c=0; c<layers; c++)
							{
								values[c] += Filled[indexd + c*HW]*weight;
							}
						}
					}
				}

				if(weights > 0.f)
				{
					FilledDisp.data[index] = disparity / weights;

					#pragma omp parallel for
					for(int c=0; c<layers; c++)
					{
						Filled.data[index + c*HW] = values[c] / weights;
					}
				}
			}
		}
	
		delete[] values;
	}
	else if(algorithm == 2) // directional prediction with median
	{
		std::vector<float> *medians = new std::vector<float>[layers];
		//#pragma omp parallel for
		for(long i=0; i<HW; i++)
		{		
			int xk = i / height;
			int yk = i % height;				
			int x, y, offset_direction = 0;

			if(direction > 0)
			{
				x = xk < offset ? offset-xk-1 : xk;
				offset_direction = xk < offset ? -direction : direction;
				y = xk % 2 ? yk : height - yk - 1; // snake-like trajectory by y-axis
			}
			else if(direction < 0)
			{
				x = xk < offset ? width-offset+xk : width - xk - 1;
				offset_direction = xk < offset ? -direction : direction;
				y = xk % 2 ? yk : height - yk - 1; // snake-like trajectory by y-axis
			}
			else
			{
				x = i % width; 
				y = i / width; 
				x = y % 2 ? x : width - x - 1;// snake-like trajectory by x-axis
				offset_direction = 0;
				//x = yk % 2 ? xk : width - xk - 1; // snake-like trajectory by x-axis
				//y = yk; // up-to-down
			}
						
			long index = Filled.Index(x, y);

			if(_isnan(Filled[index]))
			{
				int number = 0;
				#pragma omp parallel for
				for(int c=0; c<layers; c++)
				{
					medians[c].clear();
				}
			
				for(int dx=-radius; dx<=radius; dx++)
				{
					int xd = x + dx;
				
					if(offset_direction > 1 && dx > 0 || offset_direction < -1 && dx < 0)
						continue;

					if(xd<0 || xd>=width)
						continue;

					for(int dy=-radius; dy<=radius; dy++)
					{
						int yd = y + dy;

						if(yd<0 || yd>=height)
							continue;
						long indexd = Filled.Index(xd, yd);

						if(!_isnan(Filled[indexd]))
						{
							number ++;
							for(int c=0; c<layers; c++)
							{
								medians[c].push_back(Filled[indexd + c*HW]);
							}
						}
					}
				}

				if(number > 2)
				{
					#pragma omp parallel for
					for(int c=0; c<layers; c++)
					{
						std::sort(medians[c].begin(), medians[c].end());
						Filled.data[index + c*HW] = medians[c].at(number/2);
					}
				}
				else if(number > 0)
				{					
					#pragma omp parallel for
					for(int c=0; c<layers; c++)
					{
						Filled.data[index + c*HW] = 0;
						for(int n=0; n<number; n++)
						{
							Filled.data[index + c*HW] += medians[c].at(n)/number;
						}
					}
				}
			}
		}
		#pragma omp parallel for
		for(int c=0; c<layers; c++)
		{
			medians[c].clear();
		}
		delete[] medians;
	}
	else if(algorithm == 3) // 4-samples depth-weighted averaging
	{		
		#pragma omp parallel for
		for(long i=0; i<HW; i++)
		{
			if(_isnan(Disparity[i]))
			{
				int x = i / height;
				int y = i % height;
				clearColorValue(Filled, i);

				float weights = 0;

				for(int x_left=x-1; x_left>=0; x_left--)
				{
					long index_from = Disparity.Index(x_left, y);
					if(!_isnan(Disparity[index_from]))
					{
						float weightD = std::abs(maxdisp-Disparity[index_from])/(maxdisp-mindisp);
						float weightS = 1/(x-x_left);
						float weight = weightD*weightD*weightS;
						//float weight = 1.0/((x-x_left)*Disparity[index_from]);
						aggregateColorValue(Filled, i, index_from, weight);
						weights += weight;
						break;
					}
				}
				for(int x_right=x+1; x_right<width; x_right++)
				{
					long index_from = Disparity.Index(x_right, y);
					if(!_isnan(Disparity[index_from]))
					{
						float weightD = std::abs(maxdisp-Disparity[index_from])/(maxdisp-mindisp);
						float weight = weightD*weightD/(x_right-x);
						//float weight = 1.0/((x_right-x)*Disparity[index_from]);
						aggregateColorValue(Filled, i, index_from, weight);
						weights += weight;
						break;
					}
				}
				for(int y_up=y-1; y_up>=0; y_up--)
				{
					long index_from = Disparity.Index(x, y_up);
					if(!_isnan(Disparity[index_from]))
					{
						float weightD = std::abs(maxdisp-Disparity[index_from])/(maxdisp-mindisp);
						float weight = weightD*weightD/(y-y_up);
						//float weight = 1.0/((y-y_up)*Disparity[index_from]);
						aggregateColorValue(Filled, i, index_from, weight);
						weights += weight;
						break;
					}
				}
				for(int y_down=y+1; y_down<height; y_down++)
				{
					long index_from = Disparity.Index(x, y_down);
					if(!_isnan(Disparity[index_from]))
					{
						float weightD = std::abs(maxdisp-Disparity[index_from])/(maxdisp-mindisp);
						float weight = weightD*weightD/(y_down-y);
						//float weight = 1.0/((y_down-y)*Disparity[index_from]);
						aggregateColorValue(Filled, i, index_from, weight);
						weights += weight;
						break;
					}
				}

				if(weights > 0.f)
				{
					normalizeColorValue(Filled, i, weights);
				}
			}
		}	
	}
	else if(algorithm == 4) // 8-samples depth-weighted averaging
	{		
		#pragma omp parallel for
		for(long i=0; i<HW; i++)
		{
			if(isnan(Disparity[i]))
			{
				int x = i / height;
				int y = i % height;
				clearColorValue(Filled, i);

				float weights = 0;
				int j = 1;

				bool l_done = false;
				bool r_done = false;
				bool u_done = false;
				bool d_done = false;

				bool lu_done = false;
				bool ru_done = false;
				bool ld_done = false;
				bool rd_done = false;

				bool done = l_done && r_done && u_done && d_done && lu_done && ru_done && ld_done && rd_done;
				while (! done)
				{
					if(!l_done && (x-j) >= 0)
					{
						long index_i = Disparity.Index(x-j, y);
						if(!isnan(Disparity[index_i]))
						{
							float weightD = std::abs(maxdisp-Disparity[index_i])/(maxdisp-mindisp);
							float weight = weightD/j;
							aggregateColorValue(Filled, i, index_i, weight);
							weights += weight;
							l_done = true;
						}
					}
					else
					{
						l_done = true;
					}

					if(!r_done && (x+j) < width)
					{
						long index_i = Disparity.Index(x+j, y);
						if(!isnan(Disparity[index_i]))
						{
							float weightD = std::abs(maxdisp-Disparity[index_i])/(maxdisp-mindisp);
							float weight = weightD/(float)j;
							aggregateColorValue(Filled, i, index_i, weight);
							weights += weight;
							r_done = true;
						}
					}
					else
					{
						r_done = true;
					}
					
					if(!u_done && (y-j) >= 0)
					{
						long index_i = Disparity.Index(x, y-j);
						if(!isnan(Disparity[index_i]))
						{
							float weightD = std::abs(maxdisp-Disparity[index_i])/(maxdisp-mindisp);
							float weight = weightD/(float)j;
							aggregateColorValue(Filled, i, index_i, weight);
							weights += weight;
							u_done = true;
						}
					}
					else
					{
						u_done = true;
					}

					if(!d_done && (y+j) < height)
					{
						long index_i = Disparity.Index(x, y+j);
						if(!isnan(Disparity[index_i]))
						{
							float weightD = std::abs(maxdisp-Disparity[index_i])/(maxdisp-mindisp);
							float weight = weightD/(float)j;
							aggregateColorValue(Filled, i, index_i, weight);
							weights += weight;
							d_done = true;
						}
					}
					else
					{
						d_done = true;
					}

					if(!lu_done && (x-j) >= 0 && (y-j) >= 0)
					{
						long index_i = Disparity.Index(x-j, y-j);
						if(!isnan(Disparity[index_i]))
						{
							float weightD = std::abs(maxdisp-Disparity[index_i])/(maxdisp-mindisp);
							float weight = weightD/(float)j;
							aggregateColorValue(Filled, i, index_i, weight);
							weights += weight;
							lu_done = true;
						}
					}
					else
					{
						lu_done = true;
					}

					if(!ru_done && (x+j) < width && (y-j) >= 0)
					{
						long index_i = Disparity.Index(x+j, y-j);
						if(!isnan(Disparity[index_i]))
						{
							float weightD = std::abs(maxdisp-Disparity[index_i])/(maxdisp-mindisp);
							float weight = weightD/(float)j;
							aggregateColorValue(Filled, i, index_i, weight);
							weights += weight;
							ru_done = true;
						}
					}
					else
					{
						ru_done = true;
					}

					if(!ld_done && (x-j) >= 0 && (y+j) < height)
					{
						long index_i = Disparity.Index(x-j, y+j);
						if(!isnan(Disparity[index_i]))
						{
							float weightD = std::abs(maxdisp-Disparity[index_i])/(maxdisp-mindisp);
							float weight = weightD/(float)j;
							aggregateColorValue(Filled, i, index_i, weight);
							weights += weight;
							ld_done = true;
						}
					}
					else
					{
						ld_done = true;
					}

					if(!rd_done && (x+j) < width && (y+j) < height)
					{
						long index_i = Disparity.Index(x+j, y+j);
						if(!isnan(Disparity[index_i]))
						{
							float weightD = std::abs(maxdisp-Disparity[index_i])/(maxdisp-mindisp);
							float weight = weightD/(float)j;
							aggregateColorValue(Filled, i, index_i, weight);
							weights += weight;
							rd_done = true;
						}
					}
					else
					{
						rd_done = true;
					}					


					j = j + 1;
					done = l_done && r_done && u_done && d_done && lu_done && ru_done && ld_done && rd_done;
				}


				if(weights > 0.f)
				{
					normalizeColorValue(Filled, i, weights);
				}
			}
		}	
	}
	else if(algorithm == 5) // 8-samples weighted averaging
	{		
		#pragma omp parallel for
		for(long i=0; i<HW; i++)
		{
			if(isnan(Disparity[i]))
			{
				int x = i / height;
				int y = i % height;
				clearColorValue(Filled, i);

				float weights = 0;
				int j = 1;

				bool l_done = false;
				bool r_done = false;
				bool u_done = false;
				bool d_done = false;

				bool lu_done = false;
				bool ru_done = false;
				bool ld_done = false;
				bool rd_done = false;

				bool done = l_done && r_done && u_done && d_done && lu_done && ru_done && ld_done && rd_done;
				while (! done)
				{
					if(!l_done && (x-j) >= 0)
					{
						long index_i = Disparity.Index(x-j, y);
						if(!isnan(Disparity[index_i]))
						{
							float weight = Disparity[index_i]/j;
							
							aggregateColorValue(Filled, i, index_i, weight);
							weights += weight;
							l_done = true;
						}
					}
					else
					{
						l_done = true;
					}

					if(!r_done && (x+j) < width)
					{
						long index_i = Disparity.Index(x+j, y);
						if(!isnan(Disparity[index_i]))
						{
							float weight = Disparity[index_i]/j;
							;
							aggregateColorValue(Filled, i, index_i, weight);
							weights += weight;
							r_done = true;
						}
					}
					else
					{
						r_done = true;
					}
					
					if(!u_done && (y-j) >= 0)
					{
						long index_i = Disparity.Index(x, y-j);
						if(!isnan(Disparity[index_i]))
						{
							float weight = Disparity[index_i]/j;
							
							aggregateColorValue(Filled, i, index_i, weight);
							weights += weight;
							u_done = true;
						}
					}
					else
					{
						u_done = true;
					}

					if(!d_done && (y+j) < height)
					{
						long index_i = Disparity.Index(x, y+j);
						if(!isnan(Disparity[index_i]))
						{
							float weight = Disparity[index_i]/j;
							
							aggregateColorValue(Filled, i, index_i, weight);
							weights += weight;
							d_done = true;
						}
					}
					else
					{
						d_done = true;
					}

					if(!lu_done && (x-j) >= 0 && (y-j) >= 0)
					{
						long index_i = Disparity.Index(x-j, y-j);
						if(!isnan(Disparity[index_i]))
						{
							float weight = Disparity[index_i]/j;
							
							aggregateColorValue(Filled, i, index_i, weight);
							weights += weight;
							lu_done = true;
						}
					}
					else
					{
						lu_done = true;
					}

					if(!ru_done && (x+j) < width && (y-j) >= 0)
					{
						long index_i = Disparity.Index(x+j, y-j);
						if(!isnan(Disparity[index_i]))
						{
							float weight = Disparity[index_i]/j;
							
							aggregateColorValue(Filled, i, index_i, weight);
							weights += weight;
							ru_done = true;
						}
					}
					else
					{
						ru_done = true;
					}

					if(!ld_done && (x-j) >= 0 && (y+j) < height)
					{
						long index_i = Disparity.Index(x-j, y+j);
						if(!isnan(Disparity[index_i]))
						{
							float weight = Disparity[index_i]/j;
							
							aggregateColorValue(Filled, i, index_i, weight);
							weights += weight;
							ld_done = true;
						}
					}
					else
					{
						ld_done = true;
					}

					if(!rd_done && (x+j) < width && (y+j) < height)
					{
						long index_i = Disparity.Index(x+j, y+j);
						if(!isnan(Disparity[index_i]))
						{
							float weight = Disparity[index_i]/j;
							
							aggregateColorValue(Filled, i, index_i, weight);
							weights += weight;
							rd_done = true;
						}
					}
					else
					{
						rd_done = true;
					}					


					j = j + 1;
					done = l_done && r_done && u_done && d_done && lu_done && ru_done && ld_done && rd_done;
				}


				if(weights > 0.f)
				{
					normalizeColorValue(Filled, i, weights);
				}
			}
		}	
	}
	else if(algorithm == 666) // multi-sample depth-weighted averaging
	{		
		#pragma omp parallel
		{
			float *colors = new float[layers];
			float disparity = 0;

			#pragma omp for
			for(long i=0; i<HW; i++)
			{
				if(_isnan(Disparity[i]))
				{
					int x = i / height;
					int y = i % height;
			
					for(int c=0; c<layers; c++)
					{
						colors[c] = 0;
					}
					disparity = 0;
					float weights = 0;

					// find window size which has some content from all 4 directions
					int best_radius = 1;
					int x_left, x_right, y_up, y_down;

					for(x_left=x-1; x_left>=0; x_left--)
					{
						long index_from = Disparity.Index(x_left, y);
						if(!_isnan(Disparity[index_from]))
						{
							break;
						}
					}
				
					for(x_right=x+1; x_right<width; x_right++)
					{
						long index_from = Disparity.Index(x_right, y);
						if(!_isnan(Disparity[index_from]))
						{		
							break;
						}
					}
				
					for(y_up=y-1; y_up>=0; y_up--)
					{
						long index_from = Disparity.Index(x, y_up);
						if(!_isnan(Disparity[index_from]))
						{
							break;
						}
					}
				
					for(int y_down=y+1; y_down<height; y_down++)
					{
						long index_from = Disparity.Index(x, y_down);
						if(!_isnan(Disparity[index_from]))
						{
							break;
						}
					}

					best_radius = std::max(best_radius, x-x_left);
					best_radius = std::max(best_radius, x_right-x);
					best_radius = std::max(best_radius, y_up-y);
					best_radius = std::max(best_radius, y-y_down);
					const float sigma_space = best_radius / 2;
					for(int xx=std::max(0,x-best_radius); xx<=std::min(width-1, x+best_radius); xx++)
					{
						for(int yy=std::max(0,y-best_radius); yy<=std::min(height-1, y+best_radius); yy++)
						{
							long indexx = Disparity.Index(xx, yy);

							if(!_isnan(Disparity[indexx]))
							{
								float distance_space = sqrt((float)(x-xx)*(x-xx) + (y-yy)*(y-yy));
								float distance_depth = std::abs(Disparity[indexx]-mindisp);

								float weight = exp(-distance_space/sigma_space) * exp(-distance_depth);

								for(int c=0; c<layers; c++)
								{
									colors[c] += Filled[indexx + c*HW] * weight;
								}
								disparity += Disparity[indexx] * weight;

								weights += weight;
							}

						}
					}

					if(weights > 0.f)
					{
						for(int c=0; c<layers; c++)
						{
							Filled.data[i + c*HW] = colors[c] / weights;
						}

						FilledDisp.data[i] = disparity / weights;
					}
				}
			}	
			delete[] colors;
		}
	}
	/*
	else if(algorithm == 5) // depth-prioritized prediction with depth-weighted averging
	{		
		MexImage<float> Weights(width, height);
		
		// prepare weights
		#pragma omp parallel for 
		for(long i=0; i<HW; i++)
		{
			if(_isnan(FilledDisp[i]))
			{
				Weights.data[i] = 0.f; // means pixel-to-be predicted
				for(int c=0; c<layers; c++)
				{
					Filled.data[i+c*HW] = 0.f;
				}
		
				FilledDisp.data[i] = 0.f;
			}
			else
			{
				Weights.data[i] = nan; // means known pixel
			}
		}
	
		bool hasNans = true;
		while(hasNans)
		{
			for(long i=0; i<HW; i++)
			{		
				int x = i / height;
				int y = i % height;

				


			}
			
		}
		

	}
	*/
	
}
