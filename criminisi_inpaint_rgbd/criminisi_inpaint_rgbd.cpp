/** criminisi_inpaint_rgbd
*	@file criminisi_inpaint_rgbd.cpp
*	@date 28.11.2013
*	@author Sergey Smirnov <sergey.smirnov@tut.fi>
*/

#include "../common/meximage.h"
#include <float.h>
#include <cmath>
#include <algorithm>
#include <vector>
#ifndef _DEBUG
	#include <omp.h>
#endif

#include <string.h>

#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}
#endif // __cplusplus

typedef signed char int8;

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")

using namespace mymex;

void calculateSource(MexImage<bool> &Valid, MexImage<float> &IntegralConf, MexImage<bool> &Source, const int x0, const int y0, const int x1, const int y1, const int radius);
void calculateAverage(MexImage<float> &Image, MexImage<bool> &Valid, MexImage<float> &Average, const int x0, const int y0, const int x1, const int y1, const int radius);
void calculateGradient(MexImage<float> &Image, MexImage<bool> &Valid, MexImage<float> &GradientX, MexImage<float> &GradientY, const int x0, const int y0, const int x1, const int y1);
void calculateNormal(MexImage<bool> &Valid, MexImage<int8> &NormalX, MexImage<int8> &NormalY, const int x0, const int y0, const int x1, const int y1);
void inverseDepthGradient(MexImage<bool> &Valid, MexImage<float> &DepthGradientX, MexImage<float> &DepthGradientY, MexImage<float> &FilledDepth, const int best_x, const int best_y, const int radius);
void calculateDepthGradient(MexImage<float> &Depth, MexImage<bool> &Valid, MexImage<float> &DepthGradientX, MexImage<float> &DepthGradientY, const int x0, const int y0, const int x1, const int y1);
//float get_average_depth(MexImage<float> &Depth, MexImage<float> &Confidence, const int x, const int y, const int radius)
//{
//	const int width = Depth.width;
//	const int height= Depth.height;
//
//	float average_depth = 0;
//	int pxls = 0;
//
//	for(int xx=std::max(0,x-radius); xx<=std::min(width-1, x+radius); xx++)
//	{
//		for(int yy=std::max(0,y-radius); yy<=std::min(height-1, y+radius); yy++)
//		{						
//			const long indexx = Depth.Index(xx,yy);
//
//			if(!Confidence[indexx])
//			{
//				continue;
//			}
//			average_depth += Depth[indexx];
//			pxls ++;
//		}
//	}
//
//	return pxls>0 ? average_depth/pxls : 0.f;
//}

void mexFunction (int nout, mxArray* output[], int in, const mxArray* input[])
{	
#ifndef _DEBUG
	omp_set_num_threads(std::max<int>(4, omp_get_max_threads()));
	omp_set_dynamic(std::max<int>(2, omp_get_max_threads() / 2));;
#endif	

	if(in < 3 || in > 9 || nout != 6 || mxGetClassID(input[0]) != mxSINGLE_CLASS || mxGetClassID(input[1]) != mxSINGLE_CLASS || mxGetClassID(input[2]) != mxSINGLE_CLASS)
	{
		mexPrintf("USAGE: [FilledColor, FilledDepth, Dx, Dy, DataTerm, Confidence] = criminisi_inpaint_rgbd(Color, Depth, Priority, <search, radius, option1, option2, ...>);\n");
		mexPrintf("Color - multi-color single-valued image.\n");
		mexPrintf("Depth - depth or disparity map, Nan-s are treated as holes which needs to be inpainted.\n");
		mexPrintf("Priority - a single-valued priority term, to be multiplied with internal Criminisi-inpainting confidence. \n");		
		mexPrintf("search - radius of the search area. <default: 0 - whole image>\n");
		mexPrintf("radius - radius of the patch size. <default: 3>\n");
		mexPrintf("Possible options: \n");
		mexPrintf("'UpdateSource' - uses inpainted area as a source.\n");
		mexPrintf("'EarlyTermination' - Tries to speed-up exhausive patch search.\n");
		mexPrintf("'AveragedGradient' - requires 'EarlyTermination', not compatible with 'NoDataTerm'.\n");
		mexPrintf("'NoDataTerm' - Onion-peel updating approach.\n");		
		mexPrintf("'SmallerUpdate' - Updated area is smaller than a patch size.\n");
		mexPrintf("'I100' - Performs 100 iterations at max (number can vary).\n");		
		//update, upd_source,
		mexErrMsgTxt("Wrong input/output parameters!");
    }

	MexImage<float> Color(input[0]);
	MexImage<float> Depth(input[1]);	
	MexImage<float> Priority(input[2]);

	const int height = Color.height;
	const int width = Color.width;
	const int HW = Color.layer_size;
	const int colors = Color.layers;
		
	
	if(Depth.width != width || Depth.height != height || Depth.layers != 1 )
	{
		mexErrMsgTxt("Wrong Depth dimensions!");
	}

	if(Priority.width != width || Priority.height != height || Priority.layers != 1 )
	{
		mexErrMsgTxt("Wrong Priority dimensions!");
	}


	const int _search = (in > 3) ? std::abs((int)mxGetScalar(input[3])) : 0;	
	const int search = _search ? _search : std::max(width, height);
	const int radius = (in > 4) ? std::max(1, (int)mxGetScalar(input[4])) : 3;			
	
	bool _upd_source = false;
	unsigned _maxiter = 0;
	bool _smaller_update = false;
	bool _earlyTerm = false;
	bool _noDataTerm = false;
	bool _averagedGradient = false;

	if(in > 5)
	{
		char buffer[100];
		for(int i=5; i<in; i++)
		{			
			const mxArray* cell = input[i];
			if(mxGetClassID(cell) == mxCHAR_CLASS && !mxGetString(cell, buffer, 99))
			{
				if(!strcmp(buffer, "UpdateSource") || !strcmp(buffer, "us"))
				{
					_upd_source = true;
				}
				else if(!strcmp(buffer, "EarlyTermination") || !strcmp(buffer, "et"))
				{
					_earlyTerm = true;
				}
				else if(!strcmp(buffer, "SmallerUpdate") || !strcmp(buffer, "su"))
				{
					_smaller_update = true;
				}
				else if(!strcmp(buffer, "NoDataTerm") || !strcmp(buffer, "ndt"))
				{
					_noDataTerm = true;
				}
				else if(!strcmp(buffer, "AveragedGradient") || !strcmp(buffer, "ag"))
				{
					_averagedGradient = true;
				}
				else if(!strncmp(buffer, "I", 1) || !strncmp(buffer, "i", 1))
				{
					_maxiter = atoi(buffer+1);
				}
				else
				{
					mexErrMsgTxt("Cannot understand optional parameter.");
				}
			}
			else
			{
				mexErrMsgTxt("Optional parameter is not a string.");
			}
		}
	}

	const bool upd_source = _upd_source;		
	const unsigned maxiter = _maxiter;
	const int update = (_smaller_update && radius>0) ? radius-1 : radius;
	//const bool earlyTerm = _earlyTerm;
	//const bool useDataTerm = !_noDataTerm;
	//const bool averagedGradient  = useDataTerm && _averagedGradient && earlyTerm;

	const int diameter = radius*2 + 1;
	const int window = diameter*diameter;
	const float nan = sqrt(-1.f);

	matlab_size dims3d[] = {(matlab_size)height, (matlab_size)width, colors};
	matlab_size dims2d[] = {(matlab_size)height, (matlab_size)width, 1};

	output[0] = mxCreateNumericArray(3, dims3d, mxSINGLE_CLASS, mxREAL);
	output[1] = mxCreateNumericArray(3, dims2d, mxSINGLE_CLASS, mxREAL);
	
	output[2] = mxCreateNumericArray(3, dims2d, mxSINGLE_CLASS, mxREAL);
	output[3] = mxCreateNumericArray(3, dims2d, mxSINGLE_CLASS, mxREAL);	

	output[4] = mxCreateNumericArray(3, dims2d, mxSINGLE_CLASS, mxREAL);
	output[5] = mxCreateNumericArray(3, dims2d, mxSINGLE_CLASS, mxREAL);	


	MexImage<float> Filled(output[0]);
	MexImage<float> FilledDepth(output[1]);

	MexImage<float> DepthGradientX (output[2]);
	MexImage<float> DepthGradientY (output[3]);

	//MexImage<float> DepthGradientX (width, height);
	//MexImage<float> DepthGradientY (width, height);

	//MexImage<float> DataTerm (width, height);
	//MexImage<float> Confidence(width, height);		
	MexImage<float> DataTerm (output[4]);
	MexImage<float> Confidence (output[5]);
	
	//DataTerm.setval(0.f);
	//Confidence.setval(0.f);

	//MexImage<float> Average (width, height, colors);
	MexImage<float> GradientX (width, height);
	MexImage<float> GradientY (width, height);
	MexImage<float> IntegralConf(width, height);	
	MexImage<int8> NormalX (width, height);
	MexImage<int8> NormalY (width, height);
	
	
	MexImage<bool> Valid(width, height);		
	//MexImage<bool> Valid(output[2]);
	MexImage<bool> Source(width, height);	
	DepthGradientX.setval(0.f);
	DepthGradientY.setval(0.f);
	//Average.setval(0.f);
	GradientX.setval(0.f);
	GradientY.setval(0.f);
	IntegralConf.setval(0.f);
	DataTerm.setval(0.f);
	NormalX.setval(false);
	NormalY.setval(false);
	Valid.setval(false);
	Source.setval(false);

	
	unsigned long holes = 0;

	#pragma omp parallel for reduction(+:holes)
	for(long i=0; i<HW; i++)
	{
		bool ishole = isnan(Depth[i]);
		for(int c=0; c<colors; c++)
		{
			ishole |= isnan(Color[i+c*HW]);
		}

		holes += ishole;				
		Valid.data[i] = !ishole;
	}	
			
	// depth gradient
	// this function slightly updates map of valid pixels
	//calculateDepthGradient(Depth, Valid, DepthGradientX, DepthGradientY, 0, 0, width-1, height-1);		

	for(long i=0; i<HW; i++)
	{
		bool ishole = !Valid[i];
		

		FilledDepth.data[i] = (ishole) ? nan : Depth[i];
		for(int c=0; c<colors; c++)
		{
			Filled.data[i+c*HW] = (ishole) ? nan : Color[i+c*HW];
		}

		holes += ishole;				
		Valid.data[i] = !ishole;
		//IntegralDepth.data[i] = ishole ? 0.f : Depth[i];
		float confIn = Priority[i];
		confIn = isnan(confIn) ? 0.f : confIn;
		confIn = confIn < 0.f ? 0.f : (confIn > 1.f ? 1.f : confIn);			
		Confidence.data[i] = float(!ishole) * confIn;		
		IntegralConf.data[i] = Confidence[i];
	}

	#pragma omp parallel sections
	{

		#pragma omp section
		{
			// calculate fast confidence
			IntegralConf.IntegralImage(true);	
		}
		#pragma omp section
		{
			// normals map (used to find boundary patches)
			calculateNormal(Valid, NormalX, NormalY, 0,0, width-1, height-1);
		}
		
	}
	
	calculateSource(Valid, IntegralConf, Source, 0, 0, width, height, radius);



	// average map (used in early termination)
	//if(earlyTerm)
	//{
	//	calculateAverage(Color, Valid, Average, 0, 0, width-1, height-1, radius);
	//}
	
	// gradients map (isophotes for data term)
	//if(useDataTerm)
	//{
	//	if(averagedGradient)
	//	{
	//		calculateGradient(Average, Valid, GradientX, GradientY, 0, 0, width-1, height-1);		
	//	}
	//	else
		{
			calculateGradient(Color, Valid, GradientX, GradientY, 0, 0, width-1, height-1);		
			calculateGradient(Depth, Valid, DepthGradientX, DepthGradientY, 0, 0, width-1, height-1);		
		}
	//}
	
	
	//bool hasHoles = true;
	unsigned iter = 0;	
	long updated = 1;
		
	//double avg_checked = 0;
	unsigned long long searches = 0;
	const unsigned long holes_saved = holes;

	// process goes iteratively
	while(holes > 0 && updated > 0 && (!maxiter || iter < maxiter))
	{		
		updated = 0;
		
		long best_index = -1;
		float high_priority = 0.f;
		float best_conf = -0.1f;
		
		// order of inpainting - find appropriate hole :-)
		#pragma omp parallel for
		for(long i=0; i<HW; i++)
		{
			int x = i / height;
			int y = i % height;

			/*if(!NormalX[i] && !NormalY[i])
			{
				continue;
			}*/

			// if using Data Term, place for inpainting can be "outside" of valid are, but normals must be presented
			//if(useDataTerm && !NormalX[i] && !NormalY[i])
			//{
			//	continue;
			//}			
			// alternatively, without Data Term, center of patch should be valid.
			// ToDo: check if !NormalX[i] && !NormalY[i] are sufficient conditions here also
			//else 
			/*if(!useDataTerm && Valid[i]) 
			{
				continue;				
			}*/

			if(!NormalX[i] && !NormalY[i]) 
			{
				continue;
			}

			
			const float conf = IntegralConf.getIntegralAverage(x, y, radius);

			//if(useDataTerm)
			//{
				float gx = GradientY[i] + DepthGradientY[i];
				float gy = -GradientX[i] - DepthGradientX[i];
				gx = isnan(gx) ? 0 : gx;
				gy = isnan(gy) ? 0 : gy;
				//float gdx = abs(DepthGradientY[i]);
				//float gdy =- abs(DepthGradientX[i]);
				//gdx = isnan(gdx) ? 0 : gdx;
				//gdy = isnan(gdy) ? 0 : gdy;

				// calculate isophote here
				float data = abs(gx*NormalX[i] + gy*NormalY[i]);
												
				//data *= abs(gdx*NormalX[i] - gdy*NormalY[i]);				

				data = data < 0.001 ?  0.001 : data;		
				DataTerm.data[i] = data;
				float priority = data * conf;
			//} 
			
			if(priority > high_priority)
			{
				#pragma omp critical
				{
					high_priority = priority;
					best_index = i;
				}
				best_conf = conf;
			}
		}

		if(best_index < 0)
		{
			mexPrintf("ERROR: No appropriate patch to inpaint was found :-(\n");
			break;
		}

		const int best_x = best_index / height;
		const int best_y = best_index % height;
		//mexPrintf("\t inpainting pixel: (%d, %d), conf = %3.2f\n", best_x, best_y, best_conf);
		
		//// calculate average depth of the patch-to-inpaint
		//float average_depth = 0;
		//int pxls = 0;
		//for(int dx=-radius, i=0; dx<=radius; dx++)
		//{
		//	for(int dy=-radius; dy<=radius; dy++, i++)
		//	{
		//		const int xx = best_x + dx;
		//		const int yy = best_y + dy;
		//		if(xx<0 || xx>= width || yy<0 || yy>= height)
		//		{
		//			continue;
		//		}
		//		const long indexx = FilledDepth.Index(xx,yy);
		//		if(!Valid[indexx])
		//		{
		//			continue;
		//		}
		//		average_depth += FilledDepth[indexx];
		//		pxls ++;
		//	}
		//}
		//average_depth /= pxls;

		// seek for the best valid patch over whole image o_O
		float best_diff = 1000000.f;		
		long best_sindex = -1;
		//float best_avg = 1000000.f;				
		unsigned long checked = 0;
		unsigned long runned = 0;



		#pragma omp parallel for reduction(+:checked,runned)
		//for(long i=0; i<HW; i++)
		for(int x=std::max(radius, best_x-search); x<=std::min(width-1-radius, best_x+search); x++)
		for(int y=std::max(radius, best_y-search); y<=std::min(height-1-radius, best_y+search); y++)
		{
			runned ++;
			long i = Source.Index(x,y);
			if(!Source[i])
				continue;
			
			//// Early termination with averaging
			//float avgdiff = 0;

			//for(int c=0; c<colors /*&& earlyTerm*/; c++)
			//{
			//	avgdiff += abs(Average[best_index+c*HW] - Average[i + c*HW]);			
			//}

			//if(earlyTerm && avgdiff > best_diff)// && best_sindex > 0)
			//	continue;			

			//if(earlyTerm && avgdiff > best_avg)// && best_sindex > 0)
			//	continue;			
						
			//for(int c=0; c<colors; c++)
			//{
			//	avgdiff += abs(GradientXAvg[best_index+c*HW] - GradientXAvg[i + c*HW])/colors;			
			//}
			//
			//if(avgdiff > best_diff)// && best_sindex > 0)
			//	continue;			

			//for(int c=0; c<colors; c++)
			//{
			//	avgdiff += abs(GradientYAvg[best_index+c*HW] - GradientYAvg[i + c*HW])/colors;			
			//}

			//if(avgdiff > best_diff)// && best_sindex > 0)
			//	continue;			

							
			
			//const float average_search_depth = get_average_depth(Depth, ConfidenceIn, x, y, radius);
			float diff = 0;
			int pixels = 0;
			for(int dx=-radius; dx<=radius; dx++)
			{
				int sx = x + dx;
				int bx = best_x + dx;
				if(bx < 0 || bx >= width)
					continue;

				for(int dy=-radius; dy<=radius; dy++)
				{
					int sy = y + dy;
					int by = best_y + dy;
					
					if(by < 0 || by >= height)
						continue;
					
					long sindex = Filled.Index(sx, sy);
					long bindex = Filled.Index(bx, by);

					if(!Valid[bindex])
					{
						continue;
					}
					

					for(int c=0; c<colors; c++)
					{
						diff += abs(Color[sindex+c*HW] - Filled[bindex+c*HW]);
					}

					//diff += abs((Depth[sindex]-average_search_depth) - (FilledDepth[bindex]-average_depth));					

					//diff /= (colors);
					diff += abs(Depth[sindex] - FilledDepth[bindex]);
					//diff += abs(DepthGradientX[sindex] - DepthGradientX[bindex]);
					//diff += abs(DepthGradientY[sindex] - DepthGradientY[bindex]);
					
					pixels ++;
				}				
			}
			
			diff /= pixels;

			if(diff < best_diff)
			{
				#pragma omp critical
				{
					best_diff = diff;
					best_sindex = i;
				}

				//if(earlyTerm && avgdiff < best_avg*2)
				//{
				//	#pragma omp critical
				//	{
				//		best_avg = avgdiff;
				//	}
				//}
			}

			checked ++;			
		}
		
		if(best_sindex < 0)
		{
			mexPrintf("ERROR: No appropriate match for patch at (%d, %d) was found (%d (%d) patches were checked, ) \n", best_x, best_y, checked, runned);
			break;
		}

		searches += checked;
		
		const int best_sx = best_sindex / height;
		const int best_sy = best_sindex % height;
		
		//const float average_search_depth = get_average_depth(Depth, ConfidenceIn, best_sx, best_sy, radius);
		//mexPrintf("\t best patch: (%d, %d), diff = %3.2f\n", best_sx, best_sy, best_diff);

		// update found hole with found patch
		#pragma omp parallel for
		for(int dx=-update; dx<=update; dx++)
		{
			int sx = best_sx + dx;
			int bx = best_x + dx;
			if(bx < 0 || bx >= width)
				continue;

			for(int dy=-update; dy<=update; dy++)
			{
				int sy = best_sy + dy;
				int by = best_y + dy;
					
				if(by < 0 || by >= height)
					continue;
					
				long sindex = Filled.Index(sx, sy);
				long bindex = Filled.Index(bx, by);

				if(Valid[bindex])
				{
					continue;
				}

				Valid.data[bindex] = true;
				Confidence.data[bindex] = best_conf;

				for(int c=0; c<colors; c++)
				{
					Filled.data[bindex+c*HW] = Color[sindex+c*HW];
				}
				FilledDepth.data[bindex] = Depth[sindex];
				//FilledDepth.data[bindex] = Depth[sindex] - average_search_depth + average_depth;
				//DepthGradientX.data[bindex] = DepthGradientX[sindex];
				//DepthGradientY.data[bindex] = DepthGradientY[sindex];

				updated ++;
			}				
		}		
		
		//inverseDepthGradient(Valid, DepthGradientX, DepthGradientY, FilledDepth, best_x, best_y, update);

		// update normals map
		calculateNormal(Valid, NormalX, NormalY, std::max(0,best_x-update-1), std::max(0,best_y-update-1), std::min(width-1,best_x+update+1), std::min(height-1,best_y+update+1));			

		//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		//calculateDepthGradient(FilledDepth, Valid, DepthGradientX, DepthGradientY, std::max(0,best_x-update-1), std::max(0,best_y-update-1), std::min(width-1,best_x+update+1), std::min(height-1,best_y+update+1));


		//// update averages map
		//if(earlyTerm)
		//{
		//	calculateAverage(Filled, Valid, Average, std::max(0,best_x-radius-update), std::max(0,best_y-radius-update), std::min(width-1,best_x+radius+update), std::min(height-1,best_y+radius+update), radius);
		//}
		
		// update gradients
		//if(useDataTerm)
		{
			//if(averagedGradient)
			//{
			//	calculateGradient(Average, Valid, GradientX, GradientY, std::max(0,best_x-update-1), std::max(0,best_y-update-1), std::min(width-1,best_x+update+1), std::min(height-1,best_y+radius+update+1));			
			//}
			//else
			{
				calculateGradient(Filled, Valid, GradientX, GradientY, std::max(0,best_x-update-1), std::max(0,best_y-update-1), std::min(width-1,best_x+update+1), std::min(height-1,best_y+update+1));			
				calculateGradient(FilledDepth, Valid, DepthGradientX, DepthGradientY, std::max(0,best_x-update-1), std::max(0,best_y-update-1), std::min(width-1,best_x+update+1), std::min(height-1,best_y+update+1));			
			}			
		}



		//calculateAverage(GradientX, Valid, GradientXAvg, std::max(0,best_x-radius-update), std::max(0,best_y-radius-update), std::min(width-1,best_x+radius+update), std::min(height-1,best_y+radius+update), radius);
		//calculateAverage(GradientY, Valid, GradientYAvg, std::max(0,best_x-radius-update), std::max(0,best_y-radius-update), std::min(width-1,best_x+radius+update), std::min(height-1,best_y+radius+update), radius);

		holes -= updated;		
		iter ++;
		
		for(long i=0; i<HW;i++)
		{
			IntegralConf.data[i] = Confidence[i];
		}
		
		IntegralConf.IntegralImage(true);
		
		//if(upd_source)
		//{
		//	calculateSource(Valid, IntegralConf, Source, best_x-update-radius, best_y-update-radius, best_x+update+radius, best_y+update+radius, radius+1);
		//}

		//if(! iter % 100)
		//mexPrintf("Iteration %d; updated %d; remaining holes %d\n\n", iter, updated, holes);
		//mexPrintf("Iteration %d; updated %d; checked %d; remaining holes %d\n", iter, updated, checked, holes);
		//avg_checked = (avg_checked*(iter-1) + checked)/iter;
	}		
	
	mexPrintf("Inpainted pixels: %d; using patches: %d; total matches: %d. \n", holes_saved-holes, iter, searches);
	
}

inline int block_index(const int ix, const int iy, const int diameter)
{
	return ix*diameter + iy;
}

void inverseDepthGradient(MexImage<bool> &Valid, MexImage<float> &DepthGradientX, MexImage<float> &DepthGradientY, MexImage<float> &FilledDepth, const int best_x, const int best_y, const int radius)
{
	const int width = FilledDepth.width;
	const int height = FilledDepth.height;
	const int HW = FilledDepth.layer_size;
	const int diameter = radius*2 +1;
	const int window = diameter * diameter;
	const float nan = sqrt(-1.f);
	float *depths = new float[window];
	float *gradientsX = new float[window];
	float *gradientsY = new float[window];
	//float *weights = new float[window];
	bool *valids = new bool[window];

	//for(int i=0; i<window; i++)
	//{
	//	weights[i] = 0.f;
	//}

	for(int ix=0; ix<diameter; ix++)
	{
		for(int iy=0; iy<diameter; iy++)
		{
			const int i = block_index(ix, iy, diameter);				
			const int xi = best_x + ix - radius;
			const int yi = best_y + iy - radius;

			if(xi < 0 || xi >= width || yi < 0 || yi >= height)
			{
				depths[i] = i;
				gradientsX[i] = 0.f;
				gradientsY[i] = 0.f;
				valids[i] = 0;
				continue;
			}
			const long indexi = FilledDepth.Index(xi, yi);

			gradientsX[i] = DepthGradientX[indexi];
			gradientsY[i] = DepthGradientY[indexi];

			if(!Valid[indexi])
			{
				depths[i] = i;
				valids[i] = 0;
			}
			else
			{
				depths[i] = FilledDepth[indexi];
				valids[i] = 1;
			}
		}
	}
	
	bool holes_exist = true;
	while(holes_exist)
	{

		holes_exist = false;

		for(int ix=0; ix<diameter; ix++)
		{
			for(int iy=0; iy<diameter; iy++)
			{
				const int i = block_index(ix, iy, diameter);		

				if(!valids[i])
				{
					bool exists_l = ix>0 && valids[block_index(ix-1, iy, diameter)];
					bool exists_r = ix<diameter-1 && valids[block_index(ix+1, iy, diameter)];
					bool exists_t = iy>0 && valids[block_index(ix, iy-1, diameter)];
					bool exists_d = iy<diameter-1 && valids[block_index(ix, iy+1, diameter)];

					if(exists_l || exists_r || exists_t || exists_d)
					{
						float depth = 0;
						int pix = 0;
						if(exists_l)
						{
							depth += depths[block_index(ix-1, iy, diameter)] + 2*gradientsX[i];
							pix ++;
						}
						if(exists_r)
						{
							depth += depths[block_index(ix+1, iy, diameter)] - 2* gradientsX[i];
							pix ++;
						}
						if(exists_t)
						{
							depth += depths[block_index(ix, iy-1, diameter)] + 2*gradientsY[i];
							pix ++;
						}
						if(exists_d)
						{
							depth += depths[block_index(ix, iy+1, diameter)] - 2*gradientsY[i];
							pix ++;
						}

						depths[i] = depth/pix;
						valids[i] = true;
					}
					else
					{
						holes_exist = true;
					}
				}
			}
		}		
	}

	// copy back recovered depth values
	for(int ix=0; ix<diameter; ix++)
	{
		for(int iy=0; iy<diameter; iy++)
		{
			const int i = block_index(ix, iy, diameter);				
			const int xi = best_x + ix - radius;
			const int yi = best_y + iy - radius;

			if(xi < 0 || xi >= width || yi < 0 || yi >= height)
			{
				continue;
			}

			const long indexi = FilledDepth.Index(xi, yi);

			if(!Valid[indexi])
			{
				FilledDepth.data[indexi] = depths[i];
				Valid.data[indexi] = true;
			}

		}
	}

	delete[] depths, gradientsX, gradientsY;
	//delete[] weights;
	delete[] valids;
}

void calculateDepthGradient(MexImage<float> &Depth, MexImage<bool> &Valid, MexImage<float> &DepthGradientX, MexImage<float> &DepthGradientY, const int x0, const int y0, const int x1, const int y1)
{
	const int width = Depth.width;
	const int height = Depth.height;
	const int HW = Depth.layer_size;
	float nan = sqrt(-1.f);
	//const int colors = Image.layers;

	#pragma omp parallel for
	for(int x=std::max(0,x0); x <= std::min(width-1,x1); x++)
	{			
		for(int y=std::max(0,y0); y <= std::min(height-1,y1); y++)
		{
			long index = Depth.Index(x,y);
			
			if(!Valid[index])
				continue;

			
			int xl = std::max(0, x-1);
			int xr = std::min(width-1, x+1);
			int yu = std::max(0, y-1);
			int yd = std::min(height-1, y+1);

			long indexl = Depth.Index(xl, y);
			long indexr = Depth.Index(xr, y);
			long indexu = Depth.Index(x, yu);
			long indexd = Depth.Index(x, yd);

			indexl = Valid[indexl] ? indexl : index;
			indexr = Valid[indexr] ? indexr : index;
			indexu = Valid[indexu] ? indexu : index;
			indexd = Valid[indexd] ? indexd : index;
			if(indexl == indexr || indexu == indexd)
			{
				Valid.data[index] = false;
				DepthGradientX.data[index] = nan;
				DepthGradientY.data[index] = nan;
			}
			else
			{
				DepthGradientX.data[index] = (Depth[indexr]-Depth[indexl])/2;
				DepthGradientY.data[index] = (Depth[indexd]-Depth[indexu])/2;
			}
		}
	}
}


void calculateAverage(MexImage<float> &Image, MexImage<bool> &Valid, MexImage<float> &Average, const int x0, const int y0, const int x1, const int y1, const int radius)
{
	const int width = Image.width;
	const int height = Image.height;
	const int HW = Image.layer_size;
	const int colors = Image.layers;

	#pragma omp parallel for
	for(int x=std::max(0,x0); x <= std::min(width-1,x1); x++)
	{			
		for(int y=std::max(0,y0); y <= std::min(height-1,y1); y++)
		{
			long index = Image.Index(x,y);
			for(int c=0; c<colors; c++)
			{
				Average.data[index + c*HW] = 0;
			}
			int pixels = 0;

			for(int sx=std::max(0, x-radius); sx<=std::min(width-1, x+radius); sx++)
			{
				for(int sy=std::max(0, y-radius); sy<=std::min(height-1, y+radius); sy++)
				{
					long sindex = Average.Index(sx, sy);
					if(!Valid[sindex])
					{
						continue;
					}
					for(int c=0; c<colors; c++)
					{
						Average.data[index + c*HW] += Image[sindex + c*HW];
					}
					pixels ++;
				}
			}
			if(pixels > 0)
			{
				for(int c=0; c<colors; c++)
				{
					Average.data[index + c*HW] /= pixels;
				}
			}
		}
	}
}

void calculateGradient(MexImage<float> &Image, MexImage<bool> &Valid, MexImage<float> &GradientX, MexImage<float> &GradientY, const int x0, const int y0, const int x1, const int y1)
{
	const int width = Image.width;
	const int height = Image.height;
	const int HW = Image.layer_size;
	const int colors = Image.layers;

	#pragma omp parallel for
	for(int x=std::max(0,x0); x <= std::min(width-1,x1); x++)
	{			
		for(int y=std::max(0,y0); y <= std::min(height-1,y1); y++)
		{
			long index = Image.Index(x,y);
			
			if(!Valid[index])
				continue;

			GradientX.data[index] = 0;
			GradientY.data[index] = 0;

			int xl = std::max(0, x-1);
			int xr = std::min(width-1, x+1);
			int yu = std::max(0, y-1);
			int yd = std::min(height-1, y+1);

			long indexl = Image.Index(xl, y);
			long indexr = Image.Index(xr, y);
			long indexu = Image.Index(x, yu);
			long indexd = Image.Index(x, yd);

			indexl = Valid[indexl] ? indexl : index;
			indexr = Valid[indexr] ? indexr : index;
			indexu = Valid[indexu] ? indexu : index;
			indexd = Valid[indexd] ? indexd : index;
			if(indexl == indexr || indexu == indexd)
			{
				Valid.data[index] = false;
			}
			else
			{
				for(int c=0; c<colors; c++)
				{
					GradientX.data[index] += abs(Image[indexr+c*HW]-Image[indexl+c*HW])/(colors*255);
					GradientY.data[index] += abs(Image[indexd+c*HW]-Image[indexu+c*HW])/(colors*255);
				}
			}
		}
	}
}


void calculateSource(MexImage<bool> &Valid, MexImage<float> &IntegralConf, MexImage<bool> &Source, const int x0, const int y0, const int x1, const int y1, const int radius)
{
	const int width = Valid.width;
	const int height = Valid.height;
	const int HW = Valid.layer_size;
	//const int colors = Image.layers;
	const int diameter = radius*2+1;
	const int window = diameter*diameter;

	#pragma omp parallel for
	for(int x=std::max(radius,x0); x <= std::min(width-radius-1,x1); x++)
	{			
		for(int y=std::max(radius,y0); y <= std::min(height-radius-1,y1); y++)
		{
			long index = Valid.Index(x,y);
			
			if(!Valid[index])
				continue;

			int pixels = 0;

			for(int i=0; i<window; i++)
			{
				int sx = x + i/diameter - radius;
				int sy = y + i%diameter - radius;
				long sindex = Valid.Index(sx, sy);

				if(!Valid[sindex])
					break;

				pixels ++;
			}			
			//float conf = IntegralConf.getIntegralAverage(x,y,radius);
			if(pixels == window /*&& conf > 0.5*/)
			{
				Source.data[index] = true;
			}
		}
	}
}


void calculateNormal(MexImage<bool> &Valid, MexImage<int8> &NormalX, MexImage<int8> &NormalY, const int x0, const int y0, const int x1, const int y1)
{
	const int width = Valid.width;
	const int height = Valid.height;
	const int HW = Valid.layer_size;
	const int colors = Valid.layers;

	#pragma omp parallel for
	for(int x=std::max(0,x0); x <= std::min(width-1,x1); x++)
	{			
		for(int y=std::max(0,y0); y <= std::min(height-1,y1); y++)
		{
			long index = Valid.Index(x,y);			
			
			NormalX.data[index] = 0;
			NormalY.data[index] = 0;

			int xl = std::max(0, x-1);
			int xr = std::min(width-1, x+1);
			int yu = std::max(0, y-1);
			int yd = std::min(height-1, y+1);

			long indexl = Valid.Index(xl, y);
			long indexr = Valid.Index(xr, y);
			long indexu = Valid.Index(x, yu);
			long indexd = Valid.Index(x, yd);

			NormalX.data[index] = -((int8)Valid[indexr] - (int8)Valid[indexl]);
			NormalY.data[index] = -((int8)Valid[indexd] - (int8)Valid[indexu]);
		}
	}
}
