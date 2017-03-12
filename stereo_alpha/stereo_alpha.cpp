/** stereo_alpha
* @file stereo_alpha.cpp
* @author Sergey Smirnov <sergey.smirnov@tut.fi>
* @date 1.09.2015
* @copyright 3D Media Group / Tampere University of Technology
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
#include <memory>
#include <array>
#ifndef _DEBUG
#include <omp.h>
#endif
#include <Eigen\Dense>
//#define M_PI       3.14159265358979323846

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")

//typedef unsigned char uint8;
using namespace mymex;


template<int radius>
void process_alpha_colors(MexImage<uint8_t> &Left, MexImage<uint8_t> &Right, MexImage<float> &Average, const int mindisp, const int maxdisp)
{
	if (Left.layers == 1)
	{
		process_alpha<radius, 1>(Left, Right, Average, mindisp, maxdisp);
	}
	else if (Left.layers == 2)
	{
		process_alpha<radius, 2>(Left, Right, Average, mindisp, maxdisp);
	}
	else if (Left.layers == 3)
	{
		process_alpha<radius, 3>(Left, Right, Average, mindisp, maxdisp);
	}
	else if (Left.layers == 4)
	{
		process_alpha<radius, 4>(Left, Right, Average, mindisp, maxdisp);
	}
	else if (Left.layers == 5)
	{
		process_alpha<radius, 5>(Left, Right, Average, mindisp, maxdisp);
	}
	else if (Left.layers == 6)
	{
		process_alpha<radius, 6>(Left, Right, Average, mindisp, maxdisp);
	}
	else if (Left.layers == 7)
	{
		process_alpha<radius, 7>(Left, Right, Average, mindisp, maxdisp);
	}
	else if (Left.layers == 8)
	{
		process_alpha<radius, 8>(Left, Right, Average, mindisp, maxdisp);
	}
	else if (Left.layers == 9)
	{
		process_alpha<radius, 9>(Left, Right, Average, mindisp, maxdisp);
	}
	else if (Left.layers == 10)
	{
		process_alpha<radius, 10>(Left, Right, Average, mindisp, maxdisp);
	}
	else if (Left.layers == 11)
	{
		process_alpha<radius, 11>(Left, Right, Average, mindisp, maxdisp);
	}
	else if (Left.layers == 12)
	{
		process_alpha<radius, 12>(Left, Right, Average, mindisp, maxdisp);
	}
	else
	{
		mexErrMsgTxt("Too many color components in yopur images!");
	}

}

void fastGaussian(const MexImage<float> &Cost, const MexImage<float> &Temporal, const float sigma)
{
	const int width = Cost.width;
	const int height = Cost.height;

	// has to be initialized with zeros
	Temporal.setval(0.f);

	// Horisontal (left-to-right & right-to-left) pass
	#pragma omp parallel for
	for (int y = 0; y<height; y++)
	{
		// left-to-right accumulation buffer
		std::unique_ptr<float[]> temporal1 = std::unique_ptr<float[]>(new float[width]);

		// right-to-left accumulation buffer
		std::unique_ptr<float[]> temporal2 = std::unique_ptr<float[]>(new float[width]);

		temporal1[0] = Cost(0, y);
		temporal2[width - 1] = Cost(width - 1, y);
		for (int x1 = 1; x1<width; x1++)
		{
			const int x2 = width - x1 - 1;
			temporal1[x1] = (Cost(x1, y) + temporal1[x1 - 1] * sigma);
			temporal2[x2] = (Cost(x2, y) + temporal2[x2 + 1] * sigma);
		}
		for (int x = 0; x<width; x++)
		{
			Temporal(x, y) = (temporal1[x] + temporal2[x]) - Cost(x, y);
		}
	}

	// Vertical (up-to-down & down-to-up) pass
	#pragma omp parallel for
	for (int x = 0; x<width; x++)
	{
		std::unique_ptr<float[]> temporal1 = std::unique_ptr<float[]>(new float[height]);
		std::unique_ptr<float[]> temporal2 = std::unique_ptr<float[]>(new float[height]);
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
			Cost(x, y) = (temporal1[y] + temporal2[y]) - Temporal(x, y);
		}
	}
}

#define LIM(X, W) ((X) < 0 ? 0 : ((X) >= (W) ? (W-1) : (X)))
#define diameter  ((radius) * 2 + 1)
#define window  (diameter*diameter)

template<int N>
struct max_buffer
{
	float values[N];
	int indexes[N];

	max_buffer()
	{
		for (int i = 0; i < N; i++)
		{
			indexes[i] = -1;
			values[N] = 0.f;
		}
	}

	void add_value(const int _index, const float _value)
	{
		int index = _index;
		float value = _value;
		for (int i = 0; i < N; i++)
		{
			if (value > values[i])
			{
				const float a = values[i];
				const int b = indexes[i];
				values[i] = value;
				indexes[i] = index;
				value = a;
				index = b;
			}
		}
	}
};

template<int N>
struct min_buffer
{
	float values[N];
	int indexes[N];

	min_buffer()
	{
		for (int i = 0; i < N; i++)
		{
			indexes[i] = -1;
			values[N] = 0.f;
		}
	}

	void add_value(const int _index, const float _value)
	{
		int index = _index;
		float value = _value;
		for (int i = 0; i < N; i++)
		{
			if (value < values[i])
			{
				const float a = values[i];
				const int b = indexes[i];
				values[i] = value;
				indexes[i] = index;
				value = a;
				index = b;
			}
		}
	}
};

template<int radius, int colors>
void process_alpha(MexImage<uint8_t> &Left, MexImage<uint8_t> &Right, MexImage<float> &Average, const int mindisp, const int maxdisp)
{
	const double lambda = 0.0000001;
	const int width = Left.width;
	const int height = Left.height;
	const long HW = width*height; 
	const int layers = maxdisp - mindisp + 1;

	Eigen::Matrix<double, window, window> I = Eigen::Matrix<double, window, window>::Identity();
	I(window - 1, window - 1) = 0;	
	

	for (int l = 0; l < layers; l++)
	{
		const int disp = maxdisp - l;
		#pragma omp parallel for
		for (long i = 0; i < HW; i++)
		{
			const int x = i / height;
			const int y = i % height;
			const int xr = LIM(x - disp, width);
			//const int xr = x - disp < 0 ? 0 : (x-disp >= width ? width-1 : x-disp);
			Eigen::Matrix<double, window, (colors * 2 + 1)> X;

			max_buffer<3> max_vals;
			min_buffer<3> min_vals;
			

			for (int j = 0; j < window; j++)
			{
				const int dx = j / diameter - radius;
				const int dy = j % diameter - radius;
				const int xj = LIM(x + dx, width);
				const int yj = LIM(y + dy, height);
				const int xrj = LIM(xr + dx, width);
				float diff = 0;
				for (int c = 0; c < colors; c++)
				{
					X(j, c) = double(Left(xj, yj, c))/255;
					X(j, colors+c) = double(Right(xrj, yj, c)) / 255;
					diff += abs(Left(xj, yj, c) - Right(xrj, yj, c));
				}
				X(j, colors * 2) = 1.0;				
			}
			max_vals.add_value(j, diff);
			min_vals.add_value(j, diff);

			Eigen::Matrix<double, window, window> fenmu = X*X.transpose() + I*lambda;
			Eigen::Matrix<double, window, window> F = X*X.transpose() / fenmu;
			Eigen::Matrix<double, window, window> L = (I - F).transpose() * (I - F);


		}


	}
}



void mexFunction(const int nout, mxArray* output[], const int in, const mxArray* input[])
{
#ifndef _DEBUG
	omp_set_num_threads(std::max<int>(4, omp_get_max_threads()));
	omp_set_dynamic(std::max<int>(2, omp_get_max_threads() / 2));
#endif

	if (in < 5 || in > 8 || nout != 4 || mxGetClassID(input[0]) != mxUINT8_CLASS || mxGetClassID(input[1]) != mxUINT8_CLASS || mxGetClassID(input[2]) != mxSINGLE_CLASS)
	{
		mexErrMsgTxt("USAGE: [AlphaL, DispL, DispR, Reco] = stereo_alpha(uint8(Left), uint8(Right), single(DispLAverage), mindisp, maxdisp,  <radius, isL2R, a>);");
	}

	MexImage<uint8_t> Left(input[0]);
	MexImage<uint8_t> Right(input[1]);
	MexImage<float> Average(input[2]);

	const int mindisp = (int)mxGetScalar(input[3]);
	const int maxdisp = (int)mxGetScalar(input[4]);
	const int _radius = in > 5 ? std::max<int>(1, (int)mxGetScalar(input[5])) : 2;
	//const float sigma = in > 5 ?  std::min<float>(1.f, std::max<float>(0.f, (float)mxGetScalar(input[5]))) : 0.8f;
	const bool isL2R = in > 6 ? (bool)mxGetScalar(input[6]) : true;
	const float a = in > 7 ? (float)mxGetScalar(input[7]) : 1.f;
	//const float b = in > 8 ? (float)mxGetScalar(input[8]) : 2.f;
	const int layers = maxdisp - mindisp + 1;
	const int width = Left.width;
	const int height = Left.height;
	const int colors = Left.layers;
	const long HW = width*height;
	const float nan = sqrt(-1.f);
	//const int diameter = radius * 2 + 1;
	//const int window = diameter * diameter;

	const size_t dims1[] = { (size_t)height, (size_t)width, 1 };
	const size_t dims11[] = { (size_t)height, (size_t)width, (size_t)layers + 1 };
	const size_t dims3[] = { (size_t)height, (size_t)width, colors };
	output[0] = mxCreateNumericArray(3, dims1, mxSINGLE_CLASS, mxREAL);
	output[1] = mxCreateNumericArray(3, dims1, mxINT32_CLASS, mxREAL);
	output[2] = mxCreateNumericArray(3, dims1, mxINT32_CLASS, mxREAL);
	output[3] = mxCreateNumericArray(3, dims3, mxSINGLE_CLASS, mxREAL);
	//output[3] = mxCreateNumericArray(3, dims11, mxSINGLE_CLASS, mxREAL);

	MexImage<float> Alpha(output[0]);
	MexImage<uint32_t> DispL(output[1]);
	MexImage<uint32_t> DispR(output[2]);
	MexImage<float> Reconstructed(output[3]);


	if (_radius == 1)
	{
		process_alpha_colors<1>(Left, Right, Average, mindisp, maxdisp);
	}
	else if (_radius == 2)
	{
		process_alpha_colors<2>(Left, Right, Average, mindisp, maxdisp);
	}
	else if (_radius == 3)
	{
		process_alpha_colors<3>(Left, Right, Average, mindisp, maxdisp);
	}
	else if (_radius == 4)
	{
		process_alpha_colors<4>(Left, Right, Average, mindisp, maxdisp);
	}
	else if (_radius == 5)
	{
		process_alpha_colors<5>(Left, Right, Average, mindisp, maxdisp);
	}
	else if (_radius == 6)
	{
		process_alpha_colors<6>(Left, Right, Average, mindisp, maxdisp);
	}
	else
	{
		mexErrMsgTxt("Too big radius!");
	}

}



//	//MexImage<float> CostAlpha(output[2]);	
//	//MexImage<float> CostError(output[3]);
//	//MexImage<float> CostWeights(width, height, layers);
//	//CostWeights.setval(0.f);
//	//CostAlpha.setval(0.f);
//	//CostError.setval(0.f);
//	//CostL.setval(0.f)
//	MexImage<float> Error(width, height);
//	MexImage<float> TmpAlpha(width, height);
//	MexImage<float> BestError(width, height);
//	MexImage<float> Temporal(width, height);
//	BestError.setval(FLT_MAX);
//
//	for (int fi = 0; fi < layers; fi++)
//	{
//		const int dispFg = maxdisp - fi;
//		//for (int bi = fi; bi < layers; bi++)
//		//int bi = fi;
//		for (int bi = fi+1; bi < layers; bi++)
//		{
//			const int dispBg = maxdisp - bi;
//			#pragma omp parallel for
//			for (long i = 0; i < HW; i++)
//			{
//				const int x = i / height;
//				const int y = i % height;
//				const int _xfg = isL2R ? x - dispFg : x + dispFg;
//				const int _xbg = isL2R ? x - dispBg : x + dispBg;
//				//const int _xLfg = isL2R ? x + dispFg : x - dispFg;
//				//const int _xfg = isL2R ? x - dispFg : x + dispFg;
//				//const int _xbg = isL2R ? x - dispBg : x + dispBg;
//				const int xfg = _xfg < 0 ? 0 : (_xfg >= width ? width - 1 : _xfg);
//				const int xbg = _xbg < 0 ? 0 : (_xbg >= width ? width - 1 : _xbg);
//
//				//float alpha = (fi == bi)*colors;
//				float alpha = 0.f;
//				
//				//float X = 0;
//				for (int c = 0; c < colors /*&& fi != bi*/; c++)
//				{
//					const float a = abs((float(Left(x, y, c)) - float(Right(xbg, y, c))) / (float(Right(xfg, y, c)) - float(Right(xbg, y, c))));
//					alpha += a < 0.f ? 0.f : a > 1.f ? 1.f : a;
//					//X += (float(Left(x, y, c)) - float(Right(xbg, y, c))) * (float(Left(x, y, c)) - float(Right(xbg, y, c)));
//					//X += abs(float(Left(x, y, c)) - float(Right(xfg, y, c))) / float(Right(xbg, y, c));
//					//X += abs(float(Left(x, y, c)) - float(Right(xfg, y, c))) / 255;
//				}
//				
//				//X = X/colors;
//				//float error = FLT_MAX;
//
//				/*for (int a = 0.1; a < 10.f; a += 0.5)
//				{
//					float err = 0;
//					float tmp_alpha = 1 - std::min<float>(1.f, a*X);
//					for (int c = 0; c < colors; c++)
//					{
//						err += abs(float(Left(x, y, c)) - tmp_alpha * float(Right(xfg, y, c)) - (1 - tmp_alpha) * float(Right(xbg, y, c)));
//					}
//					if (err < error)
//					{
//						error = err;
//						alpha = tmp_alpha;
//					}
//				}*/
//				
//				alpha /= colors;
//				//alpha *= 1 - std::min<float>(1.f, a*X);
//				
//				
//				TmpAlpha[i] = alpha;
//				float error = 0;
//				for (int c = 0; c < colors; c++)
//				{
//					error += abs(float(Left(x, y, c)) - alpha * float(Right(xfg, y, c)) - (1 - alpha) * float(Right(xbg, y, c)));
//					//error += abs(float(Left(x, y, c)) - float(Right(xfg, y, c)));
//				}
//
//				Error[i] = error / colors;
//			}
//
//			//fastGaussian(Error, Temporal, sigma);
//
//			//Error.IntegralImage(true);
//			#pragma omp parallel for
//			for (long i = 0; i < HW; i++)
//			{
//				const int x = i / height;
//				const int y = i % height;
//				const int _xfg = isL2R ? x - dispFg : x + dispFg;
//				const int _xbg = isL2R ? x - dispBg : x + dispBg;
//				const int xfg = _xfg < 0 ? 0 : (_xfg >= width ? width - 1 : _xfg);
//				const int xbg = _xbg < 0 ? 0 : (_xbg >= width ? width - 1 : _xbg);
//				//const float error = Error.getIntegralAverage(x, y, radius);
//				const float error = Error[i];
//				if (error < BestError[i])
//				{
//					BestError[i] = error;
//					Alpha[i] = TmpAlpha[i];
//					DispF[i] = dispFg;
//					DispB[i] = dispBg;
//					//Reconstructed(x, y, 0) = error;
//					for (int c = 0; c < colors; c++)
//					{
//						Reconstructed(x, y, c) = TmpAlpha[i] * float(Right(xfg, y, c)) + (1 - TmpAlpha[i])* float(Right(xbg, y, c));
//					}
//					
//				}
//			}
//		}
//	}
//}
//
////#pragma omp parallel for
////		for (long i = 0; i < HW; i++)
////		{
////			const int x = i / height;
////			const int y = i % height;
////			const int xr = x - dispFg < 0 ? 0 : (x - dispFg >= width ? width - 1 : x - dispFg);
////
////			float diff = 0;
////			for (int c = 0; c < colors; c++)
////			{
////				diff += (int(Left(x, y, c)) - Right(x, y, c)) * (int(Left(x, y, c)) - Right(x, y, c));
////			}
////
////			Beta[i] = sqrt(diff) / (3 * 255);
////		}
//	
//		#pragma omp parallel
//		{
//			std::unique_ptr<float[]> best_alphas(new float[window]);
//			
//			
//			#pragma omp for
//			for (long i = 0; i < HW; i++)
//			{
//				const int x = i / height;
//				const int y = i % height;
//				const int xfg = x - dispFg < 0 ? 0 : (x - dispFg >= width ? width - 1 : x - dispFg);
//				float best_err = FLT_MAX;
//				float a_opt = 0;
//				for (int bi = fi + 1; bi < layers; bi++)
//				//const int bi = 0;
//				{
//					const int dispBg = maxdisp - bi;
//					const int xbg = x - dispBg < 0 ? 0 : (x - dispBg >= width ? width - 1 : x - dispBg);
//
//					std::unique_ptr<float[]> alphas(new float[window]);
//
//					//for (float ai = 0.01; ai < 1; ai += 0.03)
//					//{
//
//						int pixels = 0;
//						float err = 0;
//						for (int j = 0; j < window; j++)
//						{
//							const int dx = j / diameter - radius;
//							const int dy = j % diameter - radius;
//							const int xL = x + dx;
//							const int yL = y + dy;
//							const int xRf = (xfg + dx) < 0 ? 0 : ((xfg + dx) >= width ? width - 1 : (xfg + dx));
//							const int xRb = (xbg + dx) < 0 ? 0 : ((xbg + dx) >= width ? width - 1 : (xbg + dx));
//							////const int xRb = xbg + dx;
//
//							if (xL < 0 || xL >= width || yL < 0 || yL >= height)
//							{
//								continue;
//							}
//														
//							for (int c = 0; c < colors; c++)
//							{
//								const float a = (float(Left(xL, yL, c)) - float(Right(xRb, yL, c))) / (float(Right(xRf, yL, c)) - float(Right(xRb, yL, c)));
//								alphas[j] += a < 0.f ? 0.f : a > 1.f ? 1.f : a; 
//							}
//							alphas[j] /= colors;
//
//							//alphas[j] = 1.f - std::min<float>(1.0, ai*Beta(xL, yL));
//
//							for (int c = 0; c < colors; c++)
//							{
//								err += abs(float(Left(xL, yL, c)) - alphas[j] * float(Right(xRf, yL, c)) - (1 - alphas[j]) * float(Right(xRb, yL, c)));
//								//err += abs(float(Left(xL, yL, c)) - float(Right(xRf, yL, c)));
//							}
//
//							pixels++;
//						}
//
//
//						err /= (pixels*colors);
//						if (err < best_err)
//						{
//							best_err = err;
//							//a_opt = ai;
//							for (int j = 0; j < window; j++)
//							{
//								best_alphas[j] = alphas[j];
//							}
//						}
//					//}
//					//mexPrintf("%f, ", a_opt);
//				}
//
//				// apply best_alphas here
//				const float weight = exp(-best_err / b);
//				for (int j = 0; j < window; j++)
//				{
//					const int dx = j / diameter - radius;
//					const int dy = j % diameter - radius;
//					const int xL = x + dx;
//					const int yL = y + dy;
//					if (xL < 0 || xL >= width || yL < 0 || yL >= height)
//					{
//						continue;
//					}
//					CostAlpha(xL, yL, fi) += best_alphas[j] * weight;
//					CostWeights(xL, yL, fi) += weight;
//					CostError(xL, yL, fi) += best_err/window;
//				}
//			}
//		}
//
//	}
//
//	#pragma omp parallel for
//	for (long i = 0; i < HW; i++)
//	{
//		const int x = i / height;
//		const int y = i % height;
//
//		float best_cost = FLT_MAX;
//		
//		for (int fi = 0; fi < layers; fi++)
//		{
//			const int dispFg = maxdisp - fi;
//			if (CostError(x, y, fi) < best_cost)
//			{
//				best_cost = CostError(x, y, fi);
//				DispL[i] = dispFg;
//				AlphaL[i] = CostAlpha(x, y, fi) / CostWeights(x, y, fi);
//			}
//		}
//	}

//}