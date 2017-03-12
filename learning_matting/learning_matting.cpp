/** C++ reimplementation of Learning-Based Alpha-Matting approach
*	@file learning_matting.cpp
*	@date 18.09.2015
*	@author Sergey Smirnov <sergey.smirnov@tut.fi>
*/

#include "../common/meximage.h"
#include <float.h>
#include <cmath>
#include <atomic>
#include <memory>
#include <algorithm>
#include <vector>
#include <time.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}
#endif // __cplusplus
#ifndef _DEBUG
#include <omp.h>
#endif

#ifdef WIN32
#define isnan _isnan
#endif

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")

//#define IsNonZero(d) ((d)!=0.0)
//#define colors 3

using namespace mymex;
using namespace Eigen;

#define diameter (radius*2+1)
#define window (diameter*diameter)
//#define kernel_trick 1

static Eigen::ConjugateGradient<SparseMatrix<double>, Eigen::Upper> gradient_solver;
static bool gradient_solver_created = false;
static int gradient_solver_width = 0;
static int gradient_solver_height = 0;
static int gradient_solver_radius = 0;

static Eigen::SimplicialLDLT<SparseMatrix<double>> llt_solver;
static bool llt_solver_created = false;
static int llt_solver_width = 0;
static int llt_solver_height = 0;
static int llt_solver_radius = 0;


template<int radius>
void compute_alpha_color(const MexImage<float> &Image, const MexImage<bool> &ProcessArea, Eigen::Triplet<double> * const triplets, const size_t elements_num, const double c, const double lambda, const double trick_sigma)
{
	if (Image.layers == 1)
	{
		compute_alpha<radius, 1>(Image, ProcessArea, triplets, elements_num, c, lambda, trick_sigma);
	}
	else if (Image.layers == 2)
	{
		compute_alpha<radius, 2>(Image, ProcessArea, triplets, elements_num, c, lambda, trick_sigma);
	}
	else if (Image.layers == 3)
	{
		compute_alpha<radius, 3>(Image, ProcessArea, triplets, elements_num, c, lambda, trick_sigma);
	}
	else if (Image.layers == 4)
	{
		compute_alpha<radius, 4>(Image, ProcessArea, triplets, elements_num, c, lambda, trick_sigma);
	}
	else if (Image.layers == 5)
	{
		compute_alpha<radius, 5>(Image, ProcessArea, triplets, elements_num, c, lambda, trick_sigma);
	}
	else if (Image.layers == 6)
	{
		compute_alpha<radius, 6>(Image, ProcessArea, triplets, elements_num, c, lambda, trick_sigma);
	}
	//else if (Image.layers == 7)
	//{
	//	compute_alpha<radius, 7>(Image, ProcessArea, triplets, elements_num, c, lambda, trick_sigma);
	//}
	//else if (Image.layers == 8)
	//{
	//	compute_alpha<radius, 8>(Image, ProcessArea, triplets, elements_num, c, lambda, trick_sigma);
	//}
	//else if (Image.layers == 9)
	//{
	//	compute_alpha<radius, 9>(Image, ProcessArea, triplets, elements_num, c, lambda, trick_sigma);
	//}
	//else if (Image.layers == 10)
	//{
	//	compute_alpha<radius, 10>(Image, ProcessArea, triplets, elements_num, c, lambda, trick_sigma);
	//}
	//else if (Image.layers == 11)
	//{
	//	compute_alpha<radius, 11>(Image, ProcessArea, triplets, elements_num, c, lambda, trick_sigma);
	//}
	//else if (Image.layers == 12)
	//{
	//	compute_alpha<radius, 12>(Image, ProcessArea, triplets, elements_num, c, lambda, trick_sigma);
	//}
	//else if (Image.layers == 13)
	//{
	//	compute_alpha<radius, 13>(Image, ProcessArea, triplets, elements_num, c, lambda, trick_sigma);
	//}
	//else if (Image.layers == 14)
	//{
	//	compute_alpha<radius, 14>(Image, ProcessArea, triplets, elements_num, c, lambda, trick_sigma);
	//}
	else
	{
		mexErrMsgTxt("Too many colors in your image!\n");
	}
}

template<int radius, int colors>
void compute_alpha(const MexImage<float> &Image, const MexImage<bool> &ProcessArea, Eigen::Triplet<double> * const triplets, const size_t elements_num, const double c, const double lambda, const double trick_sigma)
{
	if (isnan(trick_sigma) || trick_sigma == 0.0)
	{
		compute_alpha_trick<radius, colors, 0>(Image, ProcessArea, triplets, elements_num, c, lambda, trick_sigma);
	}
	else
	{
		compute_alpha_trick<radius, colors, 1>(Image, ProcessArea, triplets, elements_num, c, lambda, trick_sigma);
	}
}


template<int radius, int colors, bool trick>
void compute_alpha_trick(const MexImage<float> &Image, const MexImage<bool> &ProcessArea, Eigen::Triplet<double> * const triplets, const size_t elements_num, const double c, const double lambda, const double trick_sigma)
{
	const int height = Image.height;
	const int width = Image.width;
	const int HW = Image.layer_size;
	
	//clock_t timer1 = clock();	
	
	Eigen::Matrix<double, window, window> I0 = Eigen::Matrix<double, window, window>::Identity();
	Eigen::Matrix<double, window, window> I = Eigen::Matrix<double, window, window>::Identity();
	I0(window - 1, window - 1) = 0.0;
	
	// required pixels in a single array for better parallalization
	std::vector<long> elements;

	for (long i = 0; i < HW; i++)
	{
		const int x = i / height;
		const int y = i % height;

		if (x < radius || x >= width - radius || y < radius || y >= height - radius || !ProcessArea[i])
		{
			continue;
		}
		elements.push_back(i);
	}


	//clock_t timer4 = clock();
	#pragma omp parallel 
	{
		//Eigen::Matrix<double, window, window> fenmu;
		Eigen::Matrix<double, window, window> F;
		Matrix<double, window, colors + 1> Xi;
		for (int i = 0; i < window; i++)
		{
			Xi(i, colors) = 1.0;
		}
		#pragma omp for schedule(dynamic)
		for (long j = 0; j < elements_num; j++)
		{
			const long index = elements[j];
			const int x = index / height;
			const int y = index % height;

			// data matrix
			#pragma omp parallel for
			for (int i = 0; i < window; i++)
			{
				const int dx = i / diameter - radius;
				const int dy = i % diameter - radius;
				const int xi = x + dx;
				const int yi = y + dy;
				for (int c = 0; c < colors; c++)
				{
					if (trick)
					{
						Xi(i, c) = double(Image(xi, yi, c)) / 255;
					}
					else
					{
						Xi(i, c) = isnan(Image(xi, yi, c)) ? 0 : double(Image(xi, yi, c)) / 255;
					}
					
				}
			}

			if (trick)
			{
				for (int i1 = 0; i1 < window; i1++)
				{
					F(i1, i1) = 1.0;
					for (int i2 = i1 + 1; i2 < window; i2++)
						//for (int i2 = 0; i2 < window; i2++)
					{
						double diff = 0.0;
						for (int c = 0; c < colors; c++)
						{
							diff += (isnan(Xi(i1, c)) || isnan(Xi(i2, c))) ? 0 :  (Xi(i1, c) - Xi(i2, c)) * (Xi(i1, c) - Xi(i2, c));
						}

						const double value = exp(-diff / trick_sigma);
						F(i1, i2) = value;
						F(i2, i1) = value;
					}
				}
			}
			else
			{
				F = (Xi*Xi.transpose());
			}

			F = F * ((F + I0*lambda).inverse()); // fenmu = F + I0*lambda;
			F = (I - F).transpose() * (I - F); // normalization

			
			long k = j*window*window;			
			for (int i = 0; i < window; i++)
			{
				const int dxi = i / diameter - radius;
				const int dyi = i % diameter - radius;
				const long indexi = (x + dxi)*height + (y + dyi);

				for (int j = 0; j < window; j++, k++)
				{
					const int dxj = j / diameter - radius;
					const int dyj = j % diameter - radius;
					const long indexj = (x + dxj)*height + (y + dyj);					
					triplets[k] = Eigen::Triplet<double>(indexj, indexi, F(i, j));
				}
			}
		}
	}

	
	
}

void mexFunction(int nout, mxArray* output[], int in, const mxArray* input[])
{
#ifndef _DEBUG
	omp_set_num_threads(std::max<int>(8, omp_get_max_threads()));
	omp_set_dynamic(std::max<int>(2, omp_get_max_threads()));
#endif	

	if (in < 2 || in > 7 || nout != 1 || mxGetClassID(input[0]) != mxSINGLE_CLASS || mxGetClassID(input[1]) != mxUINT8_CLASS)
	{
		mexErrMsgTxt("USAGE: [Alpha] = learning_matting(single(I), uint8(Trimap), {single(AlphaGuess)} <,radius, kernel_trick_sigma, lambda, c>);\n");
	}

	MexImage<float> Image(input[0]);
	MexImage<uint8_t> Trimap(input[1]);
	const int height = Image.height;
	const int width = Image.width;
	const int HW = Image.layer_size;
	const int colors = Image.layers;
	
	const bool solveWithGuess = (in > 2) && mxGetClassID(input[2]) == mxSINGLE_CLASS && mxGetWidth(input[2]) == width && mxGetHeight(input[2]) == height;
	const int offset = int(solveWithGuess);
	std::unique_ptr<MexImage<float>> Guess;
	
	if(solveWithGuess)
	{
		Guess.reset(new MexImage<float>(input[2]));
	}
	
	const int radius = in > 2 + offset ? std::max<int>(1, int(mxGetScalar(input[2 + offset]))) : 2;
	const double trick_sigma = in > 3 + offset ? std::max<double>(0.0, mxGetScalar(input[3 + offset])) : 0.0;
	const double lambda = in > 4 + offset ? std::max<double>(0.0, mxGetScalar(input[4 + offset])) : 0.0000001;
	const double c = in > 5 + offset ? std::max<double>(0.000001, mxGetScalar(input[5 + offset])) : 800.0;

	
	const mwSize dims[] = { (size_t)height, (size_t)width, 1 };

	clock_t timer1 = clock();

	// Shows where adjastency needs to be computed. 
	// The more Fg/Bg in Trimap, the sparsier matrix to solve!
	MexImage<bool> ProcessArea(width, height);
	ProcessArea.setval(0);

	// given values vector (pre-multiplied with c)
	Eigen::Matrix<double, -1, 1> vals(HW);

	// regularization for given elements (diagonal matrix)
	Eigen::SparseMatrix<double> C(HW, HW);
	C.setIdentity();
	
	// Errosion here
	#pragma omp parallel for schedule(dynamic)
	for (long i = 0; i < HW; i++)
	{
		const int x = i / height;
		const int y = i % height;
		if (Trimap(x, y) != 0 && Trimap(x, y) != 2 && Trimap(x, y) != 255)
		{
			C.coeffRef(i, i) = 0.0;
			vals(i) = 0.0;

			for (int xi = std::max<int>(radius, x - radius); xi <= std::min<int>(width - radius - 1, x + radius); xi++)
			{
				for (int yi = std::max<int>(radius, y - radius); yi <= std::min<int>(height - radius - 1, y + radius); yi++)
				{
					ProcessArea(xi, yi) = true;
				}
			}
		}
		else
		{
			C.coeffRef(i, i) = c;
			vals(i) = (Trimap(x, y) == 0 ? -1.0 : 1.0) *c;
		}
	}	

	// required pixels in a single array for better parallalization
	
	long j = 0;
	#pragma omp parallel for reduction(+:j)
	for (long i = 0; i < HW; i++)
	{
		const int x = i / height;
		const int y = i % height;

		if (x < radius || x >= width - radius || y < radius || y >= height - radius || !ProcessArea[i])
		{
			continue;
		}
		
		j++;
	}
	
	const size_t elements_num = j;
	std::unique_ptr<Eigen::Triplet<double>[]> triplets(new Eigen::Triplet<double>[window*window*elements_num]);
	
	clock_t timer2 = clock();

	//Matlab-allocated variables	
	output[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
	MexImage<float> Alpha(output[0]);
	if (radius == 1)
	{
		compute_alpha_color<1>(Image, ProcessArea, triplets.get(), elements_num, c, lambda, trick_sigma);
	}
	else if (radius == 2)
	{
		compute_alpha_color<2>(Image, ProcessArea, triplets.get(), elements_num, c, lambda, trick_sigma);
	} 
	else if (radius == 3)
	{
		compute_alpha_color<3>(Image, ProcessArea, triplets.get(), elements_num, c, lambda, trick_sigma);
	}
	else if (radius == 4)
	{
		compute_alpha_color<4>(Image, ProcessArea, triplets.get(), elements_num, c, lambda, trick_sigma);
	}
	else if (radius == 5)
	{
		compute_alpha_color<5>(Image, ProcessArea, triplets.get(), elements_num, c, lambda, trick_sigma);
	}
	else
	{
		mexErrMsgTxt("Radius is not supported!\n");
	}

	clock_t timer3 = clock();
	Eigen::SparseMatrix<double> Laplacian(HW, HW);
	Laplacian.setFromTriplets(&triplets[0], &triplets[window*window*elements_num]);

	clock_t timer4 = clock();
	//mexPrintf("size=%d, Lsize=%d\n", triplets.size(), Laplacian.nonZeros());
	//triplets.clear();
	triplets.reset();

	Eigen::SparseMatrix<double> D(HW, HW);
	D.setIdentity();

	Eigen::SparseMatrix<double> A = Laplacian + D*lambda + C;
	//Eigen::SimplicialLDLT<SparseMatrix<double>> solver;
	//Eigen::SimplicialLLT<SparseMatrix<double>> solver;

	
	Eigen::Matrix<double, -1, 1> AlphaEstimate(HW);

	if (solveWithGuess)
	{
		if (!(gradient_solver_created && width == gradient_solver_width && height == gradient_solver_height && radius == gradient_solver_radius))
		{
			gradient_solver.analyzePattern(A);
			gradient_solver_created = true;
			gradient_solver.setTolerance(0.000001);
			gradient_solver_width = width;
			gradient_solver_height = height;
			gradient_solver_radius = radius;
			mexPrintf("Gradient Solver created! \n");
		}
		else
		{
			mexPrintf("Gradient Solver restored! \n");
		}
		
		Matrix<double, -1, 1> GuessE(HW);
		for (long i = 0; i < HW; i++)
		{
			GuessE(i) = double(Guess->at(i)*2.0 - 1.0);
		}

		gradient_solver.factorize(A);
		AlphaEstimate = gradient_solver.solveWithGuess(vals, GuessE);
	}
	else
	{
		if (!(llt_solver_created && width == llt_solver_width && height == llt_solver_height && radius == llt_solver_radius))
		{
			llt_solver.analyzePattern(A);
			llt_solver_created = true;
			llt_solver_width = width;
			llt_solver_height = height;
			llt_solver_radius = radius;
			mexPrintf("LLT Solver created! \n");
		}
		else
		{
			mexPrintf("LLT Solver restored! \n");
		}
		
		llt_solver.factorize(A);
		AlphaEstimate = llt_solver.solve(vals);
	}	

	clock_t timer5 = clock();
	
	for (long i = 0; i < HW; i++)
	{
		int x = i / height;
		int y = i % height;
		const double value = (AlphaEstimate(i) + 1.0)/2;
		Alpha(x, y) = value < 0.00001 ? 0.f : (value > 0.99999 ? 1.f : value);
	}

	float sec1 = float(timer2 - timer1) / CLOCKS_PER_SEC;
	float sec2 = float(timer3 - timer2) / CLOCKS_PER_SEC;
	float sec3 = float(timer4 - timer3) / CLOCKS_PER_SEC;
	float sec4 = float(timer5 - timer4) / CLOCKS_PER_SEC;
	//float sec5 = float(timer6 - timer5) / CLOCKS_PER_SEC;
	//float sec6 = float(timer7 - timer6) / CLOCKS_PER_SEC;

	mexPrintf("Timers: \n timer1=%5.3f, timer2=%5.3f, timer3=%5.3f, timer4=%5.3f\n", sec1, sec2, sec3, sec4);
	
}
