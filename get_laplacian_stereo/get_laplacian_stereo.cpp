/**
*	@file get_laplacian_stereo.cpp
*	@date 12.08.2015
*	@author Sergey Smirnov <sergey.smirnov@tut.fi>
*/

#include "../common/meximage.h"
#include <float.h>
#include <cmath>
#include <atomic>
#include <algorithm>
#include <vector>
#include <Eigen\Dense>

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

#define IsNonZero(d) ((d)!=0.0)
#define colors 3
#define radius 1
#define diameter (radius*2 + 1)
#define window (diameter*diameter)

struct sparseEntry
{
	//sparseEntry(size_t _row, size_t _col, double _val) : row(_row), col(_col), value(_val)
	//{}

	size_t row;
	size_t col;
	double value;
	//
	//private: 
	//	sparseEntry() : row(0), col(0), value(NAN)
	//	{}
	bool operator<(sparseEntry &entry)
	{
		return col == entry.col ? row < entry.row : col < entry.col;
	}
};

int _cdecl compareSparseEntry(const void * entry1, const void* entry2)
{
	sparseEntry* s1 = (sparseEntry*)entry1;
	sparseEntry* s2 = (sparseEntry*)entry2;
	int colDiff = int(s1->col) - int(s2->col);
	int rowDiff = int(s1->row) - int(s2->row);
	return colDiff == 0 ? rowDiff : colDiff;
}

using namespace mymex;
using namespace Eigen;
void mexFunction(int nout, mxArray* output[], int in, const mxArray* input[])
{
#ifndef _DEBUG
	omp_set_num_threads(std::max<int>(8, omp_get_max_threads()));
	omp_set_dynamic(std::max<int>(4, omp_get_max_threads() / 2));;
#endif	

	if (in < 4 || in > 6 || nout != 1 || mxGetClassID(input[0]) != mxDOUBLE_CLASS || mxGetClassID(input[1]) != mxDOUBLE_CLASS || mxGetClassID(input[2]) != mxINT16_CLASS || mxGetClassID(input[3]) != mxLOGICAL_CLASS)
	{
		mexErrMsgTxt("USAGE: [L] = get_laplacian(double(L), double(R), int16(DispL), logical(ValidL), <epsilon, radius>);\n");
	}

	MexImage<double> Left(input[0]);
	MexImage<double> Right(input[1]);
	MexImage<int16_t> DispL(input[2]);
	MexImage<bool> ValidL(input[3]);
	const double epsilon = in > 4 ? std::max<double>(0.0, mxGetScalar(input[4])) : 0.0000001;

	//const int radius = in > 3 ? std::max<int>(1,int(mxGetScalar(input[3]))) : 1;
	//const int diameter = radius * 2 + 1;
	//const int window = diameter * diameter;
	const int height = Left.height;
	const int width = Left.width;
	const int HW = Left.layer_size;
	//const int colors = Left.layers;
	if (Right.width != width || DispL.width != width || ValidL.width != width)
	{
		mexErrMsgTxt("Resolution must coincide!");
	}
	if (Right.height != height || DispL.height != height || ValidL.height != height)
	{
		mexErrMsgTxt("Resolution must coincide!");
	}

	MexImage<bool> ValidRight(width, height);
	MexImage<bool> ErrodedLeft(width, height);
	MexImage<bool> ErrodedRight(width, height);

	ErrodedLeft.setval(true);

	// Errosion here
#pragma omp parallel for
	for (int x = 0; x < width; x++)
	{
		for (int y = 0; y < height; y++)
		{
			if (Trimap(x, y) != 0 && Trimap(x, y) != 2 && Trimap(x, y) != 255)
			{
				for (int xi = std::max<int>(0, x - radius); xi <= std::min<int>(width - 1, x + radius); xi++)
				{
					for (int yi = std::max<int>(0, y - radius); yi <= std::min<int>(height - 1, y + radius); yi++)
					{
						ErrodedLeft(xi, yi) = 0;
					}
				}
			}
		}
	}

	// number of pixels, used for adjastency calculation
	size_t elements_num = 0;
#pragma omp parallel for reduction(+: elements_num)
	for (long i = 0; i < HW; i++)
	{
		const int x = i / height;
		const int y = i % height;
		if (x < radius || x > width - radius - 1 || y < radius || y > height - radius - 1)
		{
			continue;
		}

		elements_num += (!ErrodedLeft[i]);
	}

	// number of adjastency elements in "entries" (some of them will be summed up)
	size_t const tlen = elements_num*window*window;

	std::vector<sparseEntry> entries(tlen);

	// in order to parallelize further processing, let's write all indexes in a separate array
	std::vector < long > elements(elements_num);
	for (long i = 0, j = 0; i < HW; i++)
	{
		const int x = i / height;
		const int y = i % height;

		if (x < radius || x >= width - radius || y < radius || y >= height - radius || ErrodedLeft[i])
		{
			continue;
		}
		elements[j] = i;
		j++;
	}

#pragma omp parallel for
	for (long ind_e = 0; ind_e < elements_num; ind_e++)
	{
		const long index = elements[ind_e];
		const int x = index / height;
		const int y = index % height;
		size_t k = ind_e * window*window;

		Matrix<double, window, colors> winI(window, colors);

		Vector3d win_mu(0.0, 0.0, 0.0);
		win_mu[0] = win_mu[1] = win_mu[2] = 0.0;

		for (int xi = x - radius, i = 0; xi <= x + radius; xi++)
		{
			for (int yi = y - radius; yi <= y + radius; yi++, i++)
			{
				winI(i, 0) = Left(xi, yi, 0);
				winI(i, 1) = Left(xi, yi, 1);
				winI(i, 2) = Left(xi, yi, 2);
				win_mu[0] += winI(i, 0);
				win_mu[1] += winI(i, 1);
				win_mu[2] += winI(i, 2);
			}
		}
		win_mu[0] /= window;
		win_mu[1] /= window;
		win_mu[2] /= window;

		Matrix3d win_var = (winI.transpose()*winI / window - win_mu*win_mu.transpose() + epsilon * Matrix3d::Identity() / window).inverse();

		//#pragma omp parallel for
		for (int i = 0; i < window; i++)
		{
			winI(i, 0) -= win_mu[0];
			winI(i, 1) -= win_mu[1];
			winI(i, 2) -= win_mu[2];
		}

		Matrix<double, window, window> tvals = ((winI * win_var * winI.transpose() + Matrix<double, window, window>::Constant(1.0))) / window;
		//#pragma omp parallel for
		for (int i = 0; i < window; i++)
		{
			const int xi = x + (i / diameter) - radius;
			const int yi = y + (i % diameter) - radius;
			const int index_i = xi*height + yi;
			//
			for (int j = 0; j < window; j++)
			{
				const int xj = x + (j / diameter) - radius;
				const int yj = y + (j % diameter) - radius;
				const int index_j = xj*height + yj;

				entries[k].value = tvals(j, i);
				entries[k].row = (size_t)index_i;
				entries[k].col = (size_t)index_j;

				k++;
			}
		}
	}

	//std::sort(entries.begin(), entries.end());
	std::qsort(&entries[0], tlen, sizeof(sparseEntry), compareSparseEntry);
	output[0] = mxCreateSparse(HW, HW, size_t(tlen / 2.5), mxComplexity::mxREAL);
	double* vals = mxGetPr(output[0]);
	mwIndex *row_inds = mxGetIr(output[0]);
	mwIndex *col_inds = mxGetJc(output[0]);

	// current column index (goes from 0 to HW-1 and then HW)
	size_t col = 0;

	// current index in sorted values list
	size_t k = 0;

	// current index in the sparse matrix
	size_t s = 0;

	while (k < tlen && col < HW)
	{
		// process empty columns if any
		while (col < entries[k].col && col < HW)
		{
			// add empty column to sparse matrix
			col_inds[col] = s;
			//mexPrintf("\nColumn %d (%d): ", col, s);
			col++;
		}

		if (col == HW)
		{
			// last column was empty
			break;
		}

		// current column is non-empty
		col_inds[col] = s;
		//mexPrintf("\nColumn %d (%d): ", col, s);
		// process values, until they belong to the current column
		while (k < tlen && col == entries[k].col && col < HW)
		{
			double value = entries[k].value;
			size_t row = entries[k].row;
			k++;

			// there might be several values at the same row - add them together!
			while (k < tlen && col == entries[k].col && row == entries[k].row /*&& row < HW*/)
			{
				value += entries[k].value;
				k++;
			}
			if (k < tlen && entries[k].row < row && col == entries[k].col)
			{
				mexPrintf("Sorting Error!\n");
			}
			if (value != 0.0)
			{
				row_inds[s] = row;
				vals[s] = value;
				//mexPrintf("%d (%f), ", row, value);
				s++;
			}

		}
		col++;
	}
	while (col <= HW)
	{
		col_inds[col] = s;
		col++;
	}
	entries.clear();
	mexPrintf("Number of non-empty entries: %d (allocated %d) \n", col_inds[HW], size_t(tlen / 2.5));
}
