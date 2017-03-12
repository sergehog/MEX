/** very simple, compile-time matrix library :-)
*	Cannot be used with variable-size matrixes! 
*	@file mymath.h
*	@date 20.06.2012
*	@author Sergey Smirnov <sergey.smirnov@tut.fi>
*/


#pragma once

#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}
#endif // __cplusplus

namespace mymath
{

	//template<unsigned COLS, typename T=float> class RowVector;
	//template<unsigned ROWS, typename T=float> class ColVector;

	template<unsigned ROWS, unsigned COLS, typename T=float>
	class Matrix
	{

	public:
		T data[ROWS][COLS];
		
		//! Fill-in matrix with some value
		Matrix(T value = T(0))
		{
			for(unsigned y=0; y<ROWS; y++)
			{
				for(unsigned x=0; x<COLS; x++)
				{
					data[y][x] = value;
				}
			}
		};

		Matrix(const mxArray* image)
		{			
			T* im_data = static_cast<T*>(mxGetData(image));
			const unsigned height = (mxGetDimensions(image))[0];

			for(unsigned y=0; y<ROWS; y++)
			{
				for(unsigned x=0; x<COLS; x++)
				{
					data[y][x] = im_data[y + x*height];
				}
			}
		};
		
		//C = A*B; , where A = this
		template<unsigned COLSB>
		Matrix<ROWS, COLSB, T> multiply(Matrix<COLS, COLSB, T> &B)
		{
			Matrix<ROWS, COLSB> C;
			#pragma omp parallel for
			for(int y=0; y<ROWS; y++)
			{
				#pragma omp parallel for
				for(int x=0; x<COLSB; x++)
				{
					C.data[y][x] = 0;
					for(int i=0; i<COLS; i++)
					{
						C.data[y][x] += data[y][i]*B.data[i][x];
					}
				}
			}	
			
			return C;
		};

		// C = A + B;
		Matrix<ROWS, COLS, T> operator +(Matrix<ROWS, COLS, T> &B)
		{
			Matrix<ROWS, COLS> C;
			#pragma omp parallel for
			for(unsigned y=0; y<ROWS; y++)
			{
				#pragma omp parallel for
				for(unsigned x=0; x<COLSB; x++)
				{
					C.data[y][x] = data[y][i] + B.data[i][x];
				}
			}	
			
			return C;
		};

		// A = A+B;  A += B; in-memory addition
		void operator +=(Matrix<ROWS, COLS, T> &B)
		{			
			#pragma omp parallel for
			for(unsigned y=0; y<ROWS; y++)
			{
				#pragma omp parallel for
				for(unsigned x=0; x<COLSB; x++)
				{
					data[y][x] = data[y][i] + B.data[i][x];
				}
			}				
		};		


		Matrix<COLS, ROWS, T> transpose()
		{
			Matrix<COLS, ROWS> C;
			for(unsigned y=0; y<ROWS; y++)
			{
				for(unsigned x=0; x<COLS; x++)
				{
					C.data[x][y] = data[y][x];
				}
			}
			return C;
		};


		Matrix<ROWS, COLS, T> invert()
		{
			Matrix<ROWS, COLS> C;

			return C;
		};

		
		//RowVector<COLS, T> operator[](unsigned i)
		//{
		//	RowVector<COLS, T> C;
		//	for(unsigned j=0; j<COLS; j++)
		//	{
		//		C.data[j] = data[i][j];
		//	}
		//	return C;
		//};

		void mexPrint(void)
		{
			for(unsigned y=0; y<ROWS; y++)
			{
				for(unsigned x=0; x<COLS; x++)
				{
					mexPrintf("%f\t", data[y][x]);
				}
				mexPrintf("\n");
			}
		}
	};


	//template<unsigned COLS, typename T>
	//class Matrix<1, COLS, T>
	//{
	//	RowVector<COLS, T> operator()(RowVector<COLS, T>)
	//	{
	//		RowVector<COLS, T> A;
	//		for(unsigned i=0 i<COLS; i++)
	//		{
	//			A.data[0][i] = data[0][i];
	//		}
	//	}
	//};


	template<unsigned COLS, typename T=float>
	class RowVector : public Matrix<1, COLS, T>
	{
	public: 
		RowVector(T value = T(0)) : Matrix(value)
		{
		};

		RowVector(const mxArray* image) : Matrix(image)
		{
		};

		T operator[] (int i)
		{
			return data[i];
		}

	};

	//template<unsigned ROWS, typename T=float>
	//class ColVector : public Matrix<ROWS, 1, T>
	//{
	//public: 
	//	ColVector(T value = T(0)) : Matrix(value)
	//	{
	//	};

	//	ColVector(const mxArray* image) : Matrix(image)
	//	{
	//	};

	//	T operator[] (int i)
	//	{
	//		return data[i];
	//	}

	//};

	bool invertMatrix4x4(Matrix<4, 4> *matrix, Matrix<4, 4> *out)
	{			
		float *m = matrix->data[0];
		float *o = out->data[0];
		float *inv = new float[16];
		
		float det;

		//float inv[16], det;
		int i;

		inv[0] = m[5]  * m[10] * m[15] - 
					m[5]  * m[11] * m[14] - 
					m[9]  * m[6]  * m[15] + 
					m[9]  * m[7]  * m[14] +
					m[13] * m[6]  * m[11] - 
					m[13] * m[7]  * m[10];

		inv[4] = -m[4]  * m[10] * m[15] + 
					m[4]  * m[11] * m[14] + 
					m[8]  * m[6]  * m[15] - 
					m[8]  * m[7]  * m[14] - 
					m[12] * m[6]  * m[11] + 
					m[12] * m[7]  * m[10];

		inv[8] = m[4]  * m[9] * m[15] - 
					m[4]  * m[11] * m[13] - 
					m[8]  * m[5] * m[15] + 
					m[8]  * m[7] * m[13] + 
					m[12] * m[5] * m[11] - 
					m[12] * m[7] * m[9];

		inv[12] = -m[4]  * m[9] * m[14] + 
					m[4]  * m[10] * m[13] +
					m[8]  * m[5] * m[14] - 
					m[8]  * m[6] * m[13] - 
					m[12] * m[5] * m[10] + 
					m[12] * m[6] * m[9];

		inv[1] = -m[1]  * m[10] * m[15] + 
					m[1]  * m[11] * m[14] + 
					m[9]  * m[2] * m[15] - 
					m[9]  * m[3] * m[14] - 
					m[13] * m[2] * m[11] + 
					m[13] * m[3] * m[10];

		inv[5] = m[0]  * m[10] * m[15] - 
					m[0]  * m[11] * m[14] - 
					m[8]  * m[2] * m[15] + 
					m[8]  * m[3] * m[14] + 
					m[12] * m[2] * m[11] - 
					m[12] * m[3] * m[10];

		inv[9] = -m[0]  * m[9] * m[15] + 
					m[0]  * m[11] * m[13] + 
					m[8]  * m[1] * m[15] - 
					m[8]  * m[3] * m[13] - 
					m[12] * m[1] * m[11] + 
					m[12] * m[3] * m[9];

		inv[13] = m[0]  * m[9] * m[14] - 
					m[0]  * m[10] * m[13] - 
					m[8]  * m[1] * m[14] + 
					m[8]  * m[2] * m[13] + 
					m[12] * m[1] * m[10] - 
					m[12] * m[2] * m[9];

		inv[2] = m[1]  * m[6] * m[15] - 
					m[1]  * m[7] * m[14] - 
					m[5]  * m[2] * m[15] + 
					m[5]  * m[3] * m[14] + 
					m[13] * m[2] * m[7] - 
					m[13] * m[3] * m[6];

		inv[6] = -m[0]  * m[6] * m[15] + 
					m[0]  * m[7] * m[14] + 
					m[4]  * m[2] * m[15] - 
					m[4]  * m[3] * m[14] - 
					m[12] * m[2] * m[7] + 
					m[12] * m[3] * m[6];

		inv[10] = m[0]  * m[5] * m[15] - 
					m[0]  * m[7] * m[13] - 
					m[4]  * m[1] * m[15] + 
					m[4]  * m[3] * m[13] + 
					m[12] * m[1] * m[7] - 
					m[12] * m[3] * m[5];

		inv[14] = -m[0]  * m[5] * m[14] + 
					m[0]  * m[6] * m[13] + 
					m[4]  * m[1] * m[14] - 
					m[4]  * m[2] * m[13] - 
					m[12] * m[1] * m[6] + 
					m[12] * m[2] * m[5];

		inv[3] = -m[1] * m[6] * m[11] + 
					m[1] * m[7] * m[10] + 
					m[5] * m[2] * m[11] - 
					m[5] * m[3] * m[10] - 
					m[9] * m[2] * m[7] + 
					m[9] * m[3] * m[6];

		inv[7] = m[0] * m[6] * m[11] - 
					m[0] * m[7] * m[10] - 
					m[4] * m[2] * m[11] + 
					m[4] * m[3] * m[10] + 
					m[8] * m[2] * m[7] - 
					m[8] * m[3] * m[6];

		inv[11] = -m[0] * m[5] * m[11] + 
					m[0] * m[7] * m[9] + 
					m[4] * m[1] * m[11] - 
					m[4] * m[3] * m[9] - 
					m[8] * m[1] * m[7] + 
					m[8] * m[3] * m[5];

		inv[15] = m[0] * m[5] * m[10] - 
					m[0] * m[6] * m[9] - 
					m[4] * m[1] * m[10] + 
					m[4] * m[2] * m[9] + 
					m[8] * m[1] * m[6] - 
					m[8] * m[2] * m[5];

		det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

		if (det == 0)
			return false;

		det = 1.0 / det;

		for (i = 0; i < 16; i++)
			o[i] = inv[i] * det;

		delete[] inv;

		return true;
	}



};

	
