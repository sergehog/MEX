/** MATLAB MEX FOR Sparse ICP
* @file sparse_icp.cpp
* @author Sergey Smirnov <sergey.smirnov@tut.fi>
* @date 09.09.2016
* @copyright 3D Media Group / Tampere University of Technology
*/


#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}
#endif // __cplusplus

#include "../common/meximage.h"

#include <iostream>
#include <omp.h>
#include "icp.h"

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")

using namespace mymex;

//typedef double Scalar;
typedef Eigen::Matrix<double, 3, Eigen::Dynamic> Vertices;

void mexFunction(const int nout, mxArray* output[], const int in, const mxArray* input[])
{
	omp_set_num_threads(omp_get_max_threads());
	omp_set_dynamic(omp_get_max_threads() - 1);

	if (in < 2 || in > 4 || nout != 1 || mxGetClassID(input[0]) != mxDOUBLE_CLASS || mxGetClassID(input[1]) != mxDOUBLE_CLASS)
	{
		mexErrMsgTxt("USAGE: [XYZmoved] = spiral_matting(double(XYZstatic), double(XYZmoving), <p, double(NormalsMoving)>);");
	}

	MexImage<double> XYZstatic(input[0]);	
	MexImage<double> XYZmoving(input[1]);
	MexImage<double> *_Normals;

	if (XYZstatic.width != 3 || XYZstatic.height < 4 || XYZmoving.width != 3 || XYZmoving.height < 4)
	{
		mexErrMsgTxt("Check dimentions of input XYZ arrays!");
	}

	const double p = (in > 2) ? mxGetScalar(input[2]) : 0.5;
	
	if(in > 3)
	{
		_Normals = new MexImage<double>(input[3]);

		if (_Normals->width != 3 || _Normals->height != XYZmoving.height)
		{
			delete _Normals;
			mexErrMsgTxt("Check dimentions of Normals array!");
		}
	}
	else
	{
		_Normals = new MexImage<double>(1, 1);
	}
	MexImage<double> & Normals = *_Normals;	

	//const size_t dims4[] = { 4, 4, 1 };
	const size_t dims3[] = { XYZmoving.height, (size_t)3, 1 };
	
	//output[0] = mxCreateNumericArray(3, dims4, mxSINGLE_CLASS, mxREAL);
	output[0] = mxCreateNumericArray(3, dims3, mxDOUBLE_CLASS, mxREAL);
	MexImage<double> XYZmoved(output[0]);	

	///--- Model that source will be aligned to
	Vertices vertices_target;
	vertices_target.resize(Eigen::NoChange, XYZstatic.height);
	#pragma omp parallel for
	for (int i = 0; i < vertices_target.cols(); i++)
	{
		vertices_target(0, i) = XYZstatic(0, i);
		vertices_target(1, i) = XYZstatic(1, i);
		vertices_target(2, i) = XYZstatic(2, i);
	}

	///--- Model that will be rigidly transformed
	Vertices vertices_source;
	vertices_source.resize(Eigen::NoChange, XYZmoving.height);
	#pragma omp parallel for
	for (int i = 0; i < vertices_source.cols(); i++)
	{
		vertices_source(0, i) = XYZmoving(0, i);
		vertices_source(1, i) = XYZmoving(1, i);
		vertices_source(2, i) = XYZmoving(2, i);
	}

	

	
	if (in > 3)
	{
		Vertices normals;
		normals.resize(Eigen::NoChange, XYZmoving.height);
		#pragma omp parallel for
		for (int i = 0; i < normals.cols(); i++)
		{
			normals(0, i) = Normals(0, i);
			normals(1, i) = Normals(1, i);
			normals(2, i) = Normals(2, i);
		}
		if (p > 1)
		{
			ICP::Parameters pars;
			pars.p = (p-floor(p));
			ICP::point_to_plane(vertices_source, vertices_target, normals, pars);
		}
		else
		{
			SICP::Parameters pars;
			pars.p = p;
			SICP::point_to_plane<Vertices, Vertices, Vertices>(vertices_source, vertices_target, normals);
		}		
	}
	else
	{
		if (p > 1)
		{
			ICP::Parameters pars;
			pars.p = (p - floor(p));
			ICP::point_to_point(vertices_source, vertices_target, pars);			
		}
		else
		{
			SICP::Parameters pars;
			pars.p = p;
			SICP::point_to_point(vertices_source, vertices_target, pars);			
		}		
	}
	
	
	// copy vertices once again in order to find projection
	// (there is no direct way to find out the projection matrix)
	vertices_target.resize(Eigen::NoChange, XYZmoving.height);
	#pragma omp parallel for
	for (int i = 0; i < vertices_target.cols(); i++)
	{
		vertices_target(0, i) = XYZmoving(0, i);
		vertices_target(1, i) = XYZmoving(1, i);
		vertices_target(2, i) = XYZmoving(2, i);
	}



	#pragma omp parallel for
	for (int i = 0; i < vertices_source.cols(); i++)
	{
		XYZmoved(0, i) = vertices_source(0, i);
		XYZmoved(1, i) = vertices_source(1, i);
		XYZmoved(2, i) = vertices_source(2, i);				
	}

	vertices_source.resize(Eigen::NoChange, 1);
	vertices_target.resize(Eigen::NoChange, 1);
		
	delete _Normals;
	mexPrintf("Done! \n");
}