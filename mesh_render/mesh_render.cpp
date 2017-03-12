/** 
* @file mesh_render.cpp
* @author Sergey Smirnov <sergey.smirnov@tut.fi>
* @date 31.05.2016
* @copyright 3D Media Group / Tampere University of Technology
*/

#define GLM_FORCE_CXX11  
#include <glm/glm.hpp>
//#include <glm/gtc/matrix_transform.hpp>

#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}
#endif // __cplusplus

#include "../common/meximage.h"
#include <algorithm>
#include <memory>
#ifndef _DEBUG
#include <omp.h>
#endif