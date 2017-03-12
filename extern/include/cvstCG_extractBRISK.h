/* Copyright 2012 The MathWorks, Inc. */

#ifndef _EXTRACTBRISK_
#define _EXTRACTBRISK_

#ifndef LIBMWEXTRACTBRISK_API
#    define LIBMWEXTRACTBRISK_API
#endif

#ifndef EXTERN_C
#  ifdef __cplusplus
#    define EXTERN_C extern "C"
#  else
#    define EXTERN_C extern
#  endif
#endif

#include "tmwtypes.h"

EXTERN_C LIBMWEXTRACTBRISK_API
int32_T extractBRISK_compute(const uint8_T * img, const int32_T nRows, const int32_T nCols,
                             real32_T * location, real32_T * metric,
                             real32_T * scale, real32_T * orientation,
                             int32_T * misc, const int32_T numKeyPoints,
                             void ** features, void ** keypoints);

EXTERN_C LIBMWEXTRACTBRISK_API
void extractBRISK_assignOutput(void *ptrDescriptors, void *ptrKeyPoints,
                               real32_T * location, real32_T * metric,
                               real32_T * scale, real32_T * orientation,
                               int32_T * misc, uint8_T * features);
#endif
