/*********************************************************************************************************************
 ** File: malloc.hpp
 ** Abstract:
 **
 ** Copyright 2004-2006 The MathWorks, Inc.
 *********************************************************************************************************************/

#ifndef __malloc_hpp__
#define __malloc_hpp__


/*!
 * standard malloc with exception handling for allocation failure
 */
EXTERN_C void *pmf_malloc(size_t n);


/*!
 * standard calloc function with exceptino handling for allocation failure
 * returned memory is memset to 0
 */
EXTERN_C void *pmf_calloc(size_t n, size_t size);


/*!
 * standard free function with 'NULL check'
 */
EXTERN_C void pmf_free(void *ptr);

#endif /* include guard */
