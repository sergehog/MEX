

/* this ALWAYS GENERATED file contains the IIDs and CLSIDs */

/* link this file in with the server and any clients */


 /* File created by MIDL compiler version 7.00.0555 */
/* at Fri Dec 27 19:31:47 2013
 */
/* Compiler settings for win64\mwcomutil.idl:
    Oicf, W1, Zp8, env=Win64 (32b run), target_arch=IA64 7.00.0555 
    protocol : dce , ms_ext, c_ext, robust
    error checks: allocation ref bounds_check enum stub_data 
    VC __declspec() decoration level: 
         __declspec(uuid()), __declspec(selectany), __declspec(novtable)
         DECLSPEC_UUID(), MIDL_INTERFACE()
*/
/* @@MIDL_FILE_HEADING(  ) */

#pragma warning( disable: 4049 )  /* more than 64k source lines */


#ifdef __cplusplus
extern "C"{
#endif 


#include <rpc.h>
#include <rpcndr.h>

#ifdef _MIDL_USE_GUIDDEF_

#ifndef INITGUID
#define INITGUID
#include <guiddef.h>
#undef INITGUID
#else
#include <guiddef.h>
#endif

#define MIDL_DEFINE_GUID(type,name,l,w1,w2,b1,b2,b3,b4,b5,b6,b7,b8) \
        DEFINE_GUID(name,l,w1,w2,b1,b2,b3,b4,b5,b6,b7,b8)

#else // !_MIDL_USE_GUIDDEF_

#ifndef __IID_DEFINED__
#define __IID_DEFINED__

typedef struct _IID
{
    unsigned long x;
    unsigned short s1;
    unsigned short s2;
    unsigned char  c[8];
} IID;

#endif // __IID_DEFINED__

#ifndef CLSID_DEFINED
#define CLSID_DEFINED
typedef IID CLSID;
#endif // CLSID_DEFINED

#define MIDL_DEFINE_GUID(type,name,l,w1,w2,b1,b2,b3,b4,b5,b6,b7,b8) \
        const type name = {l,w1,w2,{b1,b2,b3,b4,b5,b6,b7,b8}}

#endif !_MIDL_USE_GUIDDEF_

MIDL_DEFINE_GUID(IID, IID_IMWUtil,0xC47EA90E,0x56D1,0x11d5,0xB1,0x59,0x00,0xD0,0xB7,0xBA,0x75,0x44);


MIDL_DEFINE_GUID(IID, LIBID_MWComUtil,0xAE860D07,0x0BBC,0x4F18,0x9A,0xB0,0xBC,0xA5,0xA1,0xCC,0x6D,0x85);


MIDL_DEFINE_GUID(CLSID, CLSID_MWField,0x9B2E62B7,0xFB4D,0x4261,0x9A,0x46,0x80,0x52,0x9C,0x23,0xF6,0x2F);


MIDL_DEFINE_GUID(CLSID, CLSID_MWStruct,0xB6A79FD5,0x8D91,0x4094,0xAD,0x2F,0x29,0x67,0xAB,0x8F,0x7E,0x34);


MIDL_DEFINE_GUID(CLSID, CLSID_MWComplex,0xC8A1CF4F,0x18FA,0x4E0A,0x86,0x71,0x9B,0x5F,0x67,0xAC,0x5D,0xC3);


MIDL_DEFINE_GUID(CLSID, CLSID_MWSparse,0xE2CA8603,0x210F,0x4427,0xBE,0xF2,0xBA,0x08,0x08,0x3E,0x04,0xE7);


MIDL_DEFINE_GUID(CLSID, CLSID_MWArg,0x26504ED1,0x1C90,0x4CA0,0x8E,0x44,0x87,0x72,0x5A,0xCB,0x7F,0xF0);


MIDL_DEFINE_GUID(CLSID, CLSID_MWArrayFormatFlags,0x75E155B8,0xCE14,0x460B,0xB6,0xC3,0xA7,0x26,0xE5,0x39,0x21,0xA1);


MIDL_DEFINE_GUID(CLSID, CLSID_MWDataConversionFlags,0xC99CE902,0xE83C,0x4E30,0x8B,0xF6,0xFC,0x8C,0xEE,0x52,0x53,0x88);


MIDL_DEFINE_GUID(CLSID, CLSID_MWUtil,0x8FADE1C1,0x9450,0x4C1D,0x9A,0x7B,0xB8,0x56,0xBF,0x3D,0x55,0xF2);


MIDL_DEFINE_GUID(CLSID, CLSID_MWFlags,0x44B6A0CD,0xFDAA,0x4BC0,0x9A,0xAB,0xDE,0x23,0x04,0xAC,0x84,0xB9);

#undef MIDL_DEFINE_GUID

#ifdef __cplusplus
}
#endif



