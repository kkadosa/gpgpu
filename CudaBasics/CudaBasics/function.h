#pragma once

// We tell NVCC that this function can be used both on the host and the device code.
// __CUDACC__ is only defined by NVCC, so __host__ and __device__ is ignored by the C++ compiler.
#ifdef __CUDACC__
__host__ __device__
#endif
inline unsigned int assocFunc(unsigned int i1, unsigned int i2)
{
	// Addition
	return i1 + i2;
	
	// Maximum
	//return (i1 >= i2) ? i1 : i2;

	// Minimum
	//return (i1 >= i2) ? i2 : i1;

	// Addition, with some expensive stuff that does not affect the result
	/*
	unsigned int iFinal = i1;
	float x = 2.0f*sin((float)i2)*cos((float)i1);
	float v = 2.3f;
	for (int i = 0; i < 5; ++i)
	{
		v += pow(x, v) * exp(sin(x));
	}
	if (x > 3.0f) // sine cannot be greater than 3, so this will not run, iFinal remains i1
	{
		iFinal += (unsigned int)v;
	}
	return iFinal + i2;
	//*/
}

// Null element w.r.t. associative function assocFunc
#ifdef __CUDACC__
__host__ __device__
#endif
inline unsigned int NullElement()
{
	// Null element for Addition
	return 0;

	// Null element for Maximum
	//return 0;

	// Null element for Minimum
	//return 0xFFFFFFFF;
}
