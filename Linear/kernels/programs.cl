// TODO: Simple matrix-vector multiplication, every thread computes a complete dot product
//
// i := get_global_id(0)
//
// IF ID < n THEN:
//   yi := b[i]
//   LOOP j := 0 .. m DO:
//     yi += A[j + i * m] * x[j]
//   END LOOP
//   y[i] := yi
// END IF
__kernel
void simpleMV(const int width, const int height, __global float* ping, __global float* mat, __global float* pong, __global float* offset){
	int id = get_global_id(0);
	if (id < height) {
		float yi = offset[id];
		for (int j = 0; j < width; ++j) {
			yi += mat[id * width + j] * ping[j];
		}
		pong[id] = yi;
	}
}

// TODO: Matrix-vector multiplication with parallelization of the dot product
// Assumptions: M = 2^k, M <= maximum workgroup size
//
// i = get_group_id(0)
// j = get_local_id(0)
//
// Q[j] := A[i * M + j] * x[j]
// BARRIER
//
// Sum scan on Q (reduction)
//
// IF j = 0 THEN:
//   y[i] = Q[0] + b[i]
//
__kernel
void reduceMV(const int width, const int height, __global float* ping, __global float* mat, __global float* pong, __global float* offset, __local float* loc) {
	int gid = get_group_id(0);
	int lid = get_local_id(0);

	loc[lid] = mat[gid * width + lid] * ping[lid];
	barrier(CLK_LOCAL_MEM_FENCE);
	for (int i = 1; i < width; i *= 2) {
		if ((lid + i) < width) {
			loc[lid] += loc[lid + i];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (lid == 0) {
		pong[gid] = loc[0] + offset[gid];
	}
}

// TODO: General solution for matrix-vector multiplication, every thread processes a chunk of the dot product and visits multiple rows of the result
//
// t := get_local_id(0) / Z
// z := get_local_id(0) % Z
//
// FOR i := t ; i < n ; i := i + T :
//    Compute Q[t * Z + z] as shown in the lecture
//    Sum scan on Q (reduction)
//    IF z = 0 THEN:
//        y[i] = Q[t * Z + 0] + b[i]
//
// END FOR
__kernel
void largeMV(const int width, const int height, __global float* ping, __global float* mat, __global float* pong, __global float* offset, const int T, const int Z, __local float* loc){
	int t = get_local_id(0) / Z;
	int z = get_local_id(0) % Z;

	for (int i = t; i < width; i += T) {
		loc[t * Z + z] = 0;
		for (int j = z; j < height; j += Z) {
			loc[t * Z + z] += mat[j + i * width] * ping[j];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		for (int k = 1; k < Z; k *= 2) {
			if ((z + k) < Z) {
				loc[t*Z+z] += loc[t * Z + z + k];
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}
		if (z == 0) {
			pong[i] = loc[t * z] + offset[i];
		}
	}
}

// TODO: Gaussian elimination as shown in the lecture
// (execute the 2nd loop of the sequential implemential in parallel)
__kernel void gaussian(const int width, const int height, __global float* A){
	int id = get_global_id(0);
	if (id < height) {
		for (int k = 0; k < height; ++k) {
			float l = A[width * id + k] / (float) A[width * k + k];
			for (int j = k; j < width; ++j) {
				if (k != id) {
					A[width * id + j] = A[width * id + j] - l * A[width * k + j];
				}
			}
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
		float end = A[width * id + id];
		A[width * id + id] = 1;
		for (int i = height; i < width; ++i) {
			A[width * id + i] = A[width * id + i] / end;
		}
	}
}

