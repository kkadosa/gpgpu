// map operator with f(x) = x*x
__kernel void square(__global float* inputData,
             		     __global float* outputData)
{
  int id = get_global_id(0);
  outputData[id] = inputData[id] * inputData[id];
}

// TODO
//
// histogram[data[id]] := histogram[data[id]] + 1
//
// SYNCHRONIZATION!
__kernel
void histogram_global(__global int* data, __global int* histogram)
{
    int id = get_global_id(0);
    //histogram[data[id]] += 1; // causes WAW errors, all results are 1
    atomic_add(histogram + data[id], 1); //correct, but bottleneck
}

// TODO
//
// ID  := get_global_id(0)
// LID := get_local_id(0)
//
// IF LID < histogramSize DO:
//  lhistogram[LID] := 0
// BARRIER
//
// Add data to local histogram
//
// BARRIER
// 
// IF LID < histogramSize DO:
//  histogram[LID] = lhistogram[LID]
__kernel
void histogram_local(__global int* data, __global int* histogram, __local int* lhistogram, const int histogramSize)
{
    int id = get_global_id(0);
    int lid = get_local_id(0);

    if (lid < histogramSize) {
        lhistogram[lid] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    atomic_add(lhistogram + data[id], 1);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < histogramSize) {
        atomic_add(histogram + lid, lhistogram[lid]);
    }
}

__kernel
void reduce_global(__global float* data)
{
    int id = get_global_id(0);
    int s = get_global_size(0) / 2;
    for (; s > 0; s >>= 1) {
        if (id < s) {
            data[id] = data[id] + data[id + s];
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}

__kernel
void exscan_global(__global int* data, __global int* prefSum)
{
    int id = get_global_id(0);
    if (id > 0) {
        prefSum[id] = data[id - 1];
    } else {
        prefSum[0] = 0;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    int s = 1;
    for (; s < get_global_size(0); s *= 2) {
        int tmp = prefSum[id];
        barrier(CLK_GLOBAL_MEM_FENCE);
        if ((id + s) < get_global_size(0)) {
            prefSum[id + s] += tmp;
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}

// TODO
// ID := get_global_id(0)
// IF data[id] < 50 THEN
//   predicate = 1
// ELSE
//   predicate = 0
__kernel
void compact_predicate(__global int* data, __global int* pred)
{
        
}

// TODO
//
// exclusive scan pred to prefSum
__kernel
void compact_exscan(__global int* pred, __global int* prefSum)
{

}

// TODO
// 
// ID := get_global_id(0)
// VALUE := data[ID]
// BARRIER
// IF pred[ID] == 1 THEN
//  data[prefSum[ID]] = VALUE
__kernel
void compact_compact(__global int* data, __global int* pred, __global int* prefSum)
{
        
}