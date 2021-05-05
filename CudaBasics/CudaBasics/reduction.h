#pragma once

const size_t threadsPerBlock = 512;

void reduceGPU(unsigned int* d_idata, unsigned int* d_odata, size_t dataSize, int method, bool synchronizeDevice = true);