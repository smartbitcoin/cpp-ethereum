#ifndef _ETHASH_CU_MINER_KERNEL_GLOBALS_H_
#define _ETHASH_CU_MINER_KERNEL_GLOBALS_H_

#ifdef __INTELLISENSE__
#include <device_launch_parameters.h>
#endif

__constant__ uint32_t d_dag_size;
__constant__ uint32_t d_max_outputs;

#endif