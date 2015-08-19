/*
* Genoil's CUDA mining kernel for Ethereum
* based on Tim Hughes' opencl kernel.
* thanks to trpuvot,djm34,sp,cbuchner for things i took from ccminer.
*/

#define SHUFFLE_CUDA_VERSION 300
#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ SHUFFLE_CUDA_VERSION
#endif
#include "ethash_cu_miner_kernel.h"
#include "ethash_cu_miner_kernel_globals.h"
#include "rotl64.cuh"
#include "keccak.cuh"
#if __CUDA_ARCH__ >= SHUFFLE_CUDA_VERSION
#include "dagger_shuffle.cuh"
#else
#include "dagger.cuh"
#endif


#define SWAP64(v) \
  ((ROTL64L(v,  8) & 0x000000FF000000FF) | \
   (ROTL64L(v, 24) & 0x0000FF000000FF00) | \
   (ROTL64H(v, 40) & 0x00FF000000FF0000) | \
   (ROTL64H(v, 56) & 0xFF000000FF000000))

__global__ void
__launch_bounds__(128, 7)
ethash_init(
	uint64_t* g_state,
	hash32_t const* g_header,
	uint64_t start_nonce
)
{
	// sha3_512(header .. nonce)
	uint64_t state[25];

	uint32_t const gid = blockIdx.x * blockDim.x + threadIdx.x;

	copy(state, g_header->uint64s, 4);
	state[4] = start_nonce + gid;
	state[5] = 0x0000000000000001;
	for (uint32_t i = 6; i < 25; i++)
	{
		state[i] = 0;
	}
	state[8] = 0x8000000000000000;
	keccak_f1600_block(state, 8);
	copy(g_state + (gid * STATE_SIZE), state, 8);
}

__global__ void
__launch_bounds__(128, 7)
ethash_dagger(
	uint64_t* g_state,
	hash128_t const* g_dag
)
{
	uint32_t const gid  = blockIdx.x * blockDim.x + threadIdx.x;

	uint64_t state[8];
	copy(state, g_state + (gid * STATE_SIZE), 8);
	uint64_t state2[8];
	copy(state2, g_state + (blockDim.x * gridDim.x * STATE_SIZE) + (gid * STATE_SIZE), 8);

	// Threads work together in this phase in groups of 8.
	uint64_t const thread_id = threadIdx.x & (THREADS_PER_HASH - 1);
#if __CUDA_ARCH__ >= SHUFFLE_CUDA_VERSION
	const int start_lane = (threadIdx.x >> 3) << 3;

	const uint32_t mix_idx = (thread_id & 3);
	uint4 mix;
	uint4 mix2;

	uint32_t shuffle[16];
	uint32_t shuffle2[16];
	uint32_t * init = (uint32_t *)state;// (g_state + (gid * STATE_SIZE));
	uint32_t * init2 = (uint32_t *)state2;
	for (int i = 0; i < THREADS_PER_HASH; i++)
	{

		// share init among threads
		for (int j = 0; j < 16; j++) {
			shuffle[j]  = __shfl(init[j], start_lane + i);
			shuffle2[j] = __shfl(init2[j], start_lane + i);
		}
		// ugly but avoids local reads/writes
		if (mix_idx == 0) {
			mix = make_uint4(shuffle[0], shuffle[1], shuffle[2], shuffle[3]);
			mix2 = make_uint4(shuffle2[0], shuffle2[1], shuffle2[2], shuffle2[3]);
		}
		else if (mix_idx == 1) {
			mix = make_uint4(shuffle[4], shuffle[5], shuffle[6], shuffle[7]);
			mix2 = make_uint4(shuffle2[0], shuffle2[1], shuffle2[2], shuffle2[3]);
		}
		else if (mix_idx == 2) {
			mix = make_uint4(shuffle[8], shuffle[9], shuffle[10], shuffle[11]);
			mix2 = make_uint4(shuffle2[0], shuffle2[1], shuffle2[2], shuffle2[3]);
		}
		else {
			mix = make_uint4(shuffle[12], shuffle[13], shuffle[14], shuffle[15]);
			mix2 = make_uint4(shuffle2[0], shuffle2[1], shuffle2[2], shuffle2[3]);
		}

		uint32_t init0  = __shfl(shuffle[0] , start_lane);
		uint32_t init02 = __shfl(shuffle2[0], start_lane);

		for (uint32_t a = 0; a < ACCESSES; a += 4)
		{
			int t = ((a >> 2) & (THREADS_PER_HASH - 1));

			for (uint32_t b = 0; b < 4; b++)
			{
				if (thread_id == t)
				{
					shuffle[0] = fnv(init0 ^ (a + b), ((uint32_t *)&mix)[b]) % d_dag_size;
					shuffle2[0] = fnv(init02 ^ (a + b), ((uint32_t *)&mix2)[b]) % d_dag_size;
				}

				shuffle[0] = __shfl(shuffle[0], start_lane + t);
				shuffle2[0] = __shfl(shuffle2[0], start_lane + t);

				mix = fnv4(mix, g_dag[shuffle[0]].uint4s[thread_id]);
				mix2 = fnv4(mix2, g_dag[shuffle2[0]].uint4s[thread_id]);
			}
		}

		uint32_t thread_mix = fnv_reduce(mix);
		uint32_t thread_mix2 = fnv_reduce(mix2);

		// update mix accross threads

		for (int j = 0; j < 8; j++) {
			shuffle[j]  = __shfl(thread_mix, start_lane + j);
			shuffle2[j] = __shfl(thread_mix2, start_lane + j);
		}

		if (i == thread_id) {
			//move mix into state:
			PACK64(g_state[gid * STATE_SIZE + 8], shuffle[0], shuffle[1]);
			PACK64(g_state[gid * STATE_SIZE + 9], shuffle[2], shuffle[3]);
			PACK64(g_state[gid * STATE_SIZE + 10], shuffle[4], shuffle[5]);
			PACK64(g_state[gid * STATE_SIZE + 11], shuffle[6], shuffle[7]);
			PACK64(g_state[blockDim.x * gridDim.x  * STATE_SIZE + gid * STATE_SIZE + 8], shuffle2[0], shuffle2[1]);
			PACK64(g_state[blockDim.x * gridDim.x  * STATE_SIZE + gid * STATE_SIZE + 9], shuffle2[2], shuffle2[3]);
			PACK64(g_state[blockDim.x * gridDim.x  * STATE_SIZE + gid * STATE_SIZE + 10], shuffle2[4], shuffle2[5]);
			PACK64(g_state[blockDim.x * gridDim.x  * STATE_SIZE + gid * STATE_SIZE + 11], shuffle2[6], shuffle2[7]);
		}
	}
#else
	extern __shared__  compute_hash_share share[];

	uint32_t const hash_id = threadIdx.x >> 3;

	hash32_t mix;
	hash64_t init;
	copy(init.uint64s, state, 8);
	for (int i = 0; i < THREADS_PER_HASH; i++)
	{
		// share init with other threads
		if (i == thread_id)
			share[hash_id].init = init;

		uint4 thread_init = share[hash_id].init.uint4s[thread_id & 3];

		uint32_t thread_mix = inner_loop(thread_init, thread_id, share[hash_id].mix.uint32s, g_dag);

		share[hash_id].mix.uint32s[thread_id] = thread_mix;


		if (i == thread_id)
			mix = share[hash_id].mix;
	}

	copy(g_state + gid * STATE_SIZE + 8, mix.uint64s, 4);
#endif
}



__global__ void
__launch_bounds__(128, 7)
ethash_final(
	uint32_t* g_output,
	uint64_t* g_state,
	uint64_t target
)
{
	// sha3_512(header .. nonce)
	uint64_t state[25];

	uint32_t const gid = blockIdx.x * blockDim.x + threadIdx.x;

	copy(state, g_state + (gid * STATE_SIZE), 12);

	// keccak_256(keccak_512(header..nonce) .. mix);
	state[12] = 0x0000000000000001;
	for (uint32_t i = 13; i < 25; i++)
	{
		state[i] = 0;
	}
	state[16] = 0x8000000000000000;
	keccak_f1600_block(state, 1);

	if (SWAP64(state[0]) < target)
	{
		atomicInc(g_output, d_max_outputs);
		g_output[g_output[0]] = gid;
	}
}


void run_ethash_search(
	uint32_t blocks,
	uint32_t threads,
	cudaStream_t stream,
	uint32_t* g_output,
	hash32_t const* g_header,
	uint64_t* g_state,
	hash128_t const* g_dag,
	uint64_t start_nonce,
	uint64_t target
)
{
	ethash_init  <<<blocks, threads, 0, stream >>>(g_state, g_header, start_nonce);
	cudaDeviceSynchronize();

#if __CUDA_ARCH__ >= SHUFFLE_CUDA_VERSION
	ethash_dagger <<<blocks/2, threads, 0, stream >> >(g_state, g_dag);
#else
	ethash_dagger <<<blocks/2, threads, (sizeof(compute_hash_share) * threads) / THREADS_PER_HASH, stream >> >(g_state, g_dag);
#endif
	cudaDeviceSynchronize();

	ethash_final <<<blocks, threads, 0, stream >>>(g_output, g_state, target);
}

cudaError set_constants(
	uint32_t * dag_size,
	uint32_t * max_outputs
	)
{
	cudaError result;
	result = cudaMemcpyToSymbol(d_dag_size, dag_size, sizeof(uint32_t));
	result = cudaMemcpyToSymbol(d_max_outputs, max_outputs, sizeof(uint32_t));
	return result;
}
