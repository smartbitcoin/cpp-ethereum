#include "ethash_cu_miner_kernel_globals.h"
#include "ethash_cu_miner_kernel.h"
#include "cuda_helper.h"
#include "keccak.cuh"
#include "dagger.cuh"

#define PACK64(result, lo, hi) asm("mov.b64 %0, {%1,%2};//pack64"  : "=l"(result) : "r"(lo), "r"(hi));
#define UNPACK64(lo, hi, input) asm("mov.b64 {%0, %1}, %2;//unpack64" : "=r"(lo),"=r"(hi) : "l"(input));

__device__ uint64_t compute_hash_shuffle(
	hash32_t const* g_header,
	hash128_t const* g_dag,
	uint64_t nonce
	)
{
	// sha3_512(header .. nonce)
	//uint64_t state[25];
	uint2 state[25];


	//copy(state, g_header->uint64s, 4);
	state[0] = vectorize(g_header->uint64s[0]);
	state[1] = vectorize(g_header->uint64s[1]);
	state[2] = vectorize(g_header->uint64s[2]);
	state[3] = vectorize(g_header->uint64s[3]);
	state[4] = vectorize(nonce);
	state[5] = vectorize(0x0000000000000001ULL);
	for (uint32_t i = 6; i < 25; i++)
	{
		state[i] = make_uint2(0,0);
	}
	state[8] = vectorize(0x8000000000000000ULL);
	keccak_f1600_init(state);

	// Threads work together in this phase in groups of 8.
	const int thread_id = threadIdx.x & (THREADS_PER_HASH - 1);
	const int start_lane = threadIdx.x & ~(THREADS_PER_HASH - 1);
	const int mix_idx = (thread_id & 3);

	uint4 mix;

	uint2 shuffle[8];
	uint32_t * init = (uint32_t *)state;

	
	/*
	uint32_t init[16];
	UNPACK64(init[0], init[1], state[0]);
	UNPACK64(init[2], init[3], state[1]);
	UNPACK64(init[4], init[5], state[2]);
	UNPACK64(init[6], init[7], state[3]);
	UNPACK64(init[8], init[9], state[4]);
	UNPACK64(init[10], init[11], state[5]);
	UNPACK64(init[12], init[13], state[6]);
	UNPACK64(init[14], init[15], state[7]);
	*/
	for (int i = 0; i < THREADS_PER_HASH; i++)
	{

		// share init among threads
		for (int j = 0; j < 8; j++) {
			shuffle[j].x = __shfl(state[j].x, start_lane + i);
			shuffle[j].y = __shfl(state[j].y, start_lane + i);
		}

		// ugly but avoids local reads/writes
		if (mix_idx == 0) {
			mix = vectorize2(shuffle[0], shuffle[1]);
			//mix = make_uint4(shuffle[0], shuffle[1], shuffle[2], shuffle[3]);
		}
		else if (mix_idx == 1) {
			mix = vectorize2(shuffle[2], shuffle[3]);
			//mix = make_uint4(shuffle[4], shuffle[5], shuffle[6], shuffle[7]);
		}
		else if (mix_idx == 2) {
			mix = vectorize2(shuffle[4], shuffle[5]);
			//mix = make_uint4(shuffle[8], shuffle[9], shuffle[10], shuffle[11]);
		}
		else {
			mix = vectorize2(shuffle[6], shuffle[7]);
			//mix = make_uint4(shuffle[12], shuffle[13], shuffle[14], shuffle[15]);
		}

		uint32_t init0 = __shfl(shuffle[0].x, start_lane);

		for (uint32_t a = 0; a < ACCESSES; a += 4)
		{
			int t = ((a >> 2) & (THREADS_PER_HASH - 1));

			for (uint32_t b = 0; b < 4; b++)
			{
				if (thread_id == t)
				{
					shuffle[0].x = fnv(init0 ^ (a + b), ((uint32_t *)&mix)[b]) % d_dag_size;
				}
				shuffle[0].x = __shfl(shuffle[0].x, start_lane + t);

				mix = fnv4(mix, g_dag[shuffle[0].x].uint4s[thread_id]);
			}
		}

		uint32_t thread_mix = fnv_reduce(mix);

		// update mix accross threads

		//for (int j = 0; j < 4; j++) {
			
			shuffle[0].x = __shfl(thread_mix, start_lane + 0);
			shuffle[0].y = __shfl(thread_mix, start_lane + 1);
			shuffle[1].x = __shfl(thread_mix, start_lane + 2);
			shuffle[1].y = __shfl(thread_mix, start_lane + 3);
			shuffle[2].x = __shfl(thread_mix, start_lane + 4);
			shuffle[2].y = __shfl(thread_mix, start_lane + 5);
			shuffle[3].x = __shfl(thread_mix, start_lane + 6);
			shuffle[3].y = __shfl(thread_mix, start_lane + 7);
			//}

		if (i == thread_id) {

			//move mix into state:
			//PACK64(state[8], shuffle[0], shuffle[1]);
			//PACK64(state[9], shuffle[2], shuffle[3]);
			//PACK64(state[10], shuffle[4], shuffle[5]);
			//PACK64(state[11], shuffle[6], shuffle[7]);
			//state[8] = make_uint2(shuffle[0], shuffle[1]);
			//state[9] = make_uint2(shuffle[2], shuffle[3]);
			//state[10] = make_uint2(shuffle[4], shuffle[5]);
			//state[11] = make_uint2(shuffle[6], shuffle[7]);
			state[8] = shuffle[0];
			state[9] = shuffle[1];
			state[10] = shuffle[2];
			state[11] = shuffle[3];
		}

	}

	// keccak_256(keccak_512(header..nonce) .. mix);
	state[12] = vectorize(0x0000000000000001ULL);
	for (uint32_t i = 13; i < 25; i++)
	{
		state[i] = vectorize(0ULL);
	}
	state[16] = vectorize(0x8000000000000000);
	keccak_f1600_final(state);

	return state[0].x;
}