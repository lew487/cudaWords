/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */


#include "book.h"
#include "lock.h"
#include <string.h>
#include <stdio.h>
#include <iostream>

#define SIZE    (100*1024*1024)
//#define ELEMENTS    (SIZE / sizeof(char))
#define ELEMENTS 5
#define HASH_ENTRIES     1024

using namespace std;


struct Entry {
    char*    key;
    void            *value;
    Entry           *next;
};

struct Table {
    size_t  count;
    Entry   **entries;
    Entry   *pool;
};


__device__ __host__ size_t hash( char* str,
                                 size_t len ) {
      uint hash = 0, multiplier = 1;
  for(int i = len - 1; i >= 0; i--) {
    hash += str[i] * multiplier;
    int shifted = multiplier << 5;
    multiplier = shifted - multiplier;
  }
  return hash;
}

void initialize_table( Table &table, int entries,
                       int elements ) {
    table.count = entries;
    HANDLE_ERROR( cudaMalloc( (void**)&table.entries,
                              entries * sizeof(Entry*)) );
    HANDLE_ERROR( cudaMemset( table.entries, 0,
                              entries * sizeof(Entry*) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&table.pool,
                               elements * sizeof(Entry)) );
}

void copy_table_to_host( const Table &table, Table &hostTable) {
    hostTable.count = table.count;
    hostTable.entries = (Entry**)calloc( table.count,
                                         sizeof(Entry*) );
    hostTable.pool = (Entry*)malloc( ELEMENTS *
                                     sizeof( Entry ) );

    HANDLE_ERROR( cudaMemcpy( hostTable.entries, table.entries,
                              table.count * sizeof(Entry*),
                              cudaMemcpyDeviceToHost ) );
    HANDLE_ERROR( cudaMemcpy( hostTable.pool, table.pool,
                              ELEMENTS * sizeof( Entry ),
                              cudaMemcpyDeviceToHost ) );

    for (int i=0; i<table.count; i++) {
        if (hostTable.entries[i] != NULL)
            hostTable.entries[i] =
                (Entry*)((size_t)hostTable.entries[i] -
                (size_t)table.pool + (size_t)hostTable.pool);
    }
    for (int i=0; i<ELEMENTS; i++) {
        if (hostTable.pool[i].next != NULL)
            hostTable.pool[i].next =
                (Entry*)((size_t)hostTable.pool[i].next -
                (size_t)table.pool + (size_t)hostTable.pool);
    }
}

void free_table( Table &table ) {
    HANDLE_ERROR( cudaFree( table.pool ) );
    HANDLE_ERROR( cudaFree( table.entries ) );
}

__global__ void add_to_table( char **keys, void **values, 
                              Table table, Lock *lock ) {
				//printf("key is %s\n", &(*keys)[0]);				  
								  /*
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    while (tid < ELEMENTS) {
        //string key = keys[tid];
		printf("key is %s\n", &(*keys)[0]);
        size_t hashValue = hash( keys[tid], table.count );
        for (int i=0; i<32; i++) {
            if ((tid % 32) == i) {
                Entry *location = &(table.pool[tid]);
                location->key = keys[tid];
                location->value = values[tid];
                lock[hashValue].lock();
                location->next = table.entries[hashValue];
                table.entries[hashValue] = location;
                lock[hashValue].unlock();
            }
			
        }
        tid += stride;
    }*/
}

void verify_table( const Table &dev_table ) {
    Table   table;
    copy_table_to_host( dev_table, table );

    int count = 0;
    for (size_t i=0; i<table.count; i++) {
        Entry   *current = table.entries[i];
        while (current != NULL) {
            ++count;
            if (hash( current->key, table.count ) != i)
                printf( "%d hashed to %ld, but was located at %ld\n",
                        current->key,
                        hash(current->key, table.count), i );
            current = current->next;
        }
    }
    if (count != ELEMENTS)
        printf( "%d elements found in hash table.  Should be %ld\n",
                count, ELEMENTS );
    else
        printf( "All %d elements found in hash table.\n", count );

    free( table.pool );
    free( table.entries );
}


int main( void ) {
	/*
    unsigned int *buffer =
                     (unsigned int*)big_random_block( SIZE );*/
					 int num_arrays=5;
	char **buffer = (char *[]){ "New Game", "Continue Game", "Exit" };

    char **dev_keys;
	
	//buffer = (char**)malloc(num_arrays * sizeof(char *));
	
	/*
	for(int i=0;i<num_arrays;i++)
	{
		(*buffer)[i]="aa";
	}*/
    void  **dev_values;
	
	cudaMalloc((void**)&dev_keys, 1024 * sizeof(char *[num_arrays]));
	
	for (int i = 0; i < num_arrays; i++)
   cudaMalloc((void**)&buffer[i], sizeof(char*)*1024);

	cudaMemcpy(dev_keys, buffer, 1024*sizeof(char *[num_arrays]), cudaMemcpyHostToDevice);
	
	/*
    HANDLE_ERROR( cudaMalloc( (void**)&dev_keys, ELEMENTS ) );
	
	for (int i = 0; i < ELEMENTS; i++)
   cudaMalloc((void**)&dev_keys[i], 1024);
	
    HANDLE_ERROR( cudaMalloc( (void**)&dev_values, SIZE ) );
    HANDLE_ERROR( cudaMemcpy( dev_keys, buffer, ELEMENTS*1024,
                              cudaMemcpyHostToDevice ) );
*/
    // copy the values to dev_values here
    // filled in by user of this code example
	
	/*
	for(int i=0;i<5;i++)
	{
		cudaMalloc( (void**)&dev_values, SIZE ) 
	}*/
	

    Table table;
    initialize_table( table, HASH_ENTRIES, ELEMENTS );

    Lock    lock[HASH_ENTRIES];
    Lock    *dev_lock;
    HANDLE_ERROR( cudaMalloc( (void**)&dev_lock,
                              HASH_ENTRIES * sizeof( Lock ) ) );
    HANDLE_ERROR( cudaMemcpy( dev_lock, lock,
                              HASH_ENTRIES * sizeof( Lock ),
                              cudaMemcpyHostToDevice ) );

    cudaEvent_t     start, stop;
    HANDLE_ERROR( cudaEventCreate( &start ) );
    HANDLE_ERROR( cudaEventCreate( &stop ) );
    HANDLE_ERROR( cudaEventRecord( start, 0 ) );

	
    add_to_table<<<60,256>>>( dev_keys, dev_values,
                              table, dev_lock );

					
    HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
    HANDLE_ERROR( cudaEventSynchronize( stop ) );
    float   elapsedTime;
    HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime,
                                        start, stop ) );
    printf( "Time to hash:  %3.1f ms\n", elapsedTime );

    //verify_table( table );

    HANDLE_ERROR( cudaEventDestroy( start ) );
    HANDLE_ERROR( cudaEventDestroy( stop ) );
    free_table( table );
    HANDLE_ERROR( cudaFree( dev_lock ) );
    HANDLE_ERROR( cudaFree( dev_keys ) );
    HANDLE_ERROR( cudaFree( dev_values ) );
   // free( buffer );
    return 0;
}

