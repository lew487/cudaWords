#include <stdio.h>
#include <cuda.h>
#include <time.h>

#include <string.h>
#include <assert.h>

#include <map>
#include <iostream>
#include <cassert>
#include <sstream>

#include <vector>
#include <algorithm>    // std::transform


#define maxWordSize 1024

#define max_number_of_threads 512


using namespace std;

typedef struct gpuStringArray {
    unsigned int * pos; 
    unsigned int * length;  // could be a smaller type if strings are short
    char * data; // 32 bit data type will improve memory throughput, could be 8 bit
	unsigned int size;
} gpuStringArray;

__device__ int cmp4(const char & c1, const char & c2)
{
	return c1==c2;

}

__device__ int strncmp4(const char * s1, const char * s2, const unsigned int nwords)
{
    for(unsigned int i=0; i<nwords; i++) {
        int result = cmp4(s1[i], s2[i]);
        if (result == 0) return result;
    }

    return 1;
}

__global__ void tkernel(struct gpuStringArray *a, gpuStringArray *b, int * result, int *threadsNumber)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

	int wordsCount=b->size;
	int iterationsPerThread=wordsCount/(*threadsNumber)+1;
	
	for(int i=0; i<iterationsPerThread; i++)
	{
		if(idx>wordsCount) return;
		for(int j=0; j< a->size; j++)
		{
			
			char * s1 = a->data + a->pos[j];
			char * s2 = b->data + b->pos[idx];
			unsigned int slen = max(a->length[j], b->length[idx]);
			if(a->length[j]!= b->length[idx]) result[idx]=0;
			else result[idx]=strncmp4(s1, s2, slen);
			if(result[idx]==1) break;
		}

		
		idx+=*threadsNumber;
	}
}


void makeStringWithoutBadSigns(char *sign)
{
	int length=strlen(sign);
	
	
	if(sign[length-1]>0 && sign[length-1]>32 && sign[length-1]<65)
	{
		sign[length-1]=0;
		makeStringWithoutBadSigns(sign);
	}
}

dim3 getBlockSettings(int threads)
{
	dim3 block_size;

	if(threads>max_number_of_threads)
		block_size.x = max_number_of_threads;
	else
		block_size.x = threads;
	block_size.y = 1;
	
	
	return block_size;
}

dim3 getGridSettings(int threadsNumber)
{
	int gridNumbers = threadsNumber / max_number_of_threads;
	if (gridNumbers * max_number_of_threads < threadsNumber) ++gridNumbers;
	dim3 grid_size;
	grid_size.x=gridNumbers;
	grid_size.y=1;
	
	
	return grid_size;
}



void read_words (FILE *f, map<string, int> &m) {
    char x[maxWordSize];
	vector < string > dane;
	
    while (fscanf(f, " %1023s", x) == 1) {
		//cout << x << endl;
		makeStringWithoutBadSigns(x);
		m[x]++;
		string s = std::string(x);
		dane.push_back( s );
    }


	
}

gpuStringArray* allocateGpuStringArray(vector<unsigned int> positions, vector<unsigned int> lenghts, string str)
{
	unsigned int *cudaPos;
	char *cudaData;
	unsigned int *cudaLength;
	
	unsigned int* inputPositions = &positions[0];
	size_t positionsSize=sizeof(unsigned int)*(positions.size()+1);
	cudaMalloc((void **) &cudaPos, positionsSize);  
	cudaMemcpy(cudaPos, inputPositions, positionsSize, cudaMemcpyHostToDevice);
	
	char dataArray[str.size()+1];//as 1 char space for null is also required
	strcpy(dataArray, str.c_str());
	size_t dataSize=sizeof(char)*(str.size()+1);
	cudaMalloc((void **) &cudaData, dataSize); 
	cudaMemcpy(cudaData, dataArray, dataSize, cudaMemcpyHostToDevice);
	
	unsigned int* inputLengths= &lenghts[0];
	size_t lengthSize=sizeof(unsigned int)*(lenghts.size()+1);
	cudaMalloc((void **) &cudaLength, lengthSize);	
	cudaMemcpy(cudaLength, inputLengths, lengthSize, cudaMemcpyHostToDevice);
	
	gpuStringArray *stringArray;
	stringArray=(gpuStringArray*) malloc(sizeof(gpuStringArray));
	stringArray->pos=cudaPos;
	stringArray->length=cudaLength;
	stringArray->data=cudaData;
	stringArray->size=lenghts.size();
	
	gpuStringArray *cudaStringArray;
	size_t cudaArray=sizeof(gpuStringArray);
	cudaMalloc((void **) &cudaStringArray, cudaArray);  
	cudaMemcpy(cudaStringArray, stringArray, cudaArray, cudaMemcpyHostToDevice); // Skopiowanie danych do GPU
	
	return cudaStringArray;
}

template <typename T1, typename T2>
struct less_second {
    typedef pair<T1, T2> type;
    bool operator ()(type const& a, type const& b) const {
        return a.second < b.second;
    }
};


vector<pair<string, int> > sortMapByKeys(map<string, int> m)
{
	//copy to vector to sort by values
	vector<pair<string, int> > mapcopy(m.begin(), m.end());
	sort(mapcopy.begin(), mapcopy.end(), less_second<string, int>());
	
	return mapcopy;
}
 
int main(int argc, char** argv)
{
	char* fileInput;
	char* fileInput2;
	int threadsNumber;
	bool isDataShouldShow=false;

	
	fileInput = argv[2];
	fileInput2 = argv[3];
	threadsNumber = atoi(argv[5]);
	isDataShouldShow = atoi(argv[7]);
	
	
	//time
	float time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0); //start time
	
	map<string, int> m;
	FILE *inputFile;

	inputFile = fopen(fileInput, "r");
	read_words(inputFile, m);
	
	//1
	stringstream ss;
	vector<unsigned int> positions;
	vector<unsigned int> lenghts;
	unsigned int previousPos=0;
	for (std::map<string,int>::iterator it=m.begin(); it!=m.end(); ++it)
	{
		ss << it->first;
		positions.push_back(previousPos);
		lenghts.push_back(it->first.length());
		
		previousPos+=it->first.length();
	}
	
	string str = ss.str();
	
	
	
	gpuStringArray *inputCuda=allocateGpuStringArray(positions, lenghts, str);
	
	
	
	//2
	map<string, int> m2;
	FILE *outputFile;
	outputFile = fopen(fileInput2, "r");
	read_words(outputFile, m2);
	
	
	stringstream ss2;
	vector<unsigned int> positions2;
	vector<unsigned int> lenghts2;
	unsigned int previousPos2=0;
	for (std::map<string,int>::iterator it=m2.begin(); it!=m2.end(); ++it)
	{
		//std::cout << it->first << " => " << it->second << '\n';
		ss2 << it->first;
		positions2.push_back(previousPos2);
		lenghts2.push_back(it->first.length());
		
		previousPos2+=it->first.length();
	}
	
	string str2 = ss2.str();
	
	
	gpuStringArray *tempCuda=allocateGpuStringArray(positions2, lenghts2, str2);
	
	
	
	size_t resultSize=sizeof(int)*m2.size();
	int *result=(int*) malloc(resultSize);
	int *cudaResult;
	cudaMalloc((void **) &cudaResult, resultSize); 
	
	size_t sizeThreadsNumber=sizeof(int);
	int *cudaThreadsNumber;
	cudaMalloc((void **) &cudaThreadsNumber, sizeThreadsNumber); 
	cudaMemcpy(cudaThreadsNumber, &threadsNumber, sizeThreadsNumber, cudaMemcpyHostToDevice);
	
	
	dim3 block_size=getBlockSettings(threadsNumber);
	dim3 grid_size=getGridSettings(threadsNumber);
	tkernel<<< grid_size, block_size>>> (inputCuda, tempCuda, cudaResult, cudaThreadsNumber);
	
	cudaMemcpy(result, cudaResult, resultSize, cudaMemcpyDeviceToHost);
	
	int finalResult=0;
	
	
	
	for(int i=0; i<m2.size() ;i++)
	{
		finalResult+=result[i];
		//cout << i << " " << result[i] << endl;
	}
	

	if(isDataShouldShow)
	{
		vector<pair<string, int> > mapcopy=sortMapByKeys(m);
		
		cout << "\n 10 most used words in input of neural network" << endl;
		for(int i = mapcopy.size()-10; i < mapcopy.size(); i++)
		{
			cout << mapcopy[i].first << " -> " << mapcopy[i].second << endl;
		}
		
		cout << "\n 10 most used words in output of neural network" << endl;
		vector<pair<string, int> > mapcopy2=sortMapByKeys(m2);
		
		for(int i = mapcopy2.size()-10; i < mapcopy2.size(); i++)
		{
			cout << mapcopy2[i].first << " -> " << mapcopy2[i].second << endl;
		}
	}
	
	
	cout << "final result " << (((float)finalResult/m2.size())*100) << " %" << endl;
	
	cudaEventRecord(stop, 0); // timer stop
	 cudaEventSynchronize(stop);
	 cudaEventElapsedTime(&time, start, stop);
	 printf ("Time for the kernel: %f ms\n", time);
	
		
	return 0;

}