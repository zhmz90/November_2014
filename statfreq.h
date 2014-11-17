#include<string.h>
#include<math.h>

int * statfreq(char *ref,int mer){
/* (&"ATCGATCG", 3) --> &[12,1000,14,...] 
   input is: reference and length of mer.	
   output is: the frequents of various mers.
*/
	int *idx;
	int i;
	const int LEN_idx = mer;
	size_t SIZE_idx = LEN_idx * sizeof(int);
	idx = (int *) malloc(SIZE_idx);
	//instantize to the array to be  0s.
	for (i=0; i<pow(4,mer);i++)
		*(idx+i) = 0;
	int loc,num,sub;
	for (loc=0; loc<(strlen(ref)-mer+1); loc++){
		*(idx+hashfuc((ref+loc),mer)) += 1;
	}	
	
	return idx;
}
