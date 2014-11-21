#include<stdio.h>
#include<stdlib.h>
#include "nuc2num.h"
#include "hashfuc.h"
#include "statfreq.h"

int main(){
	int i;
	char *ref = "ACGTACGT";
	const int LEN=4,MER=2;
	int *fre;
	size_t SIZE = LEN*sizeof(int);
	fre = (int *) malloc(SIZE);
	for (i=0;i<LEN;i++){
		*(fre+i)=0;
	}
	char *ref = "ACGTACGT";
	fre = statfreq(ref,MER);
	for (i=0; i<LEN; i++){
		printf("%d\t",*(fre+i));
	}

//	free(fre);
	return 0;
}
