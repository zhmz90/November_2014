#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include "nuc2num.h"
#include "hashfuc.h"
#include "statfreq.h"

int hashfuc(char *string,int mer);
int nuc2num(char s);
int * statfreq(char *ref,int mer);



int main(){
	
	char *ref = "ACGTACGT";
	const int MER=2;
	
	int *fre;
	int i,len;
	size_t SIZE = len*sizeof(int);
	
	len = (int)pow(len,MER);

	fre = (int *)malloc(SIZE);
	for (i=0;i<len;i++){
		*(fre+i)=0;
	}

	statfreq(ref,MER,fre);
	for (i=0; i<len; i++){
		printf("%d\t",*(fre+i));
	}

//	free(fre);
	return 0;
}

int hashfuc(char *string,int mer){
/* input is a substring of a reference or read, len is the length of the string.
   output is  a integer to be used as an address of the index array following.
   example: "AAA" --> 0  "ATA" --> 4
            "AAT" --> 1  "ATT" --> 5
            "AAC" --> 2  "ATC" --> 6
            "AAG" --> 3  "ATG" --> 7
  The real string has a length of 16.
*/
        int sum=0,loc=mer,i;
        char tmp;
        for (i=0;(tmp=*(string+i)) != '\0'; i++){
                sum+= (int)pow(4,(double)(loc-1))*nuc2num(tmp);
                loc-=1;
        }

        return sum;
}

int nuc2num(char s){
/* input is a char type of nucletide such as "A","a","T","t","C","c","G","g"
   output is a number of the nucletide. 
   example: "A" --> 0  "a" --> 0
            "T" --> 1  "t" --> 1
            "C" --> 2  "c" --> 2
            "G" --> 3  "g" --> 3 
*/
        switch(s){
        case 'A': return 0;
        case 'a': return 0;
        case 'T': return 1;
        case 't': return 1;
        case 'C': return 2;
        case 'c': return 2;
        case 'G': return 3;
        case 'g': return 3;
        default:
                printf("Error happens in nuc2num(). Please check your input!\n");
                return -1;

        }


}

int statfreq(char *ref,int mer, int *fre){
/* (&"ATCGATCG", 3) --> &[12,1000,14,...] 
   input is: reference and length of mer.	
   output is: the frequents of various mers.
*/
	int *fre;
	int i;
	const int LEN_idx = mer;
	size_t SIZE_idx = LEN_idx * sizeof(int);
	idx = (int *) malloc(SIZE_idx);
	//instantize to the array to be  0s.
	for (i=0; i<pow(4,mer);i++)
		*(fre+i) = 0;
	int loc,num,sub;
	for (loc=0; loc<(strlen(ref)-mer+1); loc++){
		*(fre+hashfuc((ref+loc),mer)) += 1;
	}	
	
	return 0;
}

