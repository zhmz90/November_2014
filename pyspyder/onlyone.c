#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
//#include "nuc2num.h"
//#include "hashfuc.h"
//#include "statfreq.h"

int hashfuc(char *string,int mer);
int nuc2num(char s);

int main(){
	//get the reference from the reference file.
//	while (){

//	}
	
	//reference example
	char *ref;
	const int MER=16;
	size_t SIZE_ref = (int)*(pow(2,31))
	ref = (char *)malloc(SIZE_ref);
	
	//frequents container
	int *fre;
	int i,len_fre=(int)pow(4,MER);
	size_t SIZE = len_fre*sizeof(int);
	fre = (int *)malloc(SIZE);
	//frequents initantization
	for (i=0;i<len_fre;i++){
		*(fre+i)=0;
	}

	//statfreq(ref,MER,fre);
	for (i=0;i<(strlen(ref)-MER+1);i++){
		*(fre+hashfuc((ref+i),MER))+=1;
	}


 	//print the frequents result	
	for (i=0; i<len_fre; i++){
		printf("%d\t",*(fre+i));
	}

	
	//free the memory

	free(fre);	

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
        for (i=0; i< mer; i++){
		tmp=*(string+i);
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

