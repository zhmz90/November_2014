#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int main(){
	
	FILE *fp;
	//const int LEN = 4;
	char *buff,*b;
//	size_t SIZE = ((int)pow(4,16))*sizeof(char);
	size_t SIZE = 100000*sizeof(char);
	buff = (char *)malloc(SIZE);
	b=buff;	
//	fp = fopen("/home/h6/alignment/h100.fa","r");
	fp = fopen("/home/h6/alignment/h100.fa","r");
	int lenRead = 1;
	while (lenRead >= 0){	
		lenRead = fgets(fp, "%s", buff);
	//	printf("%d\n",lenRead);
		buff += 1;
	}
	printf("2:%s\n",b);
//	fseek();
	free(buff);

	return 0;

}
