#include<math.h>

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

