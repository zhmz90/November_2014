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

