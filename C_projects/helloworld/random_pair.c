#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int* read_file(char* filename, int length){
    FILE *myFile;
    myFile = fopen(filename, "r");

    //read file into array
    char* Array_= malloc(sizeof(int) * length);
    int* nid= malloc(sizeof(int) * length);
    int i=0;

    if (myFile == NULL){
        printf("Error Reading File\n");
        exit (0);
    }

    for (i = 0; i < length; i++){
        fscanf(myFile, "%d,", Array_[i] );
        nid[i] = atoi(Array_[i]);
    }
    printf("length is: %d\n\n", sizeof(Array_));
    // for (i = 0; i < num_of_src; i++){
    //     printf("Number is: %d\n\n", src_Array[i]);
    // }

    fclose(myFile);

    return nid;

}
    

void printRandomNumber( int low, int high )
{
    // Seed the random number generator.
    srand( time(0) );
    // Generate 2 random numbers between low and high (inlcusive).
    int a = (rand() % ( high-low )) + (low+1);
    int b = ( rand() % ( high-low ) ) + ( low + 1 );

    // Find a new low and new high.
    if ( a <= b )
    {
        low = a;
        high = b;
    } else {
        high = a;
        low = b;
    }

    printf("The new low is: %d\nThe new high is: %d\n", low, high);

    // Print 5 numbers inside the new range.
    for ( int i = 0; i < 5; i++ )
    {
        printf( "%d\n", rand() % ( high-low ) + ( low + 1 ) );
    } 
    // return    
}
void reorder_function(int* src, int* dst){
    printf('testing -------...............');
    return(0);
}

int pt_rand(int nbits) {
    int mask;
    if (0 < nbits && nbits < sizeof(int)*8) {
        mask = ~(~((unsigned int) 0) << nbits); 
    }
    else {
        mask = ~((unsigned int) 0);
    }
    return rand() & mask;
}

int *gen_rand_int_array_nodups(int length, int nbits) {
    int * a = malloc(sizeof(int)*length);
    for (int i = 0; i < length; i++) {
        int duplicate;
        do
        {
            duplicate = 0;
            a[i] = pt_rand(nbits);
            for (int j = 0; j < i; j++) {
                if (a[j] == a[i])
                {
                    duplicate = 1;
                    break;
                }
            }
        } while (duplicate);
    }
    shuffle_int_array(a, length);
    return a;
}

int main()
{
    // printRandomNumber(0, 100);
    // int* src_array = read_file('2374476_src.txt', 2374476);
    // int* dst_array = read_file('97922_dst.txt', 97922);
    // reorder_function(src_array, dst_array);

    // free(src_array);
    // free(dst_array);
    int src[] = { };
    int dst[] ={};
    return (0);
}