#include <iostream>
#include <stdio.h>
#include <omp.h>

class Geek{
    public:
        void myFunction(list output, ){
            std::cout << "Hello Geek!!!" << std::endl;

            #pragma omp parallel for
            for (int i = 0; i < partitions_.size(); i++) {
                Partition *partition = partitions_[i];
                
            }
        }
        
};
int main()
{
    // Creating an object
    Geek t; 
  
    // Calling function
    t.myFunction();  
   
    return 0;
}

extern "C" {
    Geek* Geek_new(){ return new Geek(); }
    void Geek_myFunction(Geek* geek){ geek -> myFunction(); }
}