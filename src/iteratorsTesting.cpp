#include <iostream>

int main(){

    int length = 5;
    int breadth = 6;
    int depth = 4;

    for (int i = 0; i < depth; ++i){
        for(int j = 0; j < length; ++j){
            for (int k = 0; k < breadth; ++k){
                std::cout << "i = " << i << 
                             "; j = " << j << 
                             "; k = " << k << 
                             " maps to " << (i*(length * breadth)) + (j*(breadth)) + k << std::endl; 
            }
        }
    }

    return 0;
}