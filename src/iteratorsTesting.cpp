#include <iostream>
#define Length 9
#define Breadth 9
#define Depth 9


int flatten (int x, int y, int z){
    int value = ((z * (Length * Breadth)) + (y * Length) + (x));
    return value;
}

int main(){

    /*for (int i = 0; i < depth; ++i){
        for(int j = 0; j < length; ++j){
            for (int k = 0; k < breadth; ++k){
                std::cout << "i = " << i << 
                             "; j = " << j << 
                             "; k = " << k << 
                             " maps to " << (i*(length * breadth)) + (j*(breadth)) + k << std::endl; 
            }
        }
    }*/
    for (int z = 0; z < Depth; ++z){
        for(int y = 0; y < Breadth; ++y){
            for (int x = 0; x < Length; ++x){
                    int mappedValue = flatten(x, y, z);
                    std::cout << "(" << x << "," << y << "," << z << ") = " << mappedValue << std::endl;
            }
        }
    }
    return 0;
}