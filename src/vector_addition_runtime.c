#include<stdio.h>
#define N 10000000

void vector_addn(float *out, float *a, float *b, int n) {
    for(int i = 0; i <= N; ++i){
        out[i] = a[i] + b[i];
    }
}

int main(){
    float *a, *b, *out;

    a = (float*)malloc(sizeof(float)* N);
    b = (float*)malloc(sizeof(float)* N);
    out = (float*)malloc(sizeof(float)* N);

    for (int i=0; i <= N; ++i){
        a[i] = 1.0f;
        b[i] = 1.0f;
    }
    vector_addn(out, a, b, N);
}