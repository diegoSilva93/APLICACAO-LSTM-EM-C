#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//função de ativação sigmóid
float sigmoid(float x){
    return 1.0 / (1.0 + exp(-x));
}

//função de ativação tangente hiperbólica
float tanh(float x){
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

// Célula da rede LSTM
struct LSTMCell{
    float *pesos_input; // pesos entrada célula
    float *pesos_anterior; // pesos saída anterior da célula
    float *biases; // bias da célula
    float *estado_celula; // estado da célula
    float *saida_celula; // saída da célula
}

// Inicialização da rede LSTM
struct LSTMCell* init_lstm_cell(int input_size, int hidden_size){
    struct LSTMCell *cell = (struct LSTMCell*) malloc(sizeof(struct LSTMCell));
    cell->pesos_input = (float*) malloc(sizeof(float) * (input_size + hidden_size));
    cell->pesos_anterior = (float*) malloc(sizeof(float) * (hidden_size + hidden_size));
    cell->biases = (float*) malloc(sizeof(float) * hidden_size);
    cell->estado_celula = (float*) calloc(hidden_size, sizeof(float));
    cell->saida_celula = (float*) calloc(hidden_size, sizeof(float));
}   

