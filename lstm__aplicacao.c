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
    // Inicializar pesos e bias com valores aleatórios
    for(int i = 0; i < input_size + hidden_size; i++){
        cell -> pesos_input[i] = ((float) rand())/((float)RAND_MAX); 
    }
    for(int i = 0; i < hidden_size + hidden_size; i++ ){
        cell->pesos_anterior[i] = ((float) rand()) / ((float) RAND_MAX);
    }
    for(int i = 0; i < hidden_size; i++){
        cell->biases[i] = ((float) rand()) / ((float) RAND_MAX);
    }
    return cell;
}   

// Passo de feedforward da célula LSTM
void lstm_passo(struct LSTMCell *cell, float *input, float *prev_hidden_state){
    int hidden_size = sizeof(cell->hidden_state) / sizeof(float);

    //Calcular entrada total
    float *total_input = (float*) malloc(sizeof(float) * (hidden_size * 4));
    for(int i = 0; i < hidden_size * 4; i++){
        total_input[i] = 0;
    }
    for(int i = 0; i < hidden_size; i++){
        for (int j = 0; j <hidden_size + 1; j++){
            if(j < input_size){
                total_input[i] += cell->biases[i];
            }
            else{
                if(j < input_size){
                    total_input[i] += input[j] * cell->pesos_input[i * input_size + j];
                }
                else{
                    total_input[i] += prev_hidden_state[j] * cell->pesos_anterior[i * hidden_size + j - hidden_size];
                }
            }
        }
    }
    //Calcular portão do esquecimento (forget gate)
    float *forget_gate = (float*) malloc(sizeof(float) * hidden_size);
    for(int i = 0; i < hidden_size; i++){
        forget_gate[i] = sigmoid(total_input[i]);
    }

}

//Calcular o portão de entrada
float *input_gate = (float*) malloc(sizeof(float) * hidden_size);
for(int i = 0; i < hidden_size; i++){
    input_gate[i] = sigmoid(total_input[i + hidden_size]);
}

//calcular candidato a estado da célula
float *candidate_cell_state = (float*) malloc(sizeof(float) * hidden_size);
for(int i = 0; i < hidden_size; i++){
    candidate_cell_state[i] = tanh(total_input[i + hidden_size * 2]);
}

// calcular estado da célula atualizado
for(int i = 0; i < hidden_size; i++){
    cell->estado_celula[i] = cell->estado_celula[i] * forget_gate[i] + candidate_cell_state[i] * input_gate[i];
}

//Calcular outputgate (portão de saida)
float *outputgate = (float*) malloc(sizeof(float) * hidden_size);
for(int i = 0; i < hidden_size; i++){
    outputgate[i] = sigmoid(total_input[i + hidden_size *3]);
}

//Calcular saída
for(int i = 0; i < hidden_size; i++){
    cell->hidden_state[i] = tanh(cell->cell_State[i]) * outputgate[i];
}

free(total_input);
free(forget_gate);
free(input_gate);
free(candidate_cell_state);
free(outputgate);

