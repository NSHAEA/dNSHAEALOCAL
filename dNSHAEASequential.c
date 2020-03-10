#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <sys/time.h>
#include "include/pcg_variants.h"

#define ITERATIONS 500
#define POPSIZE 1000
#define OPERATORSNUMBER 2
#define DIMENSIONS 1000

int **pop;
double **operators_probability;
int new_pop[POPSIZE][DIMENSIONS + 1];
pcg32_random_t rng;

void initPop();

void initProbabilities();

int fitness(const int *individual);

void replace_pop();

double generate_random();

void apply_genetic_operators(int *new_individual, int individual);

void one_point_crossover(int new_individuals[2][DIMENSIONS + 1], int individual);

void mutation(int *new_individual, int individual);

void copy_array(const int original[], int copy[]);

void normalize();

void print_best();

int main() {
    struct timeval ti, tf;
    gettimeofday(&ti, NULL);
    int i, j;
    double tiempo;

    pop = (int **) malloc(POPSIZE * sizeof(int *));
    operators_probability = (double **) malloc(POPSIZE * sizeof(double *));

    int m;
    for (m = 0; m < POPSIZE; m++) {
        pop[m] = (int *) malloc((DIMENSIONS + 1) * sizeof(int));
        operators_probability[m] = (double *) malloc(OPERATORSNUMBER * sizeof(double));
    }


    srandom((unsigned int) time(NULL));
    uint64_t seeds[2];
    seeds[0] = (uint64_t) random();
    seeds[1] = (uint64_t) random();
    pcg32_srandom_r(&rng, seeds[0], seeds[1]);

    initPop();
    initProbabilities();

    for (i = 0; i < ITERATIONS; i++) {
        for (j = 0; j < POPSIZE; j++) {
            apply_genetic_operators(new_pop[j], j);
        }
        replace_pop();
        print_best();
        printf("\n");
    }
    //print_best();
    gettimeofday(&tf, NULL);   // Instante final
    tiempo = (tf.tv_sec - ti.tv_sec) * 1000 + (tf.tv_usec - ti.tv_usec) / 1000.0;
    printf("\nHas tardado: %g milisegundos\n", tiempo);
    return 0;
}

void initPop() {
    int i, j;
    for (i = 0; i < POPSIZE; i++) {
        for (j = 0; j < DIMENSIONS; j++) {
            pop[i][j] = pcg32_boundedrand_r(&rng, 2);
        }
        pop[i][DIMENSIONS] = fitness(pop[i]);
    }
}

int fitness(const int *individual) {
    int i;
    int count = 0;
    for (i = 0; i < DIMENSIONS; i++) {
        count += individual[i];
    }
    return count;
}

void initProbabilities() {
    int i, j;
    for (i = 0; i < POPSIZE; i++) {
        for (j = 0; j < OPERATORSNUMBER; j++) {
            operators_probability[i][j] = (1.0 / OPERATORSNUMBER);
        }
    }
}

void apply_genetic_operators(int new_individual[], int individual) {
    int i;
    double sum = 0.0;
    double reward = -1.0;
    double random_number;
    //Select genetic operator
    for (i = 0; i < OPERATORSNUMBER; i++) {
        sum += operators_probability[individual][i];
        random_number = generate_random();
        if (random_number < sum)
            break;
    }

    //Cross
    if (i == 0) {
        int new_individuals[2][DIMENSIONS + 1];
        one_point_crossover(new_individuals, individual);
        if (new_individuals[0][DIMENSIONS] > new_individuals[1][DIMENSIONS] &&
            new_individuals[0][DIMENSIONS] > pop[individual][DIMENSIONS]) {
            reward = 1.0;
            copy_array(new_individuals[0], new_individual);
        } else if (new_individuals[1][DIMENSIONS] > new_individuals[0][DIMENSIONS] &&
                   new_individuals[1][DIMENSIONS] > pop[individual][DIMENSIONS]) {
            reward = 1.0;
            copy_array(new_individuals[1], new_individual);
        } else {
            copy_array(pop[individual], new_individual);
        }
    }
        //Mutation
    else if (i == 1) {
        int new_individual_mutation[DIMENSIONS + 1];
        mutation(new_individual_mutation, individual);
        if (new_individual_mutation[DIMENSIONS] > pop[individual][DIMENSIONS]) {
            reward = 1.0;
            copy_array(new_individual_mutation, new_individual);
        } else {
            copy_array(pop[individual], new_individual);
        }
    }

    double reward_weight = 1.0 + (reward * generate_random());
    operators_probability[individual][i] *= reward_weight;
    normalize();
}

void replace_pop() {
    int i, j;
    for (i = 0; i < POPSIZE; i++) {
        for (j = 0; j < DIMENSIONS + 1; j++) {
            pop[i][j] = new_pop[i][j];
        }
    }
}

void copy_array(const int original[], int copy[]) {
    int i;
    for (i = 0; i < DIMENSIONS + 1; i++) {
        copy[i] = original[i];
    }
}

double generate_random() {
    int h = pcg32_boundedrand_r(&rng, 1000);
    return h / 1000.0;
}

void one_point_crossover(int new_individuals[2][DIMENSIONS + 1], int individual) {
    int cross_point = pcg32_boundedrand_r(&rng, DIMENSIONS);
    int other_parent = pcg32_boundedrand_r(&rng, POPSIZE);
    int i;
    for (i = 0; i < cross_point; i++) {
        new_individuals[0][i] = pop[individual][i];
        new_individuals[1][i] = pop[other_parent][i];
    }
    for (i = cross_point; i < DIMENSIONS; i++) {
        new_individuals[0][i] = pop[other_parent][i];
        new_individuals[1][i] = pop[individual][i];
    }

    new_individuals[0][DIMENSIONS] = fitness(new_individuals[0]);
    new_individuals[1][DIMENSIONS] = fitness(new_individuals[1]);
}

void mutation(int *new_individual, int individual) {
    int mutate_point = pcg32_boundedrand_r(&rng, DIMENSIONS);
    int i;
    for (i = 0; i < DIMENSIONS; i++) {
        if (i == mutate_point)
            new_individual[i] = 1 - pop[individual][i];
        else
            new_individual[i] = pop[individual][i];
    }
    new_individual[DIMENSIONS] = fitness(new_individual);
}

void normalize() {
    int i, j;
    double sum[POPSIZE];
    for (i = 0; i < POPSIZE; i++) {
        sum[i] = 0.0;
        for (j = 0; j < OPERATORSNUMBER; j++) {
            sum[i] += operators_probability[i][j];
        }
        for (j = 0; j < OPERATORSNUMBER; j++) {
            operators_probability[i][j] /= sum[i];
        }
    }
}

void print_best() {
    int best = 0, i, pos_best = 0;
    for (i = 0; i < POPSIZE; i++) {
        if (pop[i][DIMENSIONS] > best) {
            best = pop[i][DIMENSIONS];
            pos_best = i;
        }
    }
    for (i = 0; i < DIMENSIONS; i++) {
        printf("%i", pop[pos_best][i]);
    }
    printf(" %i ", pop[pos_best][DIMENSIONS]);

    for (i = 0; i < OPERATORSNUMBER; i++) {
        printf("%.3f ", operators_probability[pos_best][i]);
    }
}

