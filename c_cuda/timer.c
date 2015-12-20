#include <sys/time.h>
#include "timer.h"

void wtime(double* t);

static double elapsed_time() {
    double t;
    wtime(&t);
    return t;
}

static double start[64], elapsed[64];

void timer_clear(int n) {
    elapsed[n] = 0.0;
}

void timer_start(int n) {
    start[n] = elapsed_time();
}

void timer_stop(int n) {
    double t, now;

    now = elapsed_time();
    t = now - start[n];
    elapsed[n] += t;
}

double timer_read(int n) {
    return elapsed[n];
}

void wtime(double* t) {
    static int sec = -1;
    struct timeval tv;
    gettimeofday(&tv, (void*)0);
    if (sec < 0) sec = tv.tv_sec;
    *t = (tv.tv_sec - sec) + 1.0e-6 * tv.tv_usec;
}
