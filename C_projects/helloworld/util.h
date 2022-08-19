#ifndef __UTIL_H__
#define __UTIL_H__

#include<stdlib.h>
#include<string.h>
#include<stdio.h>
#include<stdarg.h>
#include <time.h>
// #include <windows.h>
#include <assert.h>

#define NEW(type) (type *)malloc(sizeof(type))
#define APPLY(task, ...) task(__VA_ARGS__)
#define ABS(n) n > 0 ? n : -n
#define RANGE(min, max, number) number < min ? min : number > max ? max : number
#define EQUAL(a, b) a == b ? 0 : a < b ? -1 : 1
#define log printf

// uuid
char * createUUID();

// ȡĳ
int getNumberLength(int number);

int gettimeofday(struct timeval *tp, void *tzp);

#endif // !__UTIL_H__
