#ifndef  __DATASTRUCT__H__
#define __DATASTRUCT__H__

#include"util.h"

// 实现数据的基本类型


// 字符串类型
#define string char *
#define newString(str) strcpy((char *)malloc(strlen(str) + 1), str)

// 布尔类型
enum _Boolean { True = 1, False = 0 };
typedef enum _Boolean  Boolean;

#define let void *
// #define let int

#endif // ! __DATASTRUCT__H__
