#ifndef __ARRAY_H__
#define __ARRAY_H__

#include"dataStruct.h"

// 双向链表
typedef struct list {
	let item;
	struct list * next;
	struct list * last;
}*ListNode, *List;

#define newList() NEW(struct list)
#define newListNode() NEW(struct list)

typedef struct array *Array;

typedef Array(*Concat)(Array a, Array b);

typedef let (*Pop)(Array array);

typedef int(*Push)(Array array, let item);

typedef void(*Reverse)(Array array);

typedef let (*Shift)(Array array);

typedef int (*Unshift)(Array array, let item);

typedef Array(*Slice)(Array array, int start, int end);

typedef int(*SortBy)(let item1, let item2);

typedef void(*Sort)(Array array, SortBy sort);

typedef Boolean(*FilterBy)(let item);

typedef Array(*Filter)(Array array, FilterBy filter);

typedef let (*IndexOf)(Array array, int index);

typedef struct array {
	Pop pop;
	List head;
	List tail;
	Sort sort;
	Push push;
	int length;
	Shift shift;
	Slice slice;
	Filter filter;
	Concat concat;
	Reverse reverse;
	Unshift unshift;
	IndexOf indexOf;
}*Array;

#define newArray() NEW(struct array)

// 迭代器结构
typedef struct arrayIterator {
	ListNode node;	// 迭代器当前指向
	int count;		// 迭代次数
	Array array;
}*ArrayIterator;

#define newArrayIterator() NEW(struct arrayIterator)

// 连接两个数组，返回一个新的数组
static Array concat(Array a, Array b);

// 返回数组最后一个元素
static let pop(Array a);

// 在数组最后插入一个元素，返回新长度
static int push(Array array, let item);

// 反转整个数组，会修改原数组
static void reverse(Array array);

// 删除并返回数组的第一个元素
static let shift(Array array);

// 删除并返回数组的第一个元素
static int unshift(Array array, let item);

// 从数组中选取一部分，构成一个新数组返回
static Array slice(Array array, int start, int end);

// 对数组的元素进行排序
static void sort(Array array, SortBy sort);

// 过滤出一部分元素，filter返回True时保留，构成新数组返回
static Array filter(Array array, FilterBy filter);

// 返回指定索引位置的值
static let indexOf(Array array, int index);

// 创建数组
Array createArray(int size);

// 释放数组
void freeArray(Array * array);

// 创建数组迭代器
ArrayIterator createArrayIterator(Array array);

// 迭代器是否有下一个
Boolean hasNextArrayIterator(ArrayIterator iterator);

// 迭代到下一次
ArrayIterator nextArrayIterator(ArrayIterator iterator);

// 释放迭代器内存
void freeArrayIterator(ArrayIterator * iterator);

#endif // !__ARRAY_H__
