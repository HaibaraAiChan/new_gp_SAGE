#include<stdio.h>
#include<stdlib.h>

#include<string.h>
#include<stdarg.h>
#include<time.h>

struct node
{
    int key;
    int val;
    struct node *next;
};
struct table
{
    int size;
    struct node **list;
};
struct table *createTable(int size)
{
    // size++;
    struct table *t = (struct table *)malloc(sizeof(struct table));
    t->size = size;
    t->list = (struct node **)malloc(sizeof(struct node *) * size);
    int i;
    for (i = 0; i < size; i++)
        t->list[i] = NULL;
    return t;
}
int hashCode(struct table *t, int key)
{
    if (key < 0)
        return -(key % t->size);
    return key % t->size;
}

int printTable(struct table *t)
{
     for (int i =0; i< t->size; i++)
    {
        struct node *list = t->list[i];
        struct node *temp = list;
        printf("key=%d,  val=%d \n", temp->key, temp->val);
    }
    return 0;
}


void insert(struct table *t, int key, int val)
{
    int pos = hashCode(t, key);
    struct node *list = t->list[pos];
    struct node *newNode = (struct node *)malloc(sizeof(struct node));
    struct node *temp = list;
    while (temp)
    {
        if (temp->key == key)
        {
            temp->val = val;
            return;
        }
        temp = temp->next;
    }
    newNode->key = key;
    newNode->val = val;
    newNode->next = list;
    t->list[pos] = newNode;
    // t->size++;
}


int lookup(struct table *t, int key)
{
    int pos = hashCode(t, key);
    struct node *list = t->list[pos];
    struct node *temp = list;
    while (temp)
    {
        if (temp->key == key)
        {
            return temp->val;
        }
        temp = temp->next;
    }
    return -1;
}

int lookupVal(struct table *t, int val)
{
  
    for (int i =0; i< t->size; i++)
    {
        struct node *list = t->list[i];
        struct node *temp = list;
        // printf("temp val=%d,  check val=%d \n", temp->val, val);
        if (temp->val == val)
        {
            // printf("return  check key =%d \n", temp->key);
            return temp->key;
        }
        
    }
    return -1;
}


int removeValByKey(struct table *t, int key)
{
   
    struct node *list = t->list[key];
    struct node *entry = list;
    // printf("======== =entry key=%d,  check key=%d \n", entry->key, key);
       
    if (entry->key == key){
        // printf("  check key =%d \n", entry->key);
        // entry->key = -1;
        entry->val = -1;
        return key;
    }
    
    return -1;
}

#define NELEMS(x)  (sizeof(x) / sizeof(x[0]))

// /* arrange the N elements of ARRAY in random order.
//  * Only effective if N is much smaller than RAND_MAX;
//  * if this may not be the case, use a better random
//  * number generator. */
static void shuffle(void *array, size_t n, size_t size) {
    char tmp[size];
    char *arr = array;
    size_t stride = size * sizeof(char);

    if (n > 1) {
        size_t i;
        for (i = 0; i < n - 1; ++i) {
            size_t rnd = (size_t) rand();
            size_t j = i + rnd / (RAND_MAX / (n - i) + 1);

            memcpy(tmp, arr + j * stride, size);
            memcpy(arr + j * stride, arr + i * stride, size);
            memcpy(arr + i * stride, tmp, size);
        }
    }
}

#define print_type(count, stmt) \
    do { \
    printf("["); \
    for (size_t i = 0; i < (count); ++i) { \
        stmt; \
    } \
    printf("]\n"); \
    } while (0)

struct cmplex {
    int foo;
    double bar;
};


int main()
{
    // int length_src = 2374476;
    // int length_dst = 97922;
    // int length_src = 2000000;
    // int length_dst = 50000;
    int length_src= 62615, length_dst = 6282;
    // int length_src= 62841, length_dst = 6313;
    // int length_src= 62284, length_dst = 6283;
    // int length_src= 58798, length_dst = 5964;
    // int length_src= 57550, length_dst = 5969;
    // int length_src= 124378, length_dst = 12458;///
    // int length_src= 120581, length_dst = 12083;///
    
    int i;
    int intarr[length_src];
    for(i = 0; i < length_src; i++) {
        intarr[i] = i;
    }
    printf("src******** \n");
    // print_type(NELEMS(intarr), printf("%d,", intarr[i]));
    shuffle(intarr, NELEMS(intarr), sizeof(intarr[0]));
	// print_type(NELEMS(intarr), printf("%d,", intarr[i]));
    printf("\n");
    clock_t t_create;
    t_create = clock();
    struct table *t = createTable(length_src*2);
    for (int i=0; i <length_src; i++){
        insert(t, i, intarr[i]);
        // printf("insert %d, %d \n", i, intarr[i]);
    }
    t_create = clock() - t_create;
    double create_hash_table = ((double)t_create)/CLOCKS_PER_SEC; // in seconds
    
    printf("\n");
    //----------------------------------------------------------------------
    // int length_dst = 97922;
    //  int length_dst = 600;
    // int length_dst = 4;
    int intarr_dst[length_dst];
    for(i = 0; i < length_dst; i++) {
        intarr_dst[i] = i;
    }
     printf("dst------ \n");
    // print_type(NELEMS(intarr_dst), printf("%d,", intarr_dst[i]));
    shuffle(intarr_dst, NELEMS(intarr_dst), sizeof(intarr_dst[0]));
	// print_type(NELEMS(intarr_dst), printf("%d,", intarr_dst[i]));
    printf("\n");

    clock_t t_s;
    t_s = clock();
    for(int j=0; j< length_dst; j++){
        // printf("lookup val= %d, key=%d\n\n: ", intarr_dst[j], lookupVal(t, intarr_dst[j]));
        int remove_key = lookup(t, intarr_dst[j]);
        // int output = removeValByKey(t, remove_key);
        // printf("output nodes: %d\n", output);
    }
    // printTable(t);
    t_s = clock() - t_s;
    double time_taken = ((double)t_s)/CLOCKS_PER_SEC; // in seconds
    printf("fun() took %f seconds to execute \n", time_taken);
    printf("total took %f seconds  \n", time_taken+create_hash_table);
  
    // for (int i =0; i< t->size; i++)
    // {
    //     struct node *list = t->list[i];
    //     struct node *temp = list;
    //     // printf("temp val=%d,  check val=%d \n", temp->key, temp->val);
    // }

    return 0;
}