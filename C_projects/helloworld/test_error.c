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
struct dict
{
    int size;
    struct node *header;
    struct node *end;
};
struct dict *createDict()
{
    struct dict *t = (struct dict *)malloc(sizeof(struct dict));
    t->size = 0;
    t->header = t->end = NULL;
    // (struct node **)malloc(sizeof(struct node *))
    return t;
}
int hashCode(struct dict *t, int key)
{
    if (key < 0)
        return -(key % t->size);
    return key % t->size;
}

void printTable(struct dict *t)
{
     for (int i =0; i< t->size; i++)
    {
        struct node *list = t->list[i];
        struct node *temp = list;
        printf("key=%d,  val=%d \n", temp->key, temp->val);
        
    }
}


void insert(struct dict *t, int key, int val)
{
    // int pos = hashCode(t, key);
    struct node *newNode = (struct node *)malloc(sizeof(struct node));
    struct node *temp = t->header;
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
    newNode->next = NULL;
    t->end = newNode;
    t->size++;
}
int lookup(struct dict *t, int key)
{
    // int pos = hashCode(t, key);
    struct node *temp = t->header;
    
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

int lookupVal(struct dict *t, int val)
{
    struct node *temp = t->header;
    while (temp)
    {
        if (temp->val == val)
        {
            return temp->val;
        }
        temp = temp->next;
    }
    return -1;
}




int removeKey(struct dict *t, int key)
{
   
    struct node *entry = t->header;
    printf("======== =entry key=%d,  check key=%d \n", entry->key, key);
     while (entry)
    {
        if (entry->key == key){
            printf("  check key =%d \n", entry->key);
            
            return key;
        }
        entry = entry->next;
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
    int length_src = 10;
    int i;
    int intarr[length_src];
    for(i = 0; i < length_src; i++) {
        intarr[i] = i;
    }
    printf("src******** \n");
    // print_type(NELEMS(intarr), printf("%d,", intarr[i]));
    shuffle(intarr, NELEMS(intarr), sizeof(intarr[0]));
	print_type(NELEMS(intarr), printf("%d,", intarr[i]));
    printf("\n");

    struct table *t = createTable(length_src*2);
    for (int i=0; i <length_src; i++){
        insert(t, i, intarr[i]);
        printf("insert %d, %d \n", i, intarr[i]);
    }
    printf("\n");
    //----------------------------------------------------------------------
    // int length_dst = 97922;
    int length_dst = 1;
    int intarr_dst[length_dst];
    for(i = 0; i < length_dst; i++) {
        intarr_dst[i] = i;
    }
     printf("dst------ \n");
    // print_type(NELEMS(intarr_dst), printf("%d,", intarr_dst[i]));
    shuffle(intarr_dst, NELEMS(intarr_dst), sizeof(intarr_dst[0]));
	print_type(NELEMS(intarr_dst), printf("%d,", intarr_dst[i]));
    printf("\n");

    for(int j=0; j< length_dst; j++){
        // printf("lookup val= %d, key=%d\n\n: ", intarr_dst[j], lookupVal(t, intarr_dst[j]));
        int remove_key = lookupVal(t, intarr_dst[j]);
        int output = removeKey(t, remove_key);
        printf("output nodes: %d\n: ", output);
    }
    printTable(t);


    return 0;
}