#include"hashMap.h"

// #include <stdio.h>
#include <stdlib.h>


int defaultHashCode(HashMap hashMap, let key) {
	char * k = (char *)key;
	// int k = (int)key;
	unsigned long h = 0;
	while (*k) {
		h = (h << 4) + *k++;
		unsigned long g = h & 0xF0000000L;
		if (g) {
			h ^= g >> 24;
		}
		h &= ~g;
	}
	return h % hashMap->listSize;
}

Boolean defaultEqual(let key1, let key2) {
	return strcmp((string)key1, (string)key2) ? False : True;
}

void resetHashMap(HashMap hashMap, int listSize) {

	if (listSize < 8) return;

	// 键值对临时存储空间
	Entry tempList = newEntryList(hashMap->size);

	HashMapIterator iterator = createHashMapIterator(hashMap);
	int length = hashMap->size;
	for (int index = 0; hasNextHashMapIterator(iterator); index++) {
		// 迭代取出所有键值对
		iterator = nextHashMapIterator(iterator);
		tempList[index].key = iterator->entry->key;
		tempList[index].value = iterator->entry->value;
		tempList[index].next = NULL;
	}
	freeHashMapIterator(&iterator);

	// 清除原有键值对数据
	hashMap->size = 0;
	for (int i = 0; i < hashMap->listSize; i++) {
		Entry current = &hashMap->list[i];
		current->key = NULL;
		current->value = NULL;
		if (current->next != NULL) {
			while (current->next != NULL) {
				Entry temp = current->next->next;
				free(current->next);
				current->next = temp;
			}
		}
	}

	// 更改内存大小
	hashMap->listSize = listSize;
	Entry relist = (Entry)realloc(hashMap->list, hashMap->listSize * sizeof(struct entry));
	if (relist != NULL) {
		hashMap->list = relist;
		relist = NULL;
	}

	// 初始化数据
	for (int i = 0; i < hashMap->listSize; i++) {
		hashMap->list[i].key = NULL;
		hashMap->list[i].value = NULL;
		hashMap->list[i].next = NULL;
	}

	// 将所有键值对重新写入内存
	for (int i = 0; i < length; i++) {
		hashMap->put(hashMap, tempList[i].key, tempList[i].value);
	}
	free(tempList);
}

void defaultPut(HashMap hashMap, let key, let value) {
	// 获取哈希值
	int index = hashMap->hashCode(hashMap, key);

	if (hashMap->list[index].key == NULL) {
		hashMap->size++;
		// 该地址为空时直接存储
		hashMap->list[index].key = key;
		hashMap->list[index].value = value;
		
	}
	else {

		Entry current = &hashMap->list[index];
		while (current != NULL) {
			if (hashMap->equal(key, current->key)) {
				// 对于键值已经存在的直接覆盖
				current->value = value;
				return;
			}
			current = current->next;
		};

		// 发生冲突则创建节点挂到相应位置的next上
		Entry entry = newEntry();
		entry->key = key;
		entry->value = value;
		entry->next = hashMap->list[index].next;
		hashMap->list[index].next = entry;
		hashMap->size++;
	}

	if (hashMap->autoAssign && hashMap->size >= hashMap->listSize) {

		// 内存扩充至原来的两倍
		// *注: 扩充时考虑的是当前存储元素数量与存储空间的大小关系，而不是存储空间是否已经存满，
		// 例如: 存储空间为10，存入了10个键值对，但是全部冲突了，所以存储空间空着9个，其余的全部挂在一个上面，
		// 这样检索的时候和遍历查询没有什么区别了，可以简单这样理解，当我存入第11个键值对的时候一定会发生冲突，
		// 这是由哈希函数本身的特性(取模)决定的，冲突就会导致检索变慢，所以这时候扩充存储空间，对原有键值对进行
		// 再次散列，会把冲突的数据再次分散开，加快索引定位速度。
		resetHashMap(hashMap, hashMap->listSize * 2);
	}
}

let defaultGet(HashMap hashMap, let key) {
	if (hashMap->exists(hashMap, key)) {
		int index = hashMap->hashCode(hashMap, key);
		Entry entry = &hashMap->list[index];
		while (entry != NULL) {
			if (hashMap->equal(entry->key, key)) {
				return entry->value;
			}
			entry = entry->next;
		}
	}
	return NULL;
}

let defaultRemove(HashMap hashMap, let key) {
	int index = hashMap->hashCode(hashMap, key);
	Entry entry = &hashMap->list[index];
	if (entry->key == NULL) {
		return NULL;
	}
    let entryKey = entry->key;
	Boolean result = False;
	if (hashMap->equal(entry->key, key)) {
		hashMap->size--;
		if (entry->next != NULL) {
			Entry temp = entry->next;
			entry->key = temp->key;
			entry->value = temp->value;
			entry->next = temp->next;
			free(temp);
		}
        else {
            entry->key = NULL;
            entry->value = NULL;
        }
		result = True;
	}
	else {
		Entry p = entry;
		entry = entry->next;
		while (entry != NULL) {
			if (hashMap->equal(entry->key, key)) {
				hashMap->size--;
				p->next = entry->next;
				free(entry);
				result = True;
				break;
			}
			p = entry;
			entry = entry->next;
		};
	}

	// 如果空间占用不足一半，则释放多余内存
	if (result && hashMap->autoAssign &&  hashMap->size < hashMap->listSize / 2) {
		resetHashMap(hashMap, hashMap->listSize / 2);
	}
	return entryKey;
}

Boolean defaultExists(HashMap hashMap, let key) {
	int index = hashMap->hashCode(hashMap, key);
	Entry entry = &hashMap->list[index];
	if (entry->key == NULL) {
		return False;
	}
	else {
		while (entry != NULL) {
			if (hashMap->equal(entry->key, key)) {
				return True;
			}
			entry = entry->next;
		}
		return False;
	}
}

void defaultClear(HashMap hashMap) {
	for (int i = 0; i < hashMap->listSize; i++) {
		// 释放冲突值内存
		Entry entry = hashMap->list[i].next;
		while (entry != NULL) {
			Entry next = entry->next;
			free(entry);
			entry = next;
		}
		hashMap->list[i].next = NULL;
	}
	// 释放存储空间
	free(hashMap->list);
	hashMap->list = NULL;
	hashMap->size = -1;
	hashMap->listSize = 0;
}

HashMap createHashMap(HashCode hashCode, Equal equal) {
	HashMap hashMap = newHashMap();
	if (hashMap == NULL) {
		return NULL;
	}
	hashMap->size = 0;
	hashMap->listSize = 8;
	hashMap->hashCode = hashCode == NULL ? defaultHashCode : hashCode;
	hashMap->equal = equal == NULL ? defaultEqual : equal;
	hashMap->exists = defaultExists;
	hashMap->get = defaultGet;
	hashMap->put = defaultPut;
	hashMap->remove = defaultRemove;
	hashMap->clear = defaultClear;
	hashMap->autoAssign = True;
	
	// 起始分配8个内存空间，溢出时会自动扩充
	hashMap->list = newEntryList(hashMap->listSize);
	if (hashMap->list == NULL) {
		return NULL;
	}
	Entry p = hashMap->list;
	for (int i = 0; i < hashMap->listSize; i++) {
		p[i].key = p[i].value = p[i].next = NULL;
	}
	return hashMap;
}

HashMapIterator createHashMapIterator(HashMap hashMap) {
	HashMapIterator iterator = newHashMapIterator();
	if (iterator == NULL) {
		return NULL;
	}
	iterator->hashMap = hashMap;
	iterator->count = 0;
	iterator->hashCode = -1;
	iterator->entry = NULL;
	return iterator;
}

Boolean hasNextHashMapIterator(HashMapIterator iterator) {
	return iterator->count < iterator->hashMap->size ? True : False;
}

HashMapIterator nextHashMapIterator(HashMapIterator iterator) {
	if (hasNextHashMapIterator(iterator)) {
		if (iterator->entry != NULL && iterator->entry->next != NULL) {
			iterator->count++;
			iterator->entry = iterator->entry->next;
			return iterator;
		}
		while (++iterator->hashCode < iterator->hashMap->listSize) {
			Entry entry = &iterator->hashMap->list[iterator->hashCode];
			if (entry->key != NULL) {
				iterator->count++;
				iterator->entry = entry;
				break;
			}
		}
	}
	return iterator;
}

void freeHashMapIterator(HashMapIterator * iterator) {
	free(*iterator);
	*iterator = NULL;
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

#define Put(map, key, value) map->put(map, (let)key, (let)value);
#define Get(map, key) (char *)map->get(map, (let)key)
// #define Get(map, key) (int)map->get(map, (let)key)
#define Remove(map, key) map->remove(map, (let)key)
#define Existe(map, key) map->exists(map, (let)key)
// void reverse(char s[])
//  {
//      int i, j;
//      char c;

//      for (i = 0, j = strlen(s)-1; i<j; i++, j--) {
//          c = s[i];
//          s[i] = s[j];
//          s[j] = c;
//      }
// } 

// void itoa(int n, char s[])
//  {
//      int i, sign;

//      if ((sign = n) < 0)  /* record sign */
//          n = -n;          /* make n positive */
//      i = 0;
//      do {       /* generate digits in reverse order */
//          s[i++] = n % 10 + '0';   /* get next digit */
//      } while ((n /= 10) > 0);     /* delete it */
//      if (sign < 0)
//          s[i++] = '-';
//      s[i] = '\0';
//      reverse(s);
// } 

char *itoa(int num, char *str)
{
        if(str == NULL)
        {
                return NULL;
        }
        sprintf(str, "%d", num);
        return str;
}


int main() {
	// int a = 97922;
	int a = 10;
    int i;
    int intarr[a];
    for(i = 0; i < a; i++) {
        intarr[i] = i + 9;
    }
    
    print_type(NELEMS(intarr), printf("%d,", intarr[i]));
    shuffle(intarr, NELEMS(intarr), sizeof(intarr[0]));
	print_type(NELEMS(intarr), printf("%d,", intarr[i]));

    HashMap map = createHashMap(NULL, NULL);
	// Put(map, "1", "2");
	// printf("key: 1, exists: %s\n", Existe(map, "1") ? "true" : "false");

	// Put(map, 1, 2);
	// printf("key: 1, exists: %s\n", Existe(map, 1) ? "true" : "false");
	char key [33], val [33];
	
	for(i = 0; i < a; i++) {
		itoa(i, key);
		itoa(intarr[i], val);
		printf("key, val =(%s,%s)\n", key, val);
		if (key == "9"){
			printf("----------------------------key %s\n", key);
			if (i == 9){
				printf("----------------------------i=9, key= %d\n", 9+1);
			}
			
		}

		Put(map, key, val);
        // Put(map, itoa(i, buffer, 10), itoa(intarr[i], buffer2, 10));
    } 
    // Put(map, "asdfasdf", "asdfasdfds");
    // Put(map, "sasdasd", "asdfasdfds");
    // Put(map, "asdhfgh", "asdfasdfds");
    // Put(map, "4545", "asdfasdfds");
    // Put(map, "asdfaasdasdsdf", "asdfasdfds");
    // Put(map, "asdasg", "asdfasdfds");
    // Put(map, "qweqeqwe", "asdfasdfds");
	

    // printf("key: 4545, exists: %s\n", Existe(map, "4545") ? "true" : "false");
    // printf("4545: %s\n", Get(map, "4545"));
    // printf("remove 4545 %s\n", Remove(map, "4545") ? "true" : "false");
    // printf("remove 4545 %s\n", Remove(map, "4545") ? "true" : "false");
    // printf("key: 4545, exists: %s\n", Existe(map, "4545") ? "true" : "false");
	char test_key[33];
	itoa(9, test_key);
	
	printf("key: 9, exists: %s\n", Existe(map, test_key) ? "true" : "false");
    printf("9: %s\n", Get(map, test_key));
	printf("remove 9 %s\n", Remove(map, test_key) ? "true" : "false");
	printf("remove 9 %s\n", Remove(map, test_key) ? "true" : "false");
    printf("key: 9, exists: %s\n", Existe(map, test_key) ? "true" : "false");



    HashMapIterator iterator = createHashMapIterator(map);
    while (hasNextHashMapIterator(iterator)) {
        iterator = nextHashMapIterator(iterator);
        printf("{ key: %s, value: %s, hashcode: %d }\n",
            (char *)iterator->entry->key, (char *)iterator->entry->value, iterator->hashCode);
			
    }
    map->clear(map);
    freeHashMapIterator(&iterator);

    return 0;
}