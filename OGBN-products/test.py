# import dgl
# g = dgl.graph(([0, 0, 1, 2, 2, 5], [1, 3, 2, 3, 4, 1]))
# Block = dgl.to_block(g)
# print('Block.srcdata')
# print(Block.srcdata)
# print('Block.dstdata')
# print(Block.dstdata)
# print('Block.edges')
# print(Block.edges('all'))
# print()


# g2 = dgl.graph(([2, 1, 5, 2, 0, 0], [4, 2, 1, 3, 1, 3]))
# Block2 = dgl.to_block(g2)
# print('Block2.srcdata')
# print(Block2.srcdata)
# print('Block2.dstdata')
# print(Block2.dstdata)
# print('Block2.edges')
# print(Block2.edges('all'))
import multiprocessing as mp
# from multiprocessing import freeze_support

def go_async(index, value) :
    return str(index * int(value))

def log_result(result):
    print("Succesfully get callback! With result: ", result)

def main() :
    array = [1,3,4,5,6,7]
    pool = mp.Pool() 
    res = pool.starmap_async(go_async, enumerate(array), callback = log_result).get()        
    print("Final result: ", res)
    pool.close()
    pool.join()

if __name__ == '__main__':    
    # freeze_support()
    main()