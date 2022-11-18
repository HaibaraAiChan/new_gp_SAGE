import sys
import dgl
import torch
import numpy as np
# from dgl.frame import torchrame
# from dgl.udf import EdgeBatch, NodeBatch





def bucket_split(graph,  *, orig_nid=None):
    degs = graph.in_degrees()
    nodes = graph.dstnodes()
    if orig_nid is None:
        orig_nid = nodes
    ntype = graph.dsttypes[0]
    ntid = graph.get_ntype_id_from_dst(ntype)
    dstdata = graph._node_frames[ntid]
    # msgdata = torchrame(msgdata)

    # degree bucketing
    unique_degs, bucketor = _bucketing(degs)
    bkt_rsts = []
    bkt_nodes = []
    for deg, node_bkt, orig_nid_bkt in zip(
        unique_degs, bucketor(nodes), bucketor(orig_nid)
    ):
        if deg == 0:
            # skip reduce function for zero-degree nodes
            continue
        bkt_nodes.append(node_bkt)
        # ndata_bkt = dstdata.subframe(node_bkt)

        # order the incoming edges per node by edge ID
        eid_bkt = torch.zerocopy_to_numpy(graph.in_edges(node_bkt, form="eid"))
        assert len(eid_bkt) == deg * len(node_bkt)
        eid_bkt = np.sort(eid_bkt.reshape((len(node_bkt), deg)), 1)
        eid_bkt = torch.zerocopy_from_numpy(eid_bkt.flatten())
	
    return eid_bkt

def _bucketing(val):
    """Internal function to create groups on the values.

    Parameters
    ----------
    val : Tensor
        Value tensor.

    Returns
    -------
    unique_val : Tensor
        Unique values.
    bucketor : callable[Tensor -> list[Tensor]]
        A bucketing function that splits the given tensor data as the same
        way of how the :attr:`val` tensor is grouped.
    """
    sorted_val, idx = torch.sort(val)
    unique_val = torch.unique(sorted_val)
    bkt_idx = []
    
    # #---------------------------------------------------------------orignal part
    # for v in unique_val:
    #     eqidx = torch.nonzero_1d(torch.equal(sorted_val, v))
    #     bkt_idx.append(torch.gather_row(idx, eqidx))
    # #---------------------------------------------------------------orignal part
    #--------------------------------------------------------------*-replaced part
    My_selected_degree = torch.tensor(2)
    unique_val = [My_selected_degree]
    
    for v in unique_val:
        eqidx = torch.equal(sorted_val, v)
        temp = torch.gather_row(idx, eqidx)
        bkt_idx.append(torch.gather_row(idx, eqidx))
        print('bucketing info ===========------------=============')
        print(eqidx)
        print(torch.gather_row(idx, eqidx))
        print('bucketing info ===========------------=============')

    #--------------------------------------------------------------*-replaced part
    # for v in unique_val:
    #     eqidx = torch.nonzero_1d(torch.equal(sorted_val, v))
    #     tmp = torch.gather_row(idx, eqidx)
    #     if v ==  unique_val[-1]:
    #         nn = len(tmp)//2
    #         bkt_idx.append(tmp[:nn+1])
    #         bkt_idx.append(tmp[nn+1:])
    #     else:
    #         bkt_idx.append(tmp)

    # unique_val = np.append(unique_val,unique_val[-1]) # add the number of largest degree

    #--------------------------------------------------------------- 4 splits 
    # for v in unique_val:
    #     eqidx = torch.nonzero_1d(torch.equal(sorted_val, v))
    #     tmp = torch.gather_row(idx, eqidx)
    #     if v ==  unique_val[-1]:
    #         nn = len(tmp)//4
    #         bkt_idx.append(tmp[:nn+1])
    #         bkt_idx.append(tmp[nn+1:2*nn+1])
    #         bkt_idx.append(tmp[2*nn+1:3*nn+1])
    #         bkt_idx.append(tmp[3*nn+1:])
    #     else:
    #         bkt_idx.append(tmp)

    # unique_val = np.append(unique_val,unique_val[-1]) # add the number of largest degree
    # unique_val = np.append(unique_val,unique_val[-1])
    # unique_val = np.append(unique_val,unique_val[-1])
    # unique_val = np.append(unique_val,unique_val[-1])
    #--------------------------------------------------------------- 
    #--------------------------------------------------------------- N splits 
    # num_split = 16                                                 # N = 16
    # for v in unique_val:
    #     eqidx = torch.nonzero_1d(torch.equal(sorted_val, v))
    #     tmp = torch.gather_row(idx, eqidx)
    #     if v ==  unique_val[-1]:
    #         nn = len(tmp)//num_split
    #         bkt_idx.append(tmp[:nn+1])
    #         for i in range(1,num_split-1):
    #             bkt_idx.append(tmp[i*nn+1:((i+1)*nn+1)])
    #         bkt_idx.append(tmp[(num_split-1)*nn+1:])
    #     else:
    #         bkt_idx.append(tmp)
    # tail_degree = unique_val[-1]
    # for i in range(num_split):
    #     unique_val = np.append(unique_val,tail_degree) # add the number of largest degree
    
    #--------------------------------------------------------------- 
    def bucketor(data):
        bkts = [torch.gather_row(data, idx) for idx in bkt_idx]
        return bkts
    
    return unique_val, bucketor

