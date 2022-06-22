"""
Some MPI tools for analyse_likelihood code.
"""

import numpy as np
from mpi4py import MPI

#can only broadcast string (not a type), so first get a string, the after broadcasting, get mpi type
def np_dtype(np_dtype):
    if np_dtype == np.float32:
        return 'float32'
    elif np_dtype == np.float64: 
        return 'float64'
    elif np_dtype == np.int64:
        return 'int64'
    elif np_dtype == np.int32:
        return 'int32'
#can only broadcast string, so first get a string, the after broadcasting, get mpi type
mpi_dtype_dict = {'float32' : MPI.FLOAT, 'float64' : MPI.DOUBLE, 'int64' : MPI.LONG, 'int32' : MPI.INT}
np_dtype_dict = {'float32' : np.float32, 'float64' : np.float64, 'int64' : np.int64, 'int32' : np.int32}    #may be unnec, but to be safe.


def gather_array_by_col(senddata, comm, root=0):
    """
    by col I mean the data being gathered has been split along -1 axis and gather these chunks.
    counts and starts can be derived from the length along -1 axis! Could be more flexible, but this is thw way I gather and scatter eveything here, so should be fine. 
    maybe nice to add option of using starts and counts as args
    works for 1d,2d and prob higher!
    """
    
    size = comm.Get_size()
    rank = comm.Get_rank()  

    senddata_shape = senddata.shape
    senddata_dtype_str = (np_dtype(senddata.dtype))
    senddataT = senddata.T.copy()
    #print(senddata, senddata_shape)
    Ncols_send = senddata_shape[-1]
    Ncols_gath = comm.gather(Ncols_send, root)   #gather all the shape[-1]s of the arrays to be gathered (the send arrays). the sum of these is Ncols_recv. 
    Ncols_recv = sum(comm.bcast(Ncols_gath, root))
    #print(Ncols_recv)
    count = Ncols_recv // size + ( Ncols_recv % size > rank )    #this is count_rank
    start = comm.scan(count) - count         #start_rank

    counts = comm.gather(count, root=root)   #this is a tuple of count on each proc
    starts = comm.gather(start, root=root)

    # compute the shapes of the gatehred arrays
    senddata_shape = list(senddata_shape)
    recvdata_shape = senddata_shape.copy()  #all dims should have same shape except for -1
    recvdata_shape[-1] = Ncols_recv
    recvdata_shape = tuple(recvdata_shape)
    senddata_shape = tuple(senddata_shape)
        
    recvdata = None
    recvdataT = None
    if rank == root: 
        recvdataT = np.empty(recvdata_shape[::-1], dtype=senddata.dtype)  #careful with dtype! use same as sendata to be safe. remember to reverse shape when definind tranpose!

    senddata_dtype_mpi = mpi_dtype_dict[senddata_dtype_str]
    coltype = senddata_dtype_mpi.Create_contiguous(np.prod([s for s in senddata_shape[:-1]]))   #this should be ndim general. each 'column', i.e.e each [...,i] contains this many elements.
    coltype.Commit()
    sendbuf = senddataT
    recvbuf = [recvdataT, counts, starts, coltype]
    comm.Gatherv(sendbuf, recvbuf, root)
    coltype.Free()
        
    if rank == root:
        recvdata = recvdataT.T.copy()
    
    return recvdata

def scatter_array_by_col(senddata, comm, root=0):
    """
    For scattering you must input the sendata such that it is None on all ranks except the root.
    by col I mean split along -1 axis and scatter these chunks.
    counts and starts can be derived from the length along -1 axis! Could be more flexible, but this is the way I gather and scatter eveything here, so should be fine. 
    maybe nice to add option of using starts and counts as args
    works for 1d,2d and prob higher!
    """
    size = comm.Get_size()   # number of ranks (32 on Cori for one proc)
    rank = comm.Get_rank()   # the rank id of the rank (a number between 0 and 31)
    
    senddata_shape = None
    senddata_dtype_str = None
    senddataT = None
    if rank == 0:
        senddata_shape = senddata.shape                                      # currently this is the global peak indices
        senddata_dtype_str = (np_dtype(senddata.dtype))                       
        senddataT = senddata.T.copy()
        
    senddata_shape = comm.bcast(senddata_shape, root)   #can broacast to all procs with same variable name
    senddata_dtype_str = comm.bcast(senddata_dtype_str, root)   #can broacast to all procs with same variable name
    senddata_dtype_mpi = mpi_dtype_dict[senddata_dtype_str]
    coltype = senddata_dtype_mpi.Create_contiguous(np.prod([s for s in senddata_shape[:-1]]))
        
    # compute the shapes of the scattered arrays
    senddata_shape = list(senddata_shape)
    recvdata_shape = senddata_shape.copy()  #all dims should have same shape except for -1
    recvdata_shape[-1] = senddata_shape[-1] // size + (senddata_shape[-1] % size > rank)   # define the length of the array to be recieved by each rank (this allows for non divisible len_size / size).
    recvdata_shape = tuple(recvdata_shape)
    senddata_shape = tuple(senddata_shape)

    count_rank = recvdata_shape[-1]                          #this is the number on counts for each rank (count_rank), but we want a tuple of counts on the root rank for Scatterv, so gather on next line:
    counts = comm.gather(count_rank, root=root)

    start_rank = comm.scan(count_rank) - count_rank
    starts = comm.gather(start_rank, root=root)

    #initialize the transpose recv data array with correct rank dep size
    recvdataT = np.empty(recvdata_shape[::-1], dtype=np_dtype_dict[senddata_dtype_str])   

    coltype.Commit()
    sendbuf = [senddataT, counts, starts, coltype]     # CAREFUL np.float64 = MPI.DOUBLE()  IS THERE NOW WAY OF USING  FINDING THE SIZE FROM senddata itself?
    comm.Scatterv(sendbuf, recvdataT, root)              # no need for recbuf here. everything fits in nicely to  recvdata as we have defined itxs
    coltype.Free()

    recvdata = recvdataT.T.copy()   #dont think copy is necessary because not really goign to be doing memory related stuff after this, but will copy to be safe
    
    return recvdata