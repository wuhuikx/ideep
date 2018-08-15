import time
import numpy
import ideep4py
from ideep4py import distribute

if not distribute.available():
    print ("Distribute feature not built into iDeep,",
           "please use 'cmake -Dmultinode=ON ..' to build ideep")
    exit()

size = 99999999
shape = [size]
src_backup = numpy.zeros(shape, numpy.float32)

print ("Initialize distributed computation")
distribute.init()

world_size = distribute.get_world_size()
print ("world size = %d" % (world_size))

rank = distribute.get_rank()
print ("rank = %d" % (rank))

src_buf = (numpy.full(shape, rank, numpy.float32)
           + numpy.linspace(0.0, (shape[0]+0.0)/(shape[0]+1.0), num=shape[0],
                            endpoint=False, dtype=numpy.float32))

src_buf = ideep4py.mdarray(src_buf)
src_backup = ideep4py.mdarray(src_backup)
ideep4py.basic_copyto(src_backup, src_buf)

iter_num = 10
start = time.time()

# inplace
for i in range(iter_num):
    ideep4py.basic_copyto(src_buf, src_backup)
    distribute.allreduce(0, src_buf)
    distribute.barrier()

end = time.time()

avg_time = (end-start)/iter_num
eff_bw = 2.0*(world_size-1)/world_size * shape[0] * 32 / avg_time/1000000000
print ("[%d] Allreduce done in %f seconds, bw=%fGbps"
       % (rank, avg_time, eff_bw))
distribute.finalize()

if rank == 0:
    print ("Generate expected result...")

buf_expect = (numpy.full(shape, (world_size-1)*world_size/2.0)
              + numpy.linspace(0, shape[0]/(shape[0]+1.0)*world_size,
                               num=shape[0], endpoint=False))

if rank == 0:
    print ("[%d] Validate inplace result:" % (rank))

numpy.testing.assert_allclose(src_buf, buf_expect, rtol=1e-06)

if rank == 0:
    print ("pass!")
