import time,psutil,os
import multiprocessing as mp

def print_mem_cpu():
   start = time.time()
   while True:
      mem = psutil.virtual_memory()
      print('[%010d] pid=%010d total_mem=%010d free_mem=%05.2f cpu_usage=%05.2f' % (time.time()-start,os.getpid(),mem.total,mem.free/mem.total*100.,psutil.cpu_percent()))
      time.sleep(1)



def run(point):
   print(point)
   start = time.time()
   memorymon = mp.Process(target=print_mem_cpu)
   memorymon.start()
   try:
      batch_size = point['batch_size']
      image_size = point['image_size']
      in_channels = point['in_channels']
      out_channels = point['out_channels']
      kernel_size = point['kernel_size']
      omp_num_threads = point['omp_num_threads']

      import os
      os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)
      os.environ['MKL_NUM_THREADS'] = str(omp_num_threads)
      os.environ['KMP_HW_SUBSET'] = '1s,%sc,2t' % str(omp_num_threads)
      os.environ['KMP_AFFINITY'] = 'granularity=fine,verbose,compact,1,0'
      os.environ['KMP_BLOCKTIME'] = str(0)
      #os.environ['MKLDNN_VERBOSE'] = str(1)
      import torch

      print('torch version: ',torch.__version__,' torch file: ',torch.__file__)

      
      with torch.no_grad():
         inputs = torch.arange(batch_size * image_size**3 * in_channels,dtype=torch.float).view((batch_size,in_channels,image_size,image_size,image_size))
         print('creating layer')
         layer = torch.nn.Conv3d(in_channels,out_channels,kernel_size,stride=1,padding=1)
         layer.eval()
         print('first execution')
         outputs = layer(inputs)


         total_flop = kernel_size**3 * in_channels * out_channels * outputs.shape[-1] * outputs.shape[-2] * outputs.shape[-3] * batch_size
         
         runs = 25
         tot_time = 0.
         tt = time.time()
         print('loop')
         for i in range(runs):
            print('step',i)
            outputs = layer(inputs)
            tot_time += time.time() - tt
            tt = time.time()

         ave_time = tot_time / runs

         print('total_flop = ',total_flop,'ave_time = ',ave_time)

         ave_flops = total_flop / ave_time
         runtime = time.time() - start
         print('runtime=',runtime,'ave_flops=',ave_flops)
      memorymon.terminate()
      memorymon.join()
      return ave_flops
   except Exception as e:
      import traceback
      print('received exception: ',str(e),'for point: ',point)
      print(traceback.print_exc())
      print('runtime=',time.time() - start)
      memorymon.terminate()
      memorymon.join()
      
      return 0.


if __name__ == '__main__':
   point = {
      'batch_size': 10,
      'image_size': 64,
      'in_channels': 3,
      'out_channels': 3,
      'kernel_size': 3,
      'omp_num_threads':64,
   }

   #point = {'batch_size': 4, 'image_size': 88, 'in_channels': 56, 'kernel_size': 10, 'omp_num_threads': 64, 'out_channels': 47}

   print('flops for this setting =',run(point))

