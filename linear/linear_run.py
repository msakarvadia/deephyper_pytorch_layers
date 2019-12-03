import time
from ptflops import get_model_complexity_info

def run(point):
   start = time.time()
   try:
      batch_size = point['batch_size']
      in_features = point['in_features']
      out_features = point['out_features']
      bias = int(point['bias']) == 1
      omp_num_threads = point['omp_num_threads']
      print(point)

      import os
      os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)
      os.environ['MKL_NUM_THREADS'] = str(omp_num_threads)
      os.environ['KMP_HW_SUBSET'] = '1s,%sc,2t' % str(omp_num_threads)
      os.environ['KMP_AFFINITY'] = 'granularity=fine,verbose,compact,1,0'
      os.environ['KMP_BLOCKTIME'] = str(0)
      os.environ['MKLDNN_VERBOSE'] = str(1)
      os.environ['MKL_VERBOSE'] = str(1)
      import torch

      print('torch version: ',torch.__version__,' torch file: ',torch.__file__)

      inputs = torch.arange(batch_size * in_features,dtype=torch.float).view((batch_size,in_features))
      
      layer = torch.nn.Linear(in_features,out_features,bias=bias)
      flops, params = get_model_complexity_info(layer, tuple(inputs.shape),as_strings=False)
      print(flops)

      outputs = layer(inputs)
      
      runs = 5
      tot_time = 0.
      tt = time.time()
      for _ in range(runs):
         outputs = layer(inputs)
         tot_time += time.time() - tt
         tt = time.time()

      ave_time = tot_time / runs

      print('flop = ',flops,'ave_time = ',ave_time)

      ave_flops = flops / ave_time  #* batch_size
      
      print('runtime=',time.time() - start)
      return ave_flops
   except Exception as e:
      import traceback
      print('received exception: ',str(e))
      print(traceback.print_exc())
      print('runtime=',time.time() - start)
      return 0.


if __name__ == '__main__':
   point = {
      'batch_size': 10,
      'in_features': 512,
      'out_features': 512,
      'bias': 1,
      'omp_num_threads':64,
   }

   print('flops for this setting =',run(point))

