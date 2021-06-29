import time
import numpy as np


def run(point):
   start = time.time()
   try:
      batch_size = point['batch_size']
      image_size = point['image_size']
      conv1_in_chan  = point['conv1_in_chan']
      conv1_out_chan = point['conv1_out_chan']
      conv1_kern     = point['conv1_kern']
      pool_size      = point['pool_size']
      conv2_out_chan = point['conv2_out_chan']
      conv2_kern     = point['conv2_kern']
      fc1_out        = point['fc1_out']
      fc2_out        = point['fc2_out']
      fc3_out        = point['fc3_out']
      n_conv_block   = point['n_conv_block']

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

      class Net(torch.nn.Module):
         def __init__(self, batch_size,
                            image_size,
                            conv1_in_chan,conv1_out_chan,conv1_kern,
                            pool_size,
                            conv2_out_chan,conv2_kern,
                            fc1_out,
                            fc2_out,
                            fc3_out,
                            n_conv_block,
                            ):
            super(Net, self).__init__()

            self.flop = 0
            self.n_conv_block = n_conv_block
            self.batch_size = batch_size

            self.conv1 = torch.nn.Conv2d(conv1_in_chan, conv1_out_chan, conv1_kern)
            self.flop += conv1_kern**2 * conv1_in_chan * conv1_out_chan * image_size**2 * batch_size
            self.pool  = torch.nn.MaxPool2d(pool_size, pool_size)
            self.conv1_size = image_size-conv1_kern + 1 
            self.maxpool1_size = int((self.conv1_size - pool_size)/pool_size + 1)
            self.flop += image_size**2 * conv1_out_chan * batch_size

            self.conv2 = torch.nn.Conv2d(conv1_out_chan,conv2_out_chan,conv2_kern)
            self.flop += conv2_kern**2 * conv1_out_chan * conv2_out_chan * int(image_size/pool_size)**2 * batch_size
            self.conv2_size = self.maxpool1_size - conv2_kern + 1
            self.maxpool2_size = int((self.conv2_size - pool_size)/pool_size + 1 )

            self.view_size = conv2_out_chan * self.maxpool2_size * self.maxpool2_size
            self.fc1   = torch.nn.Linear(self.view_size, fc1_out)
            self.flop += (2*self.view_size - 1) * fc1_out * batch_size
            self.fc2   = torch.nn.Linear(fc1_out, fc2_out)
            self.flop += (2*fc1_out - 1) * fc2_out * batch_size
            self.fc3   = torch.nn.Linear(fc2_out, fc3_out)
            self.flop += (2*fc2_out - 1) * fc3_out * batch_size

         def forward(self, inputs):
            block_output = torch.zeros(inputs.shape[0],self.view_size,dtype=torch.float)
            for i in range(self.n_conv_block):
               batch = inputs[i * self.batch_size:(i + 1) * self.batch_size]

               x = self.pool(torch.nn.functional.relu(self.conv1(batch)))
               x = self.pool(torch.nn.functional.relu(self.conv2(x)))
               x = x.view(-1,self.view_size)
               block_output[i * self.batch_size:(i + 1) * self.batch_size] = x

            x = torch.nn.functional.relu(self.fc1(block_output))
            x = torch.nn.functional.relu(self.fc2(x))
            x = self.fc3(x)
            return x

      inputs = torch.arange(batch_size * n_conv_block * image_size**2 * conv1_in_chan,dtype=torch.float)
      inputs = inputs.view((batch_size * n_conv_block,conv1_in_chan,image_size,image_size))
      net = Net(batch_size,
                image_size,
                conv1_in_chan,conv1_out_chan,conv1_kern,
                pool_size,
                conv2_out_chan,conv2_kern,
                fc1_out,
                fc2_out,
                fc3_out,
                n_conv_block)
      outputs = net(inputs)

      total_flop = net.flop
      
      runs = 5
      tot_time = 0.
      tt = time.time()
      for _ in range(runs):
         outputs = net(inputs)
         tot_time += time.time() - tt
         tt = time.time()

      ave_time = tot_time / runs

      print('total_flop = ',total_flop,'ave_time = ',ave_time)

      ave_flops = total_flop / ave_time
      runtime = time.time() - start
      print('runtime=',runtime,'ave_flops=',ave_flops)

      return ave_flops
   except Exception as e:
      import traceback
      print('received exception: ',str(e),'for point: ',point)
      print(traceback.print_exc())
      print('runtime=',time.time() - start)
      return 0.


if __name__ == '__main__':
   point = {
      'batch_size': 10,
      'image_size': 32,
      'conv1_in_chan':3,
      'conv1_out_chan':6,
      'conv1_kern':5,
      'pool_size':2,
      'conv2_out_chan':16,
      'conv2_kern':5,
      'fc1_out':120,
      'fc2_out':84,
      'fc3_out': 10,
      'omp_num_threads':64,
      'n_conv_block': 6,
   }

   print('flops for this setting =',run(point))

