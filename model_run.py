import torch,time
from ptflops import get_model_complexity_info


def run(point):

   batch_size = point['batch_size']
   height = point['height']
   width = point['width']
   in_channels = point['in_channels']
   out_channels = point['out_channels']
   kernel_size = (point['kernel_size'],point['kernel_size'])


   inputs = torch.arange(batch_size * height * width * in_channels,dtype=torch.float).view((batch_size,in_channels,height,width))
   
   layer = torch.nn.Conv2d(in_channels,out_channels,kernel_size,stride=1)
   flops, params = get_model_complexity_info(layer, tuple(inputs.shape[1:]),as_strings=False)
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

   ave_flops = flops / ave_time

   return ave_flops


if __name__ == '__main__':
   point = {
      'batch_size': 10,
      'height': 512,
      'width': 512,
      'in_channels': 3,
      'out_channels': 64,
      'kernel_size': 4
   }

   print('flops for this setting =',run(point))

