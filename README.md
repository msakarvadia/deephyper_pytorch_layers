# deephyper_pytorch_layers
Use DeepHyper to find the max FLOPS for various PyTorch Layers.

## TODO:

- [ ] Benchmark DeepHyper+Balsam throughput on various platforms.
Taylor reports low utilization on Theta.
- [ ] Compute and/or measure memory usage on GPU and KNL


### Bugs and run failures

Run failures in `conv2d/`, `conv1d/`, `conv3d/`; surprisingly, `conv3d/` has fewer
`objective=0.0` evaluations than the other two.

-  `conv2d/` on Traverse: several instances of the following error:
```
Traceback (most recent call last):
  File "/home/kfelker/deephyper_pytorch_layers/conv2d/conv2d_run.py", line 36, in run
    outputs = layer(inputs)
  File "/home/kfelker/.conda/envs/frnn/lib/python3.6/site-packages/torch/nn/modules/module.py", line 547, in __call__
    result = self.forward(*input, **kwargs)
  File "/home/kfelker/.conda/envs/frnn/lib/python3.6/site-packages/torch/nn/modules/conv.py", line 343, in forward
    return self.conv2d_forward(input, self.weight)
  File "/home/kfelker/.conda/envs/frnn/lib/python3.6/site-packages/torch/nn/modules/conv.py", line 340, in conv2d_forward
    self.padding, self.dilation, self.groups)
RuntimeError: cuDNN error: CUDNN_STATUS_NOT_SUPPORTED. This error may appear if you passed in a non-contiguous input.
torch version:  1.2.0  torch file:  /home/kfelker/.conda/envs/frnn/lib/python3.6/site-packages/torch/__init__.py
received exception:  cuDNN error: CUDNN_STATUS_NOT_SUPPORTED. This error may appear if you passed in a non-contiguous input.
None
runtime= 8.049392938613892
DH-OUTPUT: 0.0
```
- Compare frequency to errors due to memory exhaustion:
```
CUDA out of memory. Tried to allocate 22.79 GiB (GPU 0; 31.75 GiB total capacity; 28.66 GiB already allocated; 1.93 GiB free; 21.30 MiB cached; 0 bytes inactive)
```

E.g. out of 101 evaluations of `conv2d/` on Traverse, total of 15x run failures:

```
(base) ➜  conv2d git:(feature/cuda) ✗ sort -t , -k 7,7 -n results.csv
batch_size,height,in_channels,kernel_size,out_channels,width,objective,elapsed_sec
128,512,3,3,64,512,0.0,19.287363052368164
146,985,64,12,58,329,0.0,1273.2861964702606
276,422,64,5,57,330,0.0,1067.2843222618103
283,655,64,14,64,976,0.0,1077.281128168106
290,844,54,11,7,176,0.0,447.2913315296173
331,193,52,12,6,777,0.0,181.2829658985138
377,559,57,10,23,224,0.0,281.2877972126007
383,458,64,7,7,215,0.0,241.2843005657196
403,184,55,7,8,646,0.0,347.2854845523834
405,887,64,10,58,135,0.0,997.284960269928
413,683,61,12,6,133,0.0,211.2883608341217
476,139,64,13,40,1021,0.0,1269.2893743515015
488,129,63,7,63,791,0.0,815.2774884700775
506,135,62,13,55,995,0.0,977.2954721450806
510,184,61,11,56,1021,0.0,307.28731894493103
```



### More PyTorch layers, etc.
- [ ] `BatchNorm1d`
- [ ] `BatchNorm2d`
- [ ] `BatchNorm3d`
- [ ] `Softmax`
- [ ] `Tanh`
- [ ] `Sigmoid`
- [ ] `ReLU`
- [ ] Pooling?
- [ ] Embedding?
- [ ] `RNN`
- [ ] `LSTM`
- [ ] `GRU`
- [ ] `Transformer`, `TransformerEncoder`, `TransformerDecoder`

See https://pytorch.org/docs/stable/nn.html


## Balsam + DeepHyper performance
All table entries were measured

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-left" />

<col  class="org-right" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Machine</th>
<th scope="col" class="org-left">Balsam job node</th>
<th scope="col" class="org-right">Nodes</th>
<th scope="col" class="org-right">DH num<sub>workers</sub></th>
<th scope="col" class="org-right">Time limit (min)</th>
<th scope="col" class="org-left">PyTorch Layer</th>
<th scope="col" class="org-right">Evaluations</th>
</tr>
</thead>

<tbody>
<tr>
<td class="org-left">ALCF Theta</td>
<td class="org-left">MPI</td>
<td class="org-right">8</td>
<td class="org-right">8</td>
<td class="org-right">60</td>
<td class="org-left">Linear</td>
<td class="org-right">1190</td>
</tr>
</tbody>

<tbody>
<tr>
<td class="org-left">Princeton Traverse</td>
<td class="org-left">Serial</td>
<td class="org-right">1</td>
<td class="org-right">5</td>
<td class="org-right">60</td>
<td class="org-left">Linear</td>
<td class="org-right">186</td>
</tr>
</tbody>

<tbody>
<tr>
<td class="org-left">Princeton Traverse</td>
<td class="org-left">MPI</td>
<td class="org-right">2</td>
<td class="org-right">10</td>
<td class="org-right">60</td>
<td class="org-left">Linear</td>
<td class="org-right">&#xa0;</td>
</tr>
</tbody>

<tbody>
<tr>
<td class="org-left">Princeton TigerGPU</td>
<td class="org-left">Serial</td>
<td class="org-right">1</td>
<td class="org-right">5</td>
<td class="org-right">120</td>
<td class="org-left">Linear</td>
<td class="org-right">246</td>
</tr>
</tbody>

<tbody>
<tr>
<td class="org-left">Princeton TigerGPU</td>
<td class="org-left">Serial</td>
<td class="org-right">3</td>
<td class="org-right">10</td>
<td class="org-right">120</td>
<td class="org-left">Linear</td>
<td class="org-right">&#xa0;</td>
</tr>
</tbody>
</table>
