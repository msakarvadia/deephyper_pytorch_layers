from deephyper.benchmark import HpProblem

Problem = HpProblem()
Problem.add_dim('batch_size',(1,128))
Problem.add_dim('height',(128,512))
Problem.add_dim('width',(128,512))
Problem.add_dim('in_channels',(2,16))
Problem.add_dim('out_channels',(2,16))
Problem.add_dim('kernel_size',(2,8))

Problem.add_starting_point(batch_size=10,height=128,width=128,in_channels=3,out_channels=16,kernel_size=3)

if __name__ == '__main__':
   print(Problem)
