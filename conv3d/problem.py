from deephyper.benchmark import HpProblem

Problem = HpProblem()
Problem.add_dim('batch_size',(1,32))
Problem.add_dim('image_size',(16,128))
Problem.add_dim('in_channels',(2,64))
Problem.add_dim('out_channels',(2,64))
Problem.add_dim('kernel_size',(2,10))
# Problem.add_dim('omp_num_threads',[64])

Problem.add_starting_point(batch_size=10,image_size=28,in_channels=2,out_channels=2,kernel_size=2,
                           ) # omp_num_threads=64)

if __name__ == '__main__':
   print(Problem)
