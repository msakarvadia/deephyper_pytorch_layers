from deephyper.benchmark import HpProblem

Problem = HpProblem()
Problem.add_dim('batch_size',(1,8192))
Problem.add_dim('in_features',(128,8192))
Problem.add_dim('out_features',(128,8192))
# Problem.add_dim('omp_num_threads',(8,64))
Problem.add_dim('bias',[0,1])

Problem.add_starting_point(batch_size=128,in_features=1024,
                           out_features=512,
                           # omp_num_threads=64,
                           bias=0)

if __name__ == '__main__':
   print(Problem)
