import numpy as np


def load_data(size=2000,height=256,width=256,channels=3):

   inputs = np.random.randn((size,height,width,channels))
   
   return inputs
   
