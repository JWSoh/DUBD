import os
import tensorflow as tf
import test
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--gpu', type=str, dest='gpu', default='0')
parser.add_argument('--sigma',type=int, dest='sigma',default=10)
parser.add_argument('--inputpath',type=str, dest='inputpath',default='testset/CBSD68')
parser.add_argument('--outputpath',type=str, dest='outputpath',default='results/CBSD68')

args=parser.parse_args()

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']=args.gpu

conf=tf.ConfigProto()
conf.gpu_options.per_process_gpu_memory_fraction=0.9

def main():
    Test= test.Test(input_path=args.inputpath,
                    output_path=args.outputpath,
                    model_path='Model',
                    sigma=args.sigma,
                    conf=conf)
    Test()

if __name__=='__main__':
    main()