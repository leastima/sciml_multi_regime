#!/bin/bash

ntrain=32768
nval=4096
ntest=4096
ng=144
datapath='./data/advdiff'



# # create poissons examples
# # poissons diffusion eigenvalue range
# e1=1  
# e2=2.5
# CUDA_VISIBLE_DEVICES=4 python utils/gen_data_poisson.py --ntrain=$ntrain --nval=$nval --ntest=$ntest \
#                     --ng=$ng --sparse --n 128 --datapath $datapath --e1 $e1 --e2 $e2

# # poissons diffusion eigenvalue range
# e1=2.5 
# e2=5.0
# CUDA_VISIBLE_DEVICES=4 python utils/gen_data_poisson.py --ntrain=$ntrain --nval=$nval --ntest=$ntest \
#                     --ng=$ng --sparse --n 128 --datapath $datapath --e1 $e1 --e2 $e2

# poissons diffusion eigenvalue range
# e1=5.0 
# e2=10.0
# CUDA_VISIBLE_DEVICES=4 python utils/gen_data_poisson.py --ntrain=$ntrain --nval=$nval --ntest=$ntest \
#                     --ng=$ng --sparse --n 128 --datapath $datapath --e1 $e1 --e2 $e2

# # poissons diffusion eigenvalue range
# e1=10.0
# e2=20.0
# CUDA_VISIBLE_DEVICES=4 python utils/gen_data_poisson.py --ntrain=$ntrain --nval=$nval --ntest=$ntest \
#                     --ng=$ng --sparse --n 128 --datapath $datapath --e1 $e1 --e2 $e2

# # poissons diffusion eigenvalue range
# e1=30.0
# e2=50.0
# CUDA_VISIBLE_DEVICES=4 python utils/gen_data_poisson.py --ntrain=$ntrain --nval=$nval --ntest=$ntest \
#                     --ng=$ng --sparse --n 128 --datapath $datapath --e1 $e1 --e2 $e2

# # poissons diffusion eigenvalue range
# e1=100.0
# e2=200.0
# CUDA_VISIBLE_DEVICES=5 python utils/gen_data_poisson.py --ntrain=$ntrain --nval=$nval --ntest=$ntest \
#                     --ng=$ng --sparse --n 128 --datapath $datapath --e1 $e1 --e2 $e2

# # poissons diffusion eigenvalue range
# e1=20.0
# e2=30.0
# CUDA_VISIBLE_DEVICES=4 python utils/gen_data_poisson.py --ntrain=$ntrain --nval=$nval --ntest=$ntest \
#                     --ng=$ng --sparse --n 128 --datapath $datapath --e1 $e1 --e2 $e2


################################################
# advection to diffusion ratio range
adr1=0.01 
adr2=0.2
CUDA_VISIBLE_DEVICES=0 python utils/gen_data_advdiff.py --ntrain=$ntrain --nval=$nval --ntest=$ntest \
                   --ng=$ng --sparse --n 128 --datapath $datapath --adr1 $adr1 --adr2 $adr2


# # advection to diffusion ratio range
# adr1=0.2 
# adr2=0.4
# CUDA_VISIBLE_DEVICES=4 python utils/gen_data_advdiff.py --ntrain=$ntrain --nval=$nval --ntest=$ntest \
#                    --ng=$ng --sparse --n 128 --datapath $datapath --adr1 $adr1 --adr2 $adr2


# advection to diffusion ratio range
adr1=0.4 
adr2=0.6
CUDA_VISIBLE_DEVICES=0 python utils/gen_data_advdiff.py --ntrain=$ntrain --nval=$nval --ntest=$ntest \
                   --ng=$ng --sparse --n 128 --datapath $datapath --adr1 $adr1 --adr2 $adr2


# # advection to diffusion ratio range
# adr1=0.6 
# adr2=0.8
# CUDA_VISIBLE_DEVICES=4 python utils/gen_data_advdiff.py --ntrain=$ntrain --nval=$nval --ntest=$ntest \
#                    --ng=$ng --sparse --n 128 --datapath $datapath --adr1 $adr1 --adr2 $adr2


# advection to diffusion ratio range
adr1=0.8 
adr2=1.0
CUDA_VISIBLE_DEVICES=0 python utils/gen_data_advdiff.py --ntrain=$ntrain --nval=$nval --ntest=$ntest \
                   --ng=$ng --sparse --n 128 --datapath $datapath --adr1 $adr1 --adr2 $adr2


# # advection to diffusion ratio range
# adr1=1.0 
# adr2=1.3
# CUDA_VISIBLE_DEVICES=4 python utils/gen_data_advdiff.py --ntrain=$ntrain --nval=$nval --ntest=$ntest \
#                    --ng=$ng --sparse --n 128 --datapath $datapath --adr1 $adr1 --adr2 $adr2


# advection to diffusion ratio range
adr1=1.3 
adr2=1.7
CUDA_VISIBLE_DEVICES=0 python utils/gen_data_advdiff.py --ntrain=$ntrain --nval=$nval --ntest=$ntest \
                   --ng=$ng --sparse --n 128 --datapath $datapath --adr1 $adr1 --adr2 $adr2


# # advection to diffusion ratio range
# adr1=1.7 
# adr2=2.0
# CUDA_VISIBLE_DEVICES=4 python utils/gen_data_advdiff.py --ntrain=$ntrain --nval=$nval --ntest=$ntest \
#                    --ng=$ng --sparse --n 128 --datapath $datapath --adr1 $adr1 --adr2 $adr2


# advection to diffusion ratio range
adr1=2.0 
adr2=2.5
CUDA_VISIBLE_DEVICES=0 python utils/gen_data_advdiff.py --ntrain=$ntrain --nval=$nval --ntest=$ntest \
                   --ng=$ng --sparse --n 128 --datapath $datapath --adr1 $adr1 --adr2 $adr2


# advection to diffusion ratio range
# adr1=2.5 
# adr2=3.0
# CUDA_VISIBLE_DEVICES=4 python utils/gen_data_advdiff.py --ntrain=$ntrain --nval=$nval --ntest=$ntest \
#                    --ng=$ng --sparse --n 128 --datapath $datapath --adr1 $adr1 --adr2 $adr2








# for AD ratio we saved a set of velocity scales that correspond to AD ration in utils/*.npy. See python script for details

# o1=1 # helmholtz wave number range
# o2=10

# create AD examples
#python utils/gen_data_advdiff.py --ntrain=$ntrain --nval=$nval --ntest=$ntest \
#                    --ng=$ng --sparse --n 128 --datapath $datapath --adr1 $adr1 --adr2 $adr2

# create Helm examples
#python utils/gen_data_helmholtz.py --ntrain=$ntrain --nval=$nval --ntest=$ntest \
#                    --ng=$ng --sparse --n 128 --datapath $datapath --o1 $o1 --o2 $o2

