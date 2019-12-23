# Fast Approximate Spectral Norm Regularization for Enhancing Robustness of DNNs

This is a pytorch implemention of the fast spectral norm regularization algorithm proposed in the paper Fast Approximate Spectral Norm Regularization for Enhancing Robustness of DNNs.

## Usage

The 'GPU_version.py' will compare our fast spectral norm regularization algorithm with the [newest algorithm](https://arxiv.org/abs/1705.10941). You can modify row 302 to switch loss among none regularizer (```loss = loss```), our regularizer (```loss = loss + loss_my_conv```) and newest regularizer (```loss = loss + loss_old_conv```).

## Dataset
Please get the data from [here](https://1drv.ms/u/s!Aqx-iMSK3x4dmq4oWXpoUTrV1Wbm6Q?e=h8JeX1).

## Test Result
Here are some of the output.
![](https://github.com/Flocculus/DCGAN_CATFACE/blob/master/someoutput/1.png)

![](https://github.com/Flocculus/DCGAN_CATFACE/blob/master/someoutput/9.png)

![](https://github.com/Flocculus/DCGAN_CATFACE/blob/master/someoutput/15.png)
