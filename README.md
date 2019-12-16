# Fast Approximate Spectral Norm Regularization for Enhancing Robustness of DNNs

This is a pytorch implemention of the fast spectral norm regularization algorithm proposed in the paper Fast Approximate Spectral Norm Regularization for Enhancing Robustness of DNNs.

## Usage

The 'GPU_version.py' will compare our fast spectral norm regularization algorithm with the [newest algorithm](https://arxiv.org/abs/1705.10941). You can modify row 302 to switch loss among none regularizer (```loss = loss```), our regularizer (```loss = loss + loss_my_conv```) and newest regularizer (```loss = loss + loss_old_conv```).

## Test Result
![](https://github.com/Flocculus/Fast-Approximate-Spectral-Norm-Regularization-for-Enhancing-Robustness-of-DNNs/blob/master/Fig/F1.png)

![](https://github.com/Flocculus/Fast-Approximate-Spectral-Norm-Regularization-for-Enhancing-Robustness-of-DNNs/blob/master/Fig/F2.png)

![](https://github.com/Flocculus/Fast-Approximate-Spectral-Norm-Regularization-for-Enhancing-Robustness-of-DNNs/blob/master/Fig/F3.png)
