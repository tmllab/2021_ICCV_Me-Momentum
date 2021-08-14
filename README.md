# Me-Momentum: Extracting Hard Confident Examples from Noisily Labeled Data

PyTorch Code for the following paper at ICCV2021:\
<b>Title</b>: <i>Me-Momentum: Extracting Hard Confident Examples from Noisily Labeled Data</i> \
<b>Authors</b>: Yingbin Bai, Tongliang Liu \
<b>Institute</b>: University of Sydney


## Abstract
Examples that are close to the decision boundary—that we term hard examples, are essential to shape accurate classifiers. Extracting confident examples has been widely studied in the community of learning with noisy labels. However, it remains elusive how to extract hard confident examples from the noisy training data. In this paper, we propose a deep learning paradigm to solve this problem, which is built on the memorization effect of deep neural networks that they would first learn simple patterns, i.e., which are defined by these shared by multiple training examples. To extract hard confident examples that contain non-simple patterns and are entangled with the inaccurately labeled examples, we borrow the idea of momentum from physics. Specifically, we alternately update the confident examples and refine the classifier. Note that the extracted confident examples in the previous round can be exploited to learn a better classifier and that the better classifier will help identify better (and hard) confident examples. We call the approach the “Momentum of Memorization” (Me-Momentum). Empirical results on benchmark-simulated and real-world label-noise data illustrate the effectiveness of Me-Momentum for extracting hard confident examples, leading to better classification performance.


## Experiments

To install requirements:

```setup
pip install -r requirements.txt
```

> 📋 Please download and place all datasets into the data directory. For Clohting1M, please run "python Clothing1m-data.npy" to generate a data file.


To run program on MNIST and CIFAR-10/100

```run
python main.py --dataset mnist --noise_type instance --noise_rate 0.2

python main.py --dataset cifar10 --noise_type symmetric --noise_rate 0.2

python main.py --dataset cifar100 --noise_type symmetric --noise_rate 0.4
```

To run program on Clothing1M

```run
python3 Clothing.py
```


## Cite Me-Momentum
If you find the code useful in your research, please consider citing our paper:

<pre>
@inproceedings{
    bai2020me_momentum,
    title={Me-Momentum: Extracting Hard Confident Examples from Noisily Labeled Data},
    author={Yingbin Bai and Tongliang Liu},
    booktitle={ICCV},
    year={2021},
}
</pre>
