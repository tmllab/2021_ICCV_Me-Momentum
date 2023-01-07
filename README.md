<div align="center">   
  
# Me-Momentum: Extracting Hard Confident Examples from Noisily Labeled Data
[![Paper](https://img.shields.io/badge/paper-ICCV-green)](https://openaccess.thecvf.com/content/ICCV2021/papers/Bai_Me-Momentum_Extracting_Hard_Confident_Examples_From_Noisily_Labeled_Data_ICCV_2021_paper.pdf)

</div>

Official implementation of [Me-Momentum: Extracting Hard Confident Examples from Noisily Labeled Data](https://openaccess.thecvf.com/content/ICCV2021/papers/Bai_Me-Momentum_Extracting_Hard_Confident_Examples_From_Noisily_Labeled_Data_ICCV_2021_paper.pdf) (ICCV2021 Oral).

Examples that are close to the decision boundary—that we term hard examples, are essential to shape accurate classifiers. Extracting confident examples has been widely studied in the community of learning with noisy labels. However, it remains elusive how to extract hard confident examples from the noisy training data. In this paper, we propose a deep learning paradigm to solve this problem, which is built on the memorization effect of deep neural networks that they would first learn simple patterns, i.e., which are defined by these shared by multiple training examples. To extract hard confident examples that contain non-simple patterns and are entangled with the inaccurately labeled examples, we borrow the idea of momentum from physics. Specifically, we alternately update the confident examples and refine the classifier. Note that the extracted confident examples in the previous round can be exploited to learn a better classifier and that the better classifier will help identify better (and hard) confident examples. We call the approach the “Momentum of Memorization” (Me-Momentum). Empirical results on benchmark-simulated and real-world label-noise data illustrate the effectiveness of Me-Momentum for extracting hard confident examples, leading to better classification performance.

<p float="left" align="center">
<img src="key idea.png" width="800" /> 
<figcaption align="center">
The illustration of the influence of hard (confident) examples in classification. Circles represent positive examples while triangles
represent negative examples. Green and blue denote examples with accurate labels while red presents examples with incorrect labels.
Blank circles and triangles represent unextracted data. (a) shows an example of classification with clean data. (b) shows noisy examples,
especially those close to the decision boundary, will significantly degenerate the accuracy of the classifier. (c) shows confident examples
help learn a fairly good classifier. (d) shows that hard confident examples are essential to train an accurate classifier.
</figcaption>
</p>



## Requirements
- This codebase is written for `python3` and 'pytorch'.
- To install necessary python packages, run `pip install -r requirements.txt`.



## Training
### Data
- Please download and place all datasets into the data directory. 
- For Clohting1M, please run "python Clothing1m-data.npy" to generate a data file.

### Training

To run program on MNIST and CIFAR-10/100

```run
python main.py --dataset mnist --noise_type instance --noise_rate 0.2

python main.py --dataset cifar10 --noise_type symmetric --noise_rate 0.2

python main.py --dataset cifar100 --noise_type symmetric --noise_rate 0.4
```

To run program on Clothing1M

```run
python Clothing.py
```

## License and Contributing
- This README is formatted based on [paperswithcode](https://github.com/paperswithcode/releasing-research-code).
- Feel free to post issues via Github. 

## Reference
If you find the code useful in your research, please consider citing our paper:

<pre>
@inproceedings{
    bai2021memomentum,
    title={Me-Momentum: Extracting Hard Confident Examples from Noisily Labeled Data},
    author={Yingbin Bai and Tongliang Liu},
    booktitle={ICCV},
    year={2021},
}
</pre>


## Contact
Please contact ybai6430@uni.sydney.edu.au if you have any question on the codes.
