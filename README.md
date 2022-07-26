# Learning Extremely Lightweight and Robust Model with Differentiable Constraints on Sparsity and Condition Number
The code used in the evaluation of "Learning Extremely Lightweight and Robust Model with Differentiable Constraints on Sparsity and Condition Number" (ECCV 2022).

This code is written based on the original HYDRA (https://github.com/inspire-group/hydra)

Learning lightweight and robust deep learning models is an enormous challenge for safety-critical devices with limited computing and memory resources, owing to robustness against adversarial attacks being proportional to network capacity. The community has extensively explored the integration of adversarial training and model compression, such as weight pruning. However, lightweight models generated by highly pruned over-parameterized models lead to sharp drops in both robust and natural accuracy. It has been observed that the parameters of these models lie in ill-conditioned weight space, i.e., the condition number of weight matrices tend to be large enough that the model is not robust. In this work, we propose a framework for building extremely lightweight models, which combines tensor product with the differentiable constraints for reducing condition number and promoting sparsity. Moreover, the proposed framework is incorporated into adversarial training with the min-max optimization scheme. We evaluate the proposed approach on VGG-16 and Visual Transformer. Experimental results on datasets such as ImageNet, SVHN, and CIFAR-10 show that we can achieve an overwhelming advantage at a high compression ratio, e.g., 200 times.
## Citations
```
@InProceedings{wei_arlst2022,
	title={Learning Extremely Lightweight and Robust Model with Differentiable Constraints on Sparsity and Condition Number},
	author={Wei, Xian and Xu, Yangyu and Huang, Yanhui and Lv, Hairong and Chen, Mingsong and Lan, Hai and Tang, Xuan},
	booktitle={European Conference on Computer Vision (ECCV)},
	pages={},
	year={2022}
    }
```
