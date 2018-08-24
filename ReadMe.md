# Reading_list
Reading list on deep learning.  Ph.d life

***
## optic disk and optic cup segmentation
* [Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114.pdf)
### segmentation
* a survey 2017 almost traditional methods [Segmentation Techniques for Computer-Aided Diagnosis of Glaucoma: A Review](https://link.springer.com/content/pdf/10.1007\%2F978-981-10-8569-7_18.pdf) :star::star:

* based on structural and gray level properties [Segmentation of optic disk and optic cup from digital fundus images for the assessment of glaucoma](https://ac.els-cdn.com/S1746809415001512/1-s2.0-S1746809415001512-main.pdf?_tid=28900699-9a58-4929-8c0c-6268f84bdaf4&acdnat=1529743095_1f81babc1977bd510cafed242e7f2ac3) no deep learning but very useful ideas. :star::star::star::star:

* convolutional filter + entropy sampling + convex hull transformation [Glaucoma detection using entropy sampling and ensemble learning for automatic optic cup and disc segmentation
](https://ac.els-cdn.com/S0895611116300775/1-s2.0-S0895611116300775-main.pdf?_tid=6678eb0e-7fcd-4a4e-89f6-fe8cad5730f4&acdnat=1529756396_01824474b64b63bab5be415a9f933f00) too complicated :star::star::star:

* semi-supervised [Semi-supervised Segmentation of Optic Cup in Retinal Fundus Images Using Variational Autoencoder](https://link.springer.com/content/pdf/10.1007\%2F978-3-319-66185-8_9.pdf) MICCAI 2017 :star::star::star::star:

* [Joint Optic Disc and Cup Segmentation Based on Multi-label Deep Network and Polar Transformation](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8252743) 2018 TMI :star::star::star::star::star:
* [Deep Retinal Image Understanding](https://arxiv.org/pdf/1609.01103.pdf) 2016 MICCAI :star::star::star::star::star:



### domain adaptation
* [Adversarial Discriminative Domain Adaptation](http://openaccess.thecvf.com/content_cvpr_2017/papers/Tzeng_Adversarial_Discriminative_Domain_CVPR_2017_paper.pdf) CVPR 2017 :star::star::star::star:
* [Adversarial Feature Augmentation for Unsupervised Domain Adaptation](https://arxiv.org/abs/1711.08561) CVPR 2018. :star::star::star::star::star:
* [Unsupervised Pixel?Level Domain Adaptation with Generative Adversarial Networks](http://openaccess.thecvf.com/content_cvpr_2017/papers/Bousmalis_Unsupervised_Pixel-Level_Domain_CVPR_2017_paper.pdf) CVPR 2017  :star::star::star::star:
* [Boosting Domain Adaptation by Discovering Latent Domains](https://arxiv.org/abs/1805.01386) CVPR 2018.oral 
* [Learning to Adapt Structured Output Space for Semantic Segmentation]().  CVPR 2018.   GTA IoU: 42.4\%    SYNTHIA IoU: 46.7\%
* [Conditional Generative Adversarial Network for Structured Domain Adaptation]() CVPR 2018.   GTA IoU 44.5\%.    SYNTHIA IoU: 41.2\%
* [Unsupervised Domain Adaptation with Similarity Learning]() for classification

[Domain Adaptation for Segmentation]

*[CycleGAN Multimodal, shape constraints ](https://arxiv.org/pdf/1802.09655.pdf)  CVPR2018  :star::star::star::star::star:


### patch-level reinforesment learning


***
## MIML (multi-instance multi label learning)
* **Deep MIML** Feng, Ji, and Zhi-Hua Zhou. "Deep MIML Network." AAAI. 2017.

### MIL
* **[survey](https://www.sciencedirect.com/science/article/pii/S0031320317304065)** Carbonneau, Marc-André, et al. "Multiple instance learning: A survey of problem characteristics and applications." Pattern Recognition (2017).  [another link here](https://arxiv.org/pdf/1612.03365.pdf) :star::star::star::star::star:
  
#### Instance Space Method
* **[EM_DD](http://papers.nips.cc/paper/1959-em-dd-an-improved-multiple-instance-learning-technique.pdf)** Zhang, Qi, and Sally A. Goldman. "EM-DD: An improved multiple-instance learning technique." Advances in neural information processing systems. 2002. [Implement](http://lamda.nju.edu.cn/code_MIL-Ensemble.ashx)
:star::star::star::star:
* **[MI_SVM](http://papers.nips.cc/paper/2232-support-vector-machines-for-multiple-instance-learning.pdf)** Andrews, Stuart, Ioannis Tsochantaridis, and Thomas Hofmann. "Support vector machines for multiple-instance learning." Advances in neural information processing systems. 2003. :star::star::star::star:
* **[MILBoost](http://papers.nips.cc/paper/2926-multiple-instance-boosting-for-object-detection.pdf)** Zhang, Cha, John C. Platt, and Paul A. Viola. "Multiple instance boosting for object detection." Advances in neural information processing systems. 2006. :star::star::star::star:
  
#### Bag Space Method
* **[Diverse Density(DD)](http://lis.csail.mit.edu/pubs/tlp/maron98framework.pdf)** Maron, Oded, and Tomás Lozano-Pérez. "A framework for multiple-instance learning." Advances in neural information processing systems. 1998. [Implement](http://lamda.nju.edu.cn/code_MIL-Ensemble.ashx) :star::star::star::star: 
* **[citation kNN](http://cogprints.org/2124/3/wang_ICML2000.pdf)** Wang, Jun, and Jean-Daniel Zucker. "Solving multiple-instance problem: A lazy learning approach." (2000): 1119-1125. [Implement](http://lamda.nju.edu.cn/code_MIL-Ensemble.ashx) 
* **[MInd](https://www.sciencedirect.com/science/article/pii/S0031320314002817)** Cheplygina, Veronika, David MJ Tax, and Marco Loog. "Multiple instance learning with bag dissimilarities." Pattern Recognition 48.1 (2015): 264-275.
* **[CCE](https://link.springer.com/content/pdf/10.1007%2Fs10115-006-0029-3.pdf)** Zhou, Zhi-Hua, and Min-Ling Zhang. "Solving multi-instance problems with classifier ensemble based on constructive clustering." Knowledge and Information Systems 11.2 (2007): 155-170. [implement](http://lamda.nju.edu.cn/code_CCE.ashx)

* **[MILES](http://ieeexplore.ieee.org/abstract/document/1717454/)** Chen, Yixin, Jinbo Bi, and James Ze Wang. "MILES: Multiple-instance learning via embedded instance selection." IEEE Transactions on Pattern Analysis and Machine Intelligence 28.12 (2006): 1931-1947.
* **[NSK-SVM](http://sci2s.ugr.es/keel/pdf/algorithm/congreso/2002-Gartner-ICML.pdf)** Gärtner, Thomas, et al. "Multi-instance kernels." ICML. Vol. 2. 2002.
* **[mi-Graph](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.149.6808&rep=rep1&type=pdf)** Zhou, Zhi-Hua, Yu-Yin Sun, and Yu-Feng Li. "Multi-instance learning by treating instances as non-iid samples." Proceedings of the 26th annual international conference on machine learning. ACM, 2009. [implement](http://lamda.nju.edu.cn/code_miGraph.ashx)
* **[BoW-SVM]()**
* **[EMD-SVM](https://link.springer.com/content/pdf/10.1023%2FA%3A1026543900054.pdf)** Rubner, Yossi, Carlo Tomasi, and Leonidas J. Guibas. "The earth mover's distance as a metric for image retrieval." International journal of computer vision 40.2 (2000): 99-121.

#### Ranking
* **** Fast bundle algorithm for multiple-instance learning
* **** Multiple-instance ranking: Learning to rank images for image retrieval

#### others
* **[MIL pooling layer](https://academic.oup.com/bioinformatics/article/32/12/i52/2288769)** Kraus, Oren Z., Jimmy Lei Ba, and Brendan J. Frey. "Classifying and segmenting microscopy images with deep multiple instance learning." Bioinformatics 32.12 (2016): i52-i59.   :star::star::star::star:

* **[ multi-instane neural network](https://lirias.kuleuven.be/bitstream/123456789/133224/1/31670.pdf)** Ramon, Jan, and Luc De Raedt. "Multi instance neural networks." Proceedings of the ICML-2000 workshop on attribute-value and relational learning. 2000. :star::star::star:



* **[ML-KNN](https://ac.els-cdn.com/S0031320307000027/1-s2.0-S0031320307000027-main.pdf?_tid=4aca996e-88bb-4a31-8efc-b590364adbd2&acdnat=1521359388_bb21b8697481230d67ebf257245dad8a)** Zhang, Min-Ling, and Zhi-Hua Zhou. "ML-KNN: A lazy learning approach to multi-label learning." Pattern recognition 40.7 (2007): 2038-2048.

#### MIL in Deep Learning
* **[Multi-Instance Deep Learning: Discover Discriminative Local Anatomies for Bodypart Recognition](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7398101)** Yan, Zhennan, et al. "Multi-instance deep learning: Discover discriminative local anatomies for bodypart recognition." IEEE transactions on medical imaging 35.5 (2016): 1332-1343.
* **[MILCNN](https://arxiv.org/pdf/1610.03155.pdf)**  Sun, Miao, et al. "Multiple instance learning convolutional neural networks for object recognition." Pattern Recognition (ICPR), 2016 23rd International Conference on. IEEE, 2016.
* **[Attention Deep MIL](https://arxiv.org/pdf/1802.04712.pdf)** Ilse, Maximilian, Jakub M. Tomczak, and Max Welling. "Attention-based Deep Multiple Instance Learning." arXiv preprint arXiv:1802.04712 (2018). :star::star::star::star::star::star:
* **[MINN](https://ac.els-cdn.com/S0031320317303382/1-s2.0-S0031320317303382-main.pdf?_tid=a6cd7eba-7151-4cf6-9bae-7920c3e0ac75&acdnat=1521535813_58887e0c10d507eeab20ecc7e9b012e5)** Wang, Xinggang, et al. "Revisiting multiple instance neural networks." Pattern Recognition 74 (2018): 15-24.  :star::star::star::star::star:

### SEMI-SUPERVISED LEARNING
* **[unsupervised loss function](https://arxiv.org/pdf/1606.04586.pdf)** Sajjadi, Mehdi, Mehran Javanmardi, and Tolga Tasdizen. "Regularization with stochastic transformations and perturbations for deep semi-supervised learning." Advances in Neural Information Processing Systems. 2016. 
* **[self-ensembling](https://arxiv.org/pdf/1610.02242.pdf)** Laine, Samuli, and Timo Aila. "Temporal ensembling for semi-supervised learning." arXiv preprint arXiv:1610.02242 (2016).


### Loss Function
* **[loss function based on probability map](https://arxiv.org/abs/1804.01793)** Jetley, Saumya, Naila Murray, and Eleonora Vig. "End-to-end saliency mapping via probability distribution prediction." Proceedings of Computer Vision and Pattern Recognition 2016 (2016): 5753-5761. 
* **[L-GM loss for image classification](https://arxiv.org/abs/1803.02988)** Wan, Weitao, et al. "Rethinking Feature Distribution for Loss Functions in Image Classification." arXiv preprint arXiv:1803.02988 (2018). [CVPR 2018]() :star::star::star::star::star: [implement](https://github.com/WeitaoVan/L-GM-loss)
* **[Crystal Loss(softmax+l\_2 norm)](https://arxiv.org/pdf/1804.01159.pdf)** Crystal Loss and Quality Pooling for Unconstrained Face Verification and Recognition.  submitted to TPAMI 2018. [previous version](https://arxiv.org/abs/1703.09507)
* **[ring loss for face recognation](https://arxiv.org/abs/1803.00130)** Zheng, Yutong, Dipan K. Pal, and Marios Savvides. "Ring loss: Convex Feature Normalization for Face Recognition." arXiv preprint arXiv:1803.00130 (2018). [CVPR 2018]() :star::star::star::star::star: [implement](https://github.com/Paralysis/ringloss)
* **[center loss](https://ydwen.github.io/papers/WenECCV16.pdf)** Wen, Yandong, et al. "A discriminative feature learning approach for deep face recognition." European Conference on Computer Vision. Springer, Cham, 2016.   :star::star::star::star::star:

### Feature Extraction
* **[Adaptive forward-backward greedy algorithm](http://ieeexplore.ieee.org/abstract/document/5895111/)** Zhang, Tong. "Adaptive forward-backward greedy algorithm for learning sparse representations." IEEE transactions on information theory 57.7 (2011): 4689-4708. 
 
 
## Unsupervised learning
### AAE 
* **[AAE](https://arxiv.org/pdf/1511.05644.pdf)** Makhzani, Alireza, et al. "Adversarial autoencoders." arXiv preprint arXiv:1511.05644 (2015).

### GAN

## pooling
* **[ContextLocNet: Context-Aware Deep Network Models for Weakly Supervised Localization](https://link.springer.com/content/pdf/10.1007\%2F978-3-319-46454-1_22.pdf)**


## Weakly Supervised Learning
* **[Weakly Supervised Object Localization with Progressive Domain Adaptation](https://www.cv-foundation.org/openaccess/content_cvpr_2016/app/S15-30.pdf)** classification and detection 2 steps. only image labels. [CVPR 2016]
* **[ICCV 2015 Deep Learning Face Attributes in the Wild ]**
* **[CVPR 2017 Weakly Supervised Cascaded Convolutional Networks](http://openaccess.thecvf.com/content_cvpr_2017/papers/Diba_Weakly_Supervised_Cascaded_CVPR_2017_paper.pdf)**
* **[CVPR 2016 Large Scale Semi-supervised Object Detection using Visual and Semantic Knowledge Transfer](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Tang_Large_Scale_Semi-Supervised_CVPR_2016_paper.pdf)** 
* **[CVPR 2018 Class Peak Response](Class Peak Response)**


## Semi-supervised
* http://ruder.io/semi-supervised/
