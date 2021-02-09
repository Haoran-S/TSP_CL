# Learning to Continuously Optimize Wireless Resource In Episodically Dynamic Environment

There has been a growing interest in developing data-driven and in particular deep neural network (DNN) based methods for modern communication tasks. For a few popular tasks such as power control, beamforming, and MIMO detection, these methods achieve state-of-the-art performance while requiring less computational efforts, less channel state information (CSI), etc.  However, it is often challenging for these approaches to learn in a dynamic environment where parameters such as CSIs keep changing. 

This work develops a methodology that enables data-driven methods to continuously learn and optimize in a dynamic environment. Specifically, we consider an "episodically dynamic" setting where the environment changes in "episodes", and in each episode the environment is stationary.  We propose to build the notion of continual learning (CL) into the modeling process of learning wireless systems, so that the learning model can incrementally adapt to the new episodes, without forgetting knowledge learned from the previous episodes. Our design is based on a novel min-max formulation which ensures certain "fairness"  across different data samples. We demonstrate the effectiveness of the CL approach by customizing it to two popular DNN based models (one for power control and one for beamforming), and testing using both synthetic and real data sets.  These numerical results show that the proposed CL approach is not only able to adapt to the new scenarios quickly and seamlessly, but importantly, it maintains high performance over the previously encountered scenarios as well. 




References: [1] Haoran Sun, Wenqiang Pu, Minghe Zhu,  Xiao Fu, Tsung-Hui Chang, and Mingyi Hong, "Learning to Continuously Optimize Wireless Resource In Episodically Dynamic Environment." arXiv preprint arXiv:2011.07782 (2020).

Paper avaible at https://arxiv.org/abs/2011.07782.

---
### Demo:

- Install python 3.6

- Install pip dependencies
```pip install -r requirements.txt```

- For simulation with synthetic dataset, please run: 
```sh script_rand.sh```

- For simulation with DeepMIMO dataset

  - follow instructions in https://github.com/Haoran-S/DeepMIMO to generate DeepMIMO dataset

  - run: ```sh script_mimo.sh```



- (optional) follow instructions in https://github.com/Haoran-S/Coordinated_Beamforming to perform simulations for Coordinated Beamforming



---

@ Haoran Sun (sun00111@umn.edu) October 2020.

All rights reserved.
