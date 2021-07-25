# Federated Learning
<b> MUIC ICCS311 Final Project: Federated Learning </b>

# Repository Files
```
# python files for federated learning implementation
- client.py
- server.py
# zipped files containing the matrices generating by Intel VTune Profiler
- client_analysis.zip
- server_analysis.zip
# play around with parameters, exploring federated learning
- play_around.zip
```
# What I have learned ...
### 1. The Concept of Fedreated Learning:
   -  A machine learning approach where an algorithm is trained across multiple decentralized devices holding local datasets with exchanging the data samples.
   -  Unlike traditional machine learning, the raw data is not aggregated to a centralized server; rather, it is left distributed on the client devices.
   -  Federated Learning Applications are used on ... 
      -  Internet of Things (IoT)
      -  Smartphones

### 2. Federated Learning Implementation using PyTorch and Flower
   -  Train a Convolutional Neural Network on CIFAR10
   -  1 server, 2 clients with the same model
   -  Logic: Using local datasets, each client produces weight updates for the model, which will then send to a server to aggregate and improve the model. And the improved model will then send back to each client. Repeat this process 3 times.
   -  NOTE: If the local datasets do not exist, the datasets will be downloaded first.
```
Project Setup
   -  server.py # host the server
   -  client.py # train the model
```
```
Package Requirements
   -  Python3
   -  Pytorch
   -  Torchvision
   -  Flower
```

### 3. Intel VTune Profiler
   -  Tool for code analysis
   -  Shows the CPU utilization, the time spent on each function call, etc
   -  Guide to performance bottleneck

# Findings: Code Analysis
<b> System Information </b>
```
CPU: 11th Gen Intel® Core™ i7-1165G7 @ 2.80GHz, 4 Cores 8 Threads. 1 Physical Processor
RAM: 16 GB
OS: UBUNTU 20.04.2 LTS
Python Verison: Python 3.8.10

NOTE: Number of Logical CPUs = Number of Physical Processor x Number of Threads = 8 Logical CPUs
```
## server.py
<img src = "https://user-images.githubusercontent.com/60769071/126859458-d6abd2ae-6821-4183-94d1-e36367289612.png" width = "100%">
<img src = "https://user-images.githubusercontent.com/60769071/126863250-8df115c7-8a55-4348-9d6d-f7767314c26b.png" width = "100%">

As regard to the Effective CPU Utilization Histogram, it is clearly shown that CPUs are not effectively utilized. In fact, they are "idle". Out of the 8 logical CPUs, only 0.03 is utilized on average. Also, the total CPU time is measured to be at 0.610s, and this is interesting because 0.368s is the effective time, while another 0.242s is the spint time (i.e., the time spent waiting).

<img src = "https://user-images.githubusercontent.com/60769071/126859526-d45270e6-6a9f-4e04-922d-99e90b12734f.png" width = "100%">

So, in this case, a significant portion of CPU time is spent waiting. However, to be fair, the main task of the <code>server.py</code> is to wait for the trained model from clients. 

<img src = "https://user-images.githubusercontent.com/60769071/126865069-a1d5e449-a5c3-4e14-9b73-cee6caa5724b.png" width = "100%">

Another interesting note is that the spin time of the function <code>sched_yield</code> alone is already 0.232s. Since the function <code>sched_yield</code> is the function that allows a thread to give up a control of a processor so that another thread can have the opportunity to run, this, therefore, suggests that the process of allowing another thread to run is needed to be improved in order to lessen the spint time and ultimately increase the performance, especially in a larger scale where there are more clients.  

## client.py
<img src = "https://user-images.githubusercontent.com/60769071/126858548-736427c7-bffe-4358-828f-7048bc963c38.png" width = "100%">
<img src = "https://user-images.githubusercontent.com/60769071/126872538-e66a0410-d58e-41ba-8f99-c54fed60311e.png" width = "100%">

Comparing the Effective CPU Utilization Histogram of the <code>client.py</code> to that of the <code>server.py</code>, it seems that the <code>client.py</code> is better in utilizing the CPUs. However, it is still in a "poor" area with 2.67 logical CPUs simultaneously utilized on average. Also, the total CPU time is 511.529s: 510.919s for the effective time and 0.0610s for the spin time (i.e., the time spent waiting).

<img src = "https://user-images.githubusercontent.com/60769071/126859253-81f2faed-b558-4579-8e14-09d48ee0c08e.png" width = "100%">

There are two functions call that take up most of the CPU time, which are <code>func@0x18c30</code> and <code>func@0x18ad0</code>. And unlike the funciton <code>sched_yeild</code> in <code>server.py</code>, the CPU time of these two functions are not a spin time; rather, they are an effective time. So, what are the tasks of these functions? To be honest, I'm not sure. However, an interesting thing is that if we trace back the call stack, we would find that both of these two functions are realted to the function <code>train()</code>. 

<img src = "https://user-images.githubusercontent.com/60769071/126873609-dc390d5c-2398-4d03-85d2-b49f8eb51677.png" width = "100%">
<img src = "https://user-images.githubusercontent.com/60769071/126873609-dc390d5c-2398-4d03-85d2-b49f8eb51677.png" width = "100%">

Now, if we look at the <code>client.py</code>, we can see that although the function <code>main()</code> is the function that takes up the most CPU time, a large portion of it actually comes from the function <code>train()</code> as shown in the images below.

<img src = "https://user-images.githubusercontent.com/60769071/126873712-f1d9cad9-3b6d-438a-a576-77db9b6b9544.png" width = "100%">
<img src = "https://user-images.githubusercontent.com/60769071/126873596-5f79bc1a-f125-4edf-b636-973b55f51787.png" width = "100%">

It is clear that the function <code>train()</code> plays a major role in the performance aspect. Therefore, it would be safe to say that this function <code>train()</code> is a performance bottleneck of the federated learning algorithm, or at least for this particular implementation.

## So, what could be improved?
Firstly, as suggested by the histogram, more logical CPUs should be utilized, and this could be done by further parallelizing the algorithm and distribute tasks to the unused ones. Also, since the function <code>train()</code> contains multiple similar instructions, one of the possible solutions could be parallelizing the function <code>train()</code> and then run it on GPU instead.
<br><br>
In fact, this has been done by one of my classmates, Mr. Bhumrapee Soonjun. He applies CUDA and runs the algorithm on the GPU. 

## Federated Learning Comparison: CPU vs GPU
According to Bhumrapee Soonjun (2021), with the same algorithm running on GPU, the elapsed time of the <code>client.py</code> is roughly 36 seconds. And this is very impressive considering the CPU implementation spent 191.555 seconds, which is approximately 6 times slower. However, like the CPU implementation, his findings also hows that the paralleism of the algorithm is still "low". 

<hr>
All in all, the process of training the data should be further parallelized and run on GPU; however, it is also very crucial to make sure that we only trigger the GPU kernel when we needed to. By minimizing the number of function calls and data transfers, the algorithm can therefore be potentially faster. 
<hr>

# References
<a href = "https://arxiv.org/pdf/1908.07873.pdf"> Federated learning: Challenges, methods, and future directions </a> <br>
<a href = "https://flower.dev/docs/quickstart_pytorch.html"> Flower: A Friendly Federated Learning Framework </a>
