# Federated Learning
<b> MUIC ICCS311 Final Project: Federated Learning </b>

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
   -  Logic: Using local datasets, each client produces weight updates for the model, which will then send to a server to aggregate and improve the model. And the improved model will then send back to each client.  
```
Project Setup
   -  server.py
   -  client.py
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
![Screenshot from 2021-07-24 13-13-15](https://user-images.githubusercontent.com/60769071/126859458-d6abd2ae-6821-4183-94d1-e36367289612.png)
![Screenshot from 2021-07-24 15-53-58](https://user-images.githubusercontent.com/60769071/126863250-8df115c7-8a55-4348-9d6d-f7767314c26b.png)

As regard to the Effective CPU Utilization Histogram, it is clearly shown that CPUs are not effectively utilized. In fact, out of 8 logical CPUs, only 0.03 is utilized on average. The total CPU time is measured to be 0.610s, and this is interesting because 0.368s is the effective time, while another 0.242s is the spint time (i.e., the time spent waiting). 

![Screenshot from 2021-07-24 13-19-41](https://user-images.githubusercontent.com/60769071/126859526-d45270e6-6a9f-4e04-922d-99e90b12734f.png)

So, in this case, a significant portion of CPU time is spent waiting. However, to be fair, the main task of the <code> server.py </code> is to wait for the trained model from clients. 
<br><br>
Another interesting note is that the spin time of the function <code> sched_yield </code> alone is already 0.232s. Since the function <code> sched_yield </code> is the function that allows a thread to give up a control of a processor so that another thread can have the opportunity to run, this, therefore, suggests that the process of allowing another thread to run is needed to be improved in order to lessen the spint time and ultimately increase the performance, especially in a larger scale where there are more clients.  

## client.py
![Screenshot from 2021-07-24 12-24-33](https://user-images.githubusercontent.com/60769071/126858548-736427c7-bffe-4358-828f-7048bc963c38.png)
![Screenshot from 2021-07-24 13-07-29](https://user-images.githubusercontent.com/60769071/126859253-81f2faed-b558-4579-8e14-09d48ee0c08e.png)

