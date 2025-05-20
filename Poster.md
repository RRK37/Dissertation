A+: Outstanding eye-catching, creative and interesting poster that provides a very clear summary of the project and excellent evidence of high-quality engineering. Likely to be prize-winning.

# Neuroscience Meets Computation

## *Simulating biologically accurate neurons for learning speech recognition.* 

> Turn this into bulletpoints

Power consumption of ANNs

Sparsity for energy efficiency

Within the human mind, the world is perceived through visual recognition, actions are guided by motor control, and meaning is extracted from sound. Simultaneously the ceaseless functioning of our internal systems, gives rise to the phenomenon of consciousness, and weaves together natural language to convey meaning. This all occurs within a budget of roughly 20 Watts of power.

Meanwhile a GPU running a large language model can consume 400 Watts, running on  20 Watts when completely idle. 

This power efficient information processing capability of biological neurons has inspired research into neuromorphic computing - the field of creating software and hardware systems to mimic the function of the human brain. The importance of power efficiency is stressed by the fact that AI power consumption is increasing exponentially [@kindigAIPowerConsumption], causing sustainability and financial conerns. The efficiency of biologically inspired computing was investigated by researchers at Intel, who found that running AI inference and solving optimisation tasks on neuromorphic hardware - in this case the Loihi 2 platform - uses **100 times less energy** and runs **50 times faster** [@daviesLoihiNeuromorphicManycore2018]. 

![[second_vs_third_gen_neuron.png]]
(a) Second generation neurons output a weighted sum of its inputs added to a bias. For instance the inputs could be floating point numbers in the range of 0 to 1, these would be multiplied by their corresponding weights and added together, along with a bias. Then the result is passed through an activation function so that the output is also in the range 0 to 1. 

(b) Third generation - neuromorphic - neurons receive spikes which are multiplied by a weight term, these spikes cause the internal voltage of the neuron to increase. The internal voltage of the neuron resets to 0 once it reaches a threshold value, at this point the neuron also releases a spike. That is why these are called *spiking* neurons. 


Information in spiking neural networks is encoded in the timings of when the spikes occur, therefore they can inherently process temporal data. For instance, an application of this temporal information processing is in speech recognition. Speech processing is a very relevant field currently, with more and more of our interaction with computers happening through natural language due to the prevalence of large language models. We can see evidence of the trajectory we are going in by taking a look at what some of the biggest companies in the world are deciding to focus on. Meta has released smart glasses in collaboration with Ray-Ban, where the only input is via voice. Apple has invested heavily in building, and marketing, a new Siri, deeply integrated with large language models. These require speech to be recognised and processed by machines quickly - so that the user can be immersed - and efficiently - due to the battery limitations of these devices. These show to be ripe applications of spiking neural networks. 


The challenge with training spiking neural networks is that spikes are non-differentiable; during spike events, the gradient shoots to infinity. This has caused doubts about weather backpropagation - the cornerstone of machine learning - could be applied for spiking neural networks. As a result, for years people explored alternatives to backpropagation. However, new research has found that applying the adjoint method - from optimisation theory - to spiking neural networks enables calculating the exact gradient of the loss with respect to the weight. This method is called Eventprop.

Loss functions quantify how bad or good a model has done on a given test. Once a loss function is calculated, the gradient tells the optimisation algorithm how to adjust the weights to minimise the loss function - and therefore improve the model. Therefore choosing a good loss function is decisive for a model's accuracy, stability, and speed of learning. 

A recent study on Eventprop found that using an exponentially weighted sum of output voltage achieved state-of-the-art accuracy for a speech recognition task called Spiking Heidelberg Digits. 
$$
\mathcal{L}_{sum-exp} = -\frac{1}{N_{batch}} \sum^{N_{bathc}}_{m=1}log(\frac{exp(\int^{T}_{0} e^{-t/T}V^{m}_{l(m)}(t) dt )}{\sum^{N_{out}}_{k=1}exp(\int^{T}_{0}e^{-t/T}V^{m}_{k}(t) dt)})
$$



My research looked into finding a better loss function and applying it to Eventprop. 






What is a Neural Network?

A collection of neurons connected by synapses. 

The synapse weights determine how strong a connection between two neurons is. 

These weights are adjusted by algorithms such as backpropagation to be able to model a system accurately.

The output of a neural network is often called "inference", as the model is inferring what the output should be based on previous examples. 

Layers in neural networks can be connected back into themselves, this is known as recurrence and forms recurrent neural networks.


What is a Spiking Neuron?

Biologically inspired spiking neurons - third generation neurons - behave very differently than the standard artificial neurons - second generation neurons.

(a) Second generation neurons output a weighted sum of its inputs added to a bias. The output is transformed by an activation function and passed on to the next neurons.

(b) Third generation - neuromorphic - neurons receive spikes through weighted synapses. Spikes cause the internal voltage of the neuron to increase. When the internal voltage reaches a threshold, the neuron outputs a spike to the next neurons. 


Why Spiking Neural Networks?

The human mind - 20 Watts:

Perceives world through visual recognition.
Guides actions through motor control.
Extracts meaning from sound.
Keeps ceaseless functioning of our internal systems.
Gives rise to the phenomenon of consciousness.
Weaves together natural language to convey abstract meaning.

A GPU - 400 Watts:

Runs a large language model.
Produces AI slop and hallucinates. 


With the global power consumption of AI increasing exponentially, spiking neural networks offer an efficient alternative. Studies have shown that running AI inference on neuromorphic platforms as opposed to optimised traditional platforms uses 100x less energy and runs 50x faster. 


Challenges of Training Spiking Neural Networks

Less researched than traditional artificial neural networks.
Spikes are non-differentiable so there have been doubts about the application backpropagation.
Harder to parallelise computationally due to the strict sequential nature of spiking neurons.


Training Spiking Neural Networks Using "Eventprop"

A novel method for training SNNs.
An application of adjoint method from optimisation theory. 
Calculates the exact gradient despite non-continuity of spikes.
Applies precise backpropagation.
Training algorithm 4x faster and uses 3x less memory than top competitors which use approximate gradients.
Achieves state-of-the-art performance for spiking speech dataset - SHD.


Need for a Novel Loss Function


Current loss function has achieved state-of-the-art performance on spiking speech dataset - SHD. 
Current loss function fixes inefficiencies caused by spikes occuring near the end of the recording.
However, it emphasises very early spikes the most. I hypothesised that this hinders learning rate due to the fact that spikes happening instantaneously to the recording starting are caused by noise.


Deriving and Implementing a Novel Loss Function

I implemented a Gaussian-based loss function, aiming to minimise the effect of noise caused by initial spikes at the output.
I derived the mathematical Eventprop scheme necessary to propagate the error signals and gradient backwards in the network.
This greatly increased the learning rate over the previous state-of-the-art method. 