\documentclass{article}
\usepackage{amsmath}
\usepackage{amssymb} % For symbols like \mathcal
\usepackage{amsfonts}

\begin{document}

\section*{Nomenclature}

\subsection*{General Mathematical and Physical Symbols}
\begin{itemize}
    \item <span class="math-inline">t</span>: Time
    \item <span class="math-inline">T</span>: Total duration of a trial or recording period
    \item <span class="math-inline">dt</span> or <span class="math-inline">DT</span>: Infinitesimal time step or discrete time step in simulations
    \item <span class="math-inline">f\(x\)</span>: A function of variable <span class="math-inline">x</span>
    \item <span class="math-inline">\\frac\{dx\}\{dt\}</span> or <span class="math-inline">\\dot\{x\}</span>: First derivative of <span class="math-inline">x</span> with respect to time <span class="math-inline">t</span>
    \item <span class="math-inline">\\frac\{\\partial f\}\{\\partial x\}</span>: Partial derivative of function <span class="math-inline">f</span> with respect to variable <span class="math-inline">x</span>
    \item <span class="math-inline">\\sum</span>: Summation operator
    \item <span class="math-inline">\\int</span>: Integral operator
    \item <span class="math-inline">e</span>: Base of the natural logarithm
    \item <span class="math-inline">\\log\(x\)</span>: Natural logarithm of <span class="math-inline">x</span>
    \item <span class="math-inline">\\Theta\(x\)</span>: Heaviside step function (typically 1 if <span class="math-inline">x \> 0</span>, 0 otherwise)
    \item <span class="math-inline">\\delta\_\{ij\}</span>: Kronecker delta (1 if <span class="math-inline">i\=j</span>, 0 otherwise)
    \item <span class="math-inline">V</span>: Voltage or Potential
    \item <span class="math-inline">I</span>: Electric Current
    \item <span class="math-inline">C</span>: Capacitance
    \item <span class="math-inline">g</span>: Conductance
    \item <span class="math-inline">\\tau</span>: Time constant
\end{itemize}

\subsection*{Neuron Model Variables}
\subsubsection*{General Spiking Neuron}
\begin{itemize}
    \item <span class="math-inline">V\_\{th\}</span> or <span class="math-inline">\\vartheta</span>: Firing threshold voltage
    \item <span class="math-inline">V\_\{reset\}</span>: Reset potential after a spike
    \item <span class="math-inline">S\[t\]</span>: Spike event at time <span class="math-inline">t</span> (binary)
\end{itemize}

\subsubsection*{Hodgkin-Huxley Model}
\begin{itemize}
    \item <span class="math-inline">V\_m</span>: Membrane potential
    \item <span class="math-inline">C\_m</span>: Membrane capacitance
    \item <span class="math-inline">I\_\{Na\}</span>: Sodium ion current
    \item <span class="math-inline">I\_K</span>: Potassium ion current
    \item <span class="math-inline">I\_L</span>: Leakage current
    \item <span class="math-inline">I\_\{ext\}</span>: External applied current
    \item <span class="math-inline">\\overline\{g\}\_\{Na\}, \\overline\{g\}\_\{K\}, \\overline\{g\}\_\{L\}</span>: Maximum conductances for <span class="math-inline">Na^\+</span>, <span class="math-inline">K^\+</span>, and Leakage channels, respectively
    \item <span class="math-inline">m, h</span>: Activation and inactivation gating variables for <span class="math-inline">Na^\+</span> channels
    \item <span class="math-inline">n</span>: Activation gating variable for <span class="math-inline">K^\+</span> channels
    \item <span class="math-inline">E\_\{Na\}, E\_K, E\_L</span>: Equilibrium (Nernst) potentials for <span class="math-inline">Na^\+</span>, <span class="math-inline">K^\+</span>, and Leakage, respectively
    \item <span class="math-inline">\\alpha\_x, \\beta\_x</span>: Voltage-dependent rate constants for gating variable <span class="math-inline">x \\in \\\{m,h,n\\\}</span>
\end{itemize}

\subsubsection*{Leaky Integrate-and-Fire (LIF) Model}
\begin{itemize}
    \item <span class="math-inline">V\[t\]</span> (or <span class="math-inline">V\(t\)</span>): Membrane potential at time <span class="math-inline">t</span>
    \item <span class="math-inline">H\[t\]</span>: Intermediate membrane potential at time <span class="math-inline">t</span> before reset logic
    \item <span class="math-inline">X\[t\]</span>: Input current at time <span class="math-inline">t</span> resulting from incoming spikes
    \item <span class="math-inline">\\tau\_\{mem\}</span>: Membrane time constant (often denoted just as <span class="math-inline">\\tau</span> in LIF equations)
    \item <span class="math-inline">\\tau\_\{syn\}</span>: Synaptic time constant
    \item <span class="math-inline">I\(t\)</span>: Total input current to the neuron at time <span class="math-inline">t</span> (used in differential equation form)
\end{itemize}

\subsubsection*{Izhikevich Model}
\begin{itemize}
    \item <span class="math-inline">v</span>: Membrane potential
    \item <span class="math-inline">u</span>: Membrane recovery variable
    \item <span class="math-inline">a, b, c, d</span>: Parameters of the Izhikevich model
\end{itemize}

\subsection*{Artificial Neural Network (ANN) & Spiking Neural Network (SNN) Symbols}
\begin{itemize}
    \item <span class="math-inline">w\_i</span>: Synaptic weight of the <span class="math-inline">i</span>-th input
    \item <span class="math-inline">x\_i</span>: Value of the <span class="math-inline">i</span>-th input
    \item <span class="math-inline">b</span>: Bias term (in ANNs)
    \item <span class="math-inline">\\varphi\(\\cdot\)</span>: Activation function (in ANNs)
    \item <span class="math-inline">y</span>: Output of a neuron or network
    \item <span class="math-inline">W^\{xh\}</span>: Weight matrix connecting input layer <span class="math-inline">x</span> to hidden layer <span class="math-inline">h</span>
    \item <span class="math-inline">W^\{hy\}</span>: Weight matrix connecting hidden layer <span class="math-inline">h</span> to output layer <span class="math-inline">y</span>
    \item <span class="math-inline">h\_j</span>: Activation of the <span class="math-inline">j</span>-th hidden neuron
    \item <span class="math-inline">e\_n</span>: Spike from neuron <span class="math-inline">n</span> (often a vector indicating which neuron spiked)
    \item <span class="math-inline">N\_\{batch\}</span>: Number of samples/trials in a batch
    \item <span class="math-inline">N\_\{out\}</span> or <span class="math-inline">N\_\{class\}</span>: Number of output neurons or classes
\end{itemize}

\subsection*{Training Algorithm Symbols}
\subsubsection*{General Training}
\begin{itemize}
    \item <span class="math-inline">L</span> or <span class="math-inline">\\mathcal\{L\}</span>: Loss function or cost function
    \item <span class="math-inline">y\_i</span>: Actual output of the <span class="math-inline">i</span>-th unit/class
    \item <span class="math-inline">\\hat\{y\}\_i</span> or <span class="math-inline">y\_\{label\}</span>: Target or desired output for the <span class="math-inline">i</span>-th unit/class
    \item <span class="math-inline">\\eta</span>: Learning rate
    \item <span class="math-inline">\\theta\_k</span>: <span class="math-inline">k</span>-th trainable parameter of the model
\end{itemize}

\subsubsection*{Backpropagation & Gradient Descent}
\begin{itemize}
    \item <span class="math-inline">\\frac\{\\partial L\}\{\\partial w\}</span> or <span class="math-inline">\\frac\{d\\mathcal\{L\}\}\{dw\}</span>: Gradient of the loss function <span class="math-inline">L</span> (or <span class="math-inline">\\mathcal\{L\}</span>) with respect to weight <span class="math-inline">w</span>
\end{itemize}

\subsubsection*{Eventprop}
\begin{itemize}
    \item <span class="math-inline">\\lambda\_V\(t\)</span>: Adjoint variable associated with the membrane potential <span class="math-inline">V</span> at time <span class="math-inline">t</span>
    \item <span class="math-inline">\\lambda\_I\(t\)</span>: Adjoint variable associated with the input current <span class="math-inline">I</span> at time <span class="math-inline">t</span>
    \item <span class="math-inline">\\tilde\{\\lambda\}\_\{V,j\}, \\tilde\{\\lambda\}\_\{I,j\}</span>: Batch