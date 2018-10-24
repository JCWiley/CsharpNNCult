using Neural_Net_Cultivator.Activation_Functions;
using Neural_Net_Cultivator.Inheritance;
using Neural_Net_Cultivator.Tools_and_Reference;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Neural_Net_Cultivator.Neural_Nets
{
    [Serializable]
    class MultiLayerPerceptron : Inheritance.BaseNeuralNetwork
    {
        #region Base Class Methods

        /// <summary>
        /// Initializes the network with the specified input.  Defaults to Sigmoid activation function.
        /// </summary>
        /// <param name="learningRate">The rate at which the network will learn.</param>
        /// <param name="inputCount">The input count of the first (input) layer.</param>
        /// <param name="hiddenLayers">Number of hidden layers for the network to contain.</param>
        /// <param name="hiddenLayerNodes">The number of hidden nodes per hidden layer.</param>
        /// <param name="outputCount">Number of outputs for the network to produce.</param>
        /// <param name="incMomentum">The incremental momentum of the network.</param>
        public override void CreateNetwork(double learningRate, int inputCount, int hiddenLayerNodes, int outputCount, double incMomentum)
        {
            // This class exposes a more specific creation function.  Passthrough to that, using standard 1 hidden layer network
            // and default Sigmoid functcion for activation.
            CreateNetwork(learningRate, inputCount, 1, hiddenLayerNodes, outputCount, incMomentum, new AF_Sigmoid());
        }

        /// <summary>
        /// Executes the network with the given data as input.
        /// </summary>
        /// <param name="inputArray">Array of input data.</param>
        /// <returns>
        /// Resulting data from the network.
        /// </returns>
        public override double[] Execute(params double[] inputArray)
        {
            #region Array slicing method
            /*  Unneeded in this implementation
            //set up variables for creating array slices
            int sliceStart = 0,
                sliceEnd;


            for (int i = 0; i < inputArray.Length; i++)
            {
                //set the slice to contain the number of inputs taken by this node
                sliceEnd = sliceStart + Nodes[0][i].NumInputs;

                //Grab a slice of the input array representing this node's inputs
                //(slices are start-inclusive, end-exclusive)
                Nodes[0][i].SetInputs(inputArray.Slice(sliceStart, sliceEnd));

                //set the next slice start according to the end of this slice
                sliceStart=sliceEnd;
            }
             */
            #endregion

            for (int i = 0; i < inputArray.Length; i++)
            {
                Nodes[0][i].SetInputs(inputArray[i]);
            }

            //forward-prop the network
            return ForwardPropagate();
        }


        /// <summary>
        /// Set the errors relative to the current network output.
        /// </summary>
        /// <param name="triggerBackProp">True to trigger back-propagation after setting errors.</param>
        /// <param name="outputError">Array of error values relative to current network output. Size must equal number of outputs.</param>
        /// <exception cref="System.ArgumentOutOfRangeException">Called when number of errors passed in does not match number of network outputs.</exception>
        public override void SetError(bool triggerBackProp, params double[] outputError)
        {
            if(outputError.Length != m_numOutputs)
            {
                throw new System.ArgumentOutOfRangeException("Number of expectedOutputs passed to MultiLayerPerceptron.SetError does not match actual output node count."
                                                            + "(received:  " + outputError.Length.ToString() + " - required: " + m_numOutputs.ToString() + ")");
            }

            // copy to internal array
            outputError.CopyTo(m_outputErrorValues, 0);

            if(triggerBackProp)
            {
                BackPropagate();
            }
        }

        /// <summary>
        /// Extracts all weights from the network, contained in a quad-indexed dictionary.  Bias is stored where indexes 1/3 and 2/4 match, addressing a single node.
        /// </summary>
        /// <returns>Quad-indexed dictionary containing weight values. 
        /// 4-Tuple key uses 0-base integral indexers [node1Layer, node1PositionInLayer, node2Layer, node2PositionInLayer]</returns>
        public override Dictionary<Tuple<int, int, int, int>, double> ExtractWeights()
        {
            Dictionary<Tuple<int, int, int, int>, double> weights = new Dictionary<Tuple<int,int,int,int>,double>();
            IBaseNeuralNode currentNode;
 
            //TODO:: This should thread readily.
            // Start at layer 1, processing weights between current and previous layer.
            for (int layerIndex = 1; layerIndex < Nodes.Length; layerIndex++)
            {
                for (int nodeIndex = 0; nodeIndex < Nodes[layerIndex].Length; nodeIndex++)
                {
                    //we can now use currentNode in place of Nodes[layerIndex][nodeIndex]
                    currentNode = Nodes[layerIndex][nodeIndex];

                    //The input count for a node denotes connections, and is equal to the number of nodes in the prior layer
                    for (int connectionIndex = 0; connectionIndex < currentNode.Inputs.Length; connectionIndex++)
                    {
                        // add the weight between the previous layer at the specified connection index, and the current layer at the current node index.
                        // fetch weight from current node at the current connection index.
                        weights.Add(Tuple.Create<int, int, int, int>(layerIndex-1, connectionIndex, layerIndex, nodeIndex), currentNode.Weights[connectionIndex]);
                    }

                    //save the bias
                    weights.Add(Tuple.Create<int, int, int, int>(layerIndex, nodeIndex, layerIndex, nodeIndex), currentNode.Bias);
                }
            }

            return weights;
        }
        /// <summary>
        /// Extracts all weights from the network, contained in a list format
        /// </summary>
        /// <returns>list of doubles representing weights, biases are stored after their repective layers
        public override List<Double> ExtractWeightList()
        {
            List<double> Weights = new List<double>();
            IBaseNeuralNode currentNode;

            // Start at layer 1, processing weights between current and previous layer.
            for (int layerIndex = 1; layerIndex < Nodes.Length; layerIndex++)
            {
                for (int nodeIndex = 0; nodeIndex < Nodes[layerIndex].Length; nodeIndex++)
                {
                    //we can now use currentNode in place of Nodes[layerIndex][nodeIndex]
                    currentNode = Nodes[layerIndex][nodeIndex];

                    //The input count for a node denotes connections, and is equal to the number of nodes in the prior layer
                    for (int connectionIndex = 0; connectionIndex < currentNode.Inputs.Length; connectionIndex++)
                    {
                        // add the weight between the previous layer at the specified connection index, and the current layer at the current node index.
                        // fetch weight from current node at the current connection index.
                        Weights.Add(currentNode.Weights[connectionIndex]);
                    }

                    //save the bias
                    Weights.Add(currentNode.Bias);
                }
            }

            return Weights;
        }
        /// <summary>
        /// Propagate weight changes through network using previously set error values.  Currently only implemented for 3-layer nets.
        /// </summary>
        /// <exception cref="System.NotImplementedException">
        /// Thrown if the current network is not a 3-layer net.
        /// </exception>
        protected override void BackPropagate()
        {
            if(m_numHiddenLayers == 1)
            {
                // Common Neural Net size
                ThreeLayerBackProp();
            }
            else
            {
                throw new NotImplementedException("Currently, only three-layer networks are supported.");
            }
        }

        /// <summary>
        /// Propagates execution of perceptrons forward through network.  Uses current stored input values for layer 0.
        /// </summary>
        /// <returns>
        /// The output of the last layer in the network
        /// </returns>
        protected override double[] ForwardPropagate()
        {
            //size of input layer
            double[] lastLayerOutputs = new double[Nodes[0].Length];
            double[] thisLayerOutputs;

            //execute nodes in the input layer (using their last stored input values).
            //save the results.
            for (int i = 0; i < Nodes[0].Length; i++)
            {
                lastLayerOutputs[i] = Nodes[0][i].Execute();
            }

            // For each array of perceptrons (each layer), not counting the input layer
            foreach (Perceptron[] layer in Nodes.Skip(1))
            {
                //Size this layer's output array to fit the number of nodes in this layer
                thisLayerOutputs = new double[layer.Length];

                //Execute each perceptron, passing in last layer's results and collecting the output
                //TODO:: Thread this
                for (int i = 0; i < layer.Length; i++)
                {
                    thisLayerOutputs[i] = layer[i].Execute(lastLayerOutputs);
                }
                //update the old layer output to point to this layer's results.  C# is neat.
                lastLayerOutputs = thisLayerOutputs;

                #region alternate method
                //TODO:: Determine which is best for threading
                /*
                //Execute each perceptron, passing in the outputs from last layer
                //TODO:: Thread this.
                foreach (Perceptron node in layer)
                {
                    node.Execute(layerOutputs);
                }
                
                //Once all execution is done, resize the array for this layer's outputs, and store results
                layerOutputs = new double[layer.Length]
                //TODO:: Thread this?  Can multiple threads safely access different elements in one array var?
                for (int i = 0; i < layer.Length; i++)
                {
                    layerOutputs[i] = layer[i].output);
                }*/
                #endregion
            }

            return lastLayerOutputs;
        }

        #endregion

        /// <summary>
        /// Initializes the network with the specified input.  Allows user-specification of activation function.
        /// </summary>
        /// <param name="learningRate">The rate at which the network will learn.</param>
        /// <param name="inputCount">The input count of the first (input) layer.</param>
        /// <param name="hiddenLayers">Number of hidden layers for the network to contain.</param>
        /// <param name="hiddenLayerNodes">The number of hidden nodes per hidden layer.</param>
        /// <param name="outputCount">Number of outputs for the network to produce.</param>
        /// <param name="incMomentum">The incremental momentum of the network.</param>
        /// <param name="activationFunction">The activation function to use for node output calculation.</param>
        /// <exception cref="System.NotSupportedException">MultiLayerPerceptron currently only supports 3-layer computation.</exception>
        public void CreateNetwork(double learningRate, int inputCount, int hiddenLayers, int hiddenLayerNodes, int outputCount, double incMomentum, IActivationFunction activationFunction)
        {
            #region Refresher on how-to ragged 2D array
            // msdn.microsoft.com/en-us-library/2s05feca.aspx
            //Nodes = new Perceptron[numHiddenLayers+2][];  // +2 for input/output
            //Nodes[0] = new Perceptron[numInOutNodes];     // Input layer
            //Nodes[1] = new Perceptron[nodesInHidden];
            //...                                           // for each hidden layer
            //Nodes[N-1] = new Perceptron[nodesInHidden];
            //Nodes[N] = new Perceptron[numInOutNodes];     // output layer
            #endregion
            if(hiddenLayers != 1)
            {
                throw new System.NotSupportedException("MultiLayerPerceptron currently only supports 3-layer computation.");
            }

            // Used to determine # of inputs when initializing each layer.
            int previousLayerNodes;

            // 2D node array.  First dimension is number of layers.
            // Create one array for each hidden layer, plus one each for input/output layers
            Nodes = new IBaseNeuralNode[hiddenLayers + 2][];

            #region Assign members
            m_learningRate = learningRate;
            m_incMomentum = incMomentum;
            m_numInputs = inputCount;
            m_numHiddenLayers = hiddenLayers;
            m_numOutputs = outputCount;
            #endregion

            #region Input layer
            // Size the input layer according to number of inputs
            Nodes[0] = new InputNode[inputCount];


            // Create the input nodes.
            for (int i = 0; i < inputCount; i++)
            {
                //Use InputNode class rather than Perceptron
                Nodes[0][i] = new InputNode();
            }
            #endregion

            #region Hidden layers
            // Nodes[0] is the input layer, so we start loop at 1,
            // iterate through # of hidden layers.
            // First layer nodes have same # inputs as number of input nodes.
            // Future layers have same # inputs as number of nodes in previous layer.
            previousLayerNodes = inputCount;
            for (int i = 1; i <= hiddenLayers; i++)
            {
                // Each Node[i] between 1 and hiddenLayers is a hidden layer.
                Nodes[i] = new Perceptron[hiddenLayerNodes];

                // Initialize this layer's nodes.
                for (int j = 0; j < Nodes[i].Length; j++)
                {
                    // Create the hidden nodes.  # of inputs equal to previous layer's node count.
                    // Default learning rate and threshold, random generated starting weights
                    Nodes[i][j] = new Perceptron(previousLayerNodes, activationFunction);
                }

                // Set node count for next layer's inputs
                previousLayerNodes = hiddenLayerNodes;
            }
            #endregion

            #region Output layer
            // Size the output layer according to number of outputs
            // Nodes has size of (hiddenLayers+2).  Subtract 1 to get last element = (hiddenLayers+1)
            Nodes[hiddenLayers + 1] = new Perceptron[outputCount];
            // Size the output error array too
            m_outputErrorValues = new double[outputCount];

            // Create the output nodes.  # of inputs equal to previous layer's node count.
            // Default learning rate and threshold, random generated starting weights
            for (int i = 0; i < outputCount; i++)
            {
                Nodes[hiddenLayers + 1][i] = new Perceptron(previousLayerNodes, activationFunction);
            }
            #endregion
        }

        /// <summary>
        /// Simple back-propagation algorithm for a 3-layer network.
        /// </summary>
        protected void ThreeLayerBackProp()
        {
            #region vars
            int hiddenNodes = Nodes[1].Length;
            //INeuralNode currentNode;
            IBaseNeuralNode currentOutputNode;
            IBaseNeuralNode currentHiddenNode;

            //representative of error value and "direction"
            double[] outputDirection = new double[m_numOutputs];
            double[] hiddenDirection = new double[hiddenNodes];

            // Temporary storage per node error sum
            double tempErrorSum = 0;
            #endregion

            #region Determine outputDirection
            //for each output node
            for (int i = 0; i < m_numOutputs; i++)
            {
                //get the current output node. We can now use currentOutputNode in place of any outputData[i]
                currentOutputNode = Nodes[2][i];

                // set error direction to equal derivative of pre-activated sum, multiplied by error value
                outputDirection[i] = currentOutputNode.Activation.Derivative(currentOutputNode.Sum) * m_outputErrorValues[i];
            }
            #endregion

            #region Determine hiddenDirection
            // for each hidden node
            for (int hiddenIndex = 0; hiddenIndex < hiddenNodes; hiddenIndex++)
            {
                // get the current hidden node.  We can now use currentHiddenNode in place of any hiddenData[hiddenIndex]
                currentHiddenNode = Nodes[1][hiddenIndex];

                //reset the error sum for the node
                tempErrorSum = 0;

                //set up the first half of hiddenDirection calculation, by storing the derivative of pre-activated node sum
                //this is equivalent to hiddenDirection[i] = deriv(hiddenSum[i])
                hiddenDirection[hiddenIndex] = currentHiddenNode.Activation.Derivative(currentHiddenNode.Sum);

                for (int outputIndex = 0; outputIndex < m_numOutputs; outputIndex++)
                {
                    //get the current output node. We can now use currentOutputNode in place of any outputData[outputIndex]
                    currentOutputNode = Nodes[2][outputIndex];

                    //sum up the error-weight product of all output connections to this hidden node 
                    //[output direction for the current output node] * [the output node's hidden-to-output weight for the current hidden node]
                    tempErrorSum += outputDirection[outputIndex] * currentOutputNode.Weights[hiddenIndex];
                }
                   
                //second half of hiddenDirection calculation - multiply by the sum of error in the output connections
                hiddenDirection[hiddenIndex] *= tempErrorSum;
            }
            #endregion

            #region Update weights
            //for each output node
            for (int i = 0; i < m_numOutputs; i++)
            {
                Nodes[2][i].AdjustWeights(outputDirection[i], m_incMomentum);
            }

            //for each hidden node
            for (int i = 0; i < hiddenNodes; i++)
            {
                Nodes[1][i].AdjustWeights(hiddenDirection[i], m_incMomentum);
            }
            #endregion

        }

        /// <summary>
        /// Sets the weights of the network according to the input.
        /// </summary>
        /// <param name="weights">Quad-indexed dictionary containing weight values. 
        /// 4-Tuple key expects 0-base integral indexers [node1Layer, node1PositionInLayer, node2Layer, node2PositionInLayer]</param>
        public void SetWeights(Dictionary<Tuple<int, int, int, int>, double> weights)
        {
            /* weights tetra-coordinate:
             * Item1    - node1Layer
             * Item2    - node1Position
             * Item3    - node2Layer
             * Item4    - node2Position
            */

            Tuple<int, int, int, int> key;

            foreach (KeyValuePair<Tuple<int, int, int, int>, double> weightPoint in weights)
            {
                key = weightPoint.Key;

                if(key.Item1==key.Item3 && key.Item2==key.Item4)
                {
                    //Value is node bias
                    Nodes[key.Item1][key.Item2].Bias = weightPoint.Value;
                }
                else
                {
                    if (key.Item1 != (key.Item3 - 1))
                    {
                        throw new InvalidOperationException("Malformed quad-index in MultiLayerPerceptron.SetWeights.  node1Layer should always be equal to or one less than node2Layer.");
                    }
                    Nodes[key.Item3][key.Item4].Weights[key.Item2] = weightPoint.Value;
                }
            }
        }

        /// <summary>
        /// Sets the weights of the network according to the input.
        /// </summary>
        /// <param name="weights">List containing weight values. 
        /// 
        public void SetWeightsList(List<double> weights)
        {
            int location = 0;
            // Start at layer 1, processing weights between current and previous layer.
            for (int layerIndex = 1; layerIndex < Nodes.Length; layerIndex++)
            {
                for (int nodeIndex = 0; nodeIndex < Nodes[layerIndex].Length; nodeIndex++)
                {

                    for (int connectionIndex = 0; connectionIndex < Nodes[layerIndex][nodeIndex].Inputs.Length; connectionIndex++)
                    {
                        Nodes[layerIndex][nodeIndex].Weights[connectionIndex] = weights[location + connectionIndex] ;
                    }

                    //manage recording current location in list
                    location += Nodes[layerIndex][nodeIndex].Inputs.Length;
                    //set the bias
                    Nodes[layerIndex][nodeIndex].Bias = weights[location];
                    //adjust for bias extraction
                    location += 1;
                   
                }
            }
        }
        #region Properties
        public IBaseNeuralNode[][] Nodes { get; set; }
        #endregion

    }
}
