using System.Collections.Generic;
using System;

namespace Neural_Net_Cultivator.Inheritance
{
    
    public interface IBaseNeuralNetwork 
    {

        /// <summary>
        /// Initializes the network with the specified input.
        /// </summary>
        /// <param name="learningRate">The rate at which the network will learn.</param>
        /// <param name="inputCount">The input count of the first (input) layer.</param>
        /// <param name="hiddenLayerNodes">The number of hidden nodes per hidden layer.</param>
        /// <param name="outputCount">Number of outputs for the network to produce.</param>
        /// <param name="incMomentum">The incremental momentum of the network.</param>
        void CreateNetwork(double learningRate, int inputCount, int hiddenLayerNodes, int outputCount, double incMomentum);

        /// <summary>
        /// Executes the network with the given data as input.
        /// </summary>
        /// <param name="inputArray">Array of data to be passed in to the network.</param>
        /// <returns>Resulting data from the network.</returns>
        double[] Execute(params double[] inputArray);

        /// <summary>
        /// Set the errors relative to the current network output.
        /// </summary>
        /// <param name="triggerBackProp">True to trigger back-propagation after setting errors.</param>
        /// <param name="outputError">Array of error values relative to current network output. Size must equal number of outputs.</param>
        void SetError(bool triggerBackProp, params double[] outputError);

        /// <summary>
        /// Extracts all weights from the network, contained in a quad-indexed dictionary.
        /// </summary>
        /// <returns>Quad-indexed dictionary containing weight values. 
        /// Indexing format is 0-base, and uses (int node1Layer, int node1PositionInLayer, int node2Layer, int node2PositionInLayer)</returns>
        Dictionary<Tuple<int, int, int, int>, double> ExtractWeights();

        /// <summary>
        /// Extracts all weights from the network, contained in a list format
        /// </summary>
        /// <returns>list of doubles representing weights, biases are stored after their repective layers
        List<Double> ExtractWeightList();

        /// <summary>
        /// Saves a neural network to the specified location on disk.
        /// </summary>
        /// <param name="filePath">The file path and name to save to.  Both relative and absolute paths are supported.</param>
        /// <param name="humanReadable">If set to <c>true</c>, file will be saved in human-readable format.</param>
        void SaveNN(string filePath, bool humanReadable = false);

        // Static function present in BaseNeuralNetwork
        //static void LoadNN(string filePath, bool humanReadable = false);

    }
}
