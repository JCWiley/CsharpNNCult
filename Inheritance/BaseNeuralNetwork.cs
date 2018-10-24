using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;

namespace Neural_Net_Cultivator.Inheritance
{
    
    [Serializable]
    abstract public class BaseNeuralNetwork : IBaseNeuralNetwork
    {

        #region Interface Methods


        /// <summary>
        /// Initializes the network with the specified input.
        /// </summary>
        /// <param name="learningRate">The rate at which the network will learn.</param>
        /// <param name="inputCount">The input count of the first (input) layer.</param>
        /// <param name="hiddenLayerNodes">The number of hidden nodes per hidden layer.</param>
        /// <param name="outputCount">Number of outputs for the network to produce.</param>
        /// <param name="incMomentum">The incremental momentum of the network.</param>
        public abstract void CreateNetwork(double learningRate, int inputCount, int hiddenLayerNodes, int outputCount, double incMomentum);

        /// <summary>
        /// Executes the network with the given data as input.
        /// </summary>
        /// <param name="inputArray">Array of data to be passed in to the network.</param>
        /// <returns>
        /// Resulting data from the network.
        /// </returns>
        public abstract double[] Execute(params double[] inputArray);

        /// <summary>
        /// Set the errors relative to the current network output.
        /// </summary>
        /// <param name="triggerBackProp">True to trigger back-propagation after setting errors.</param>
        /// <param name="outputError">Array of error values relative to current network output. Size must equal number of outputs.</param>
        public abstract void SetError(bool triggerBackProp, params double[] outputError);

        /// <summary>
        /// Extracts all weights from the network, contained in a quad-indexed dictionary.
        /// </summary>
        /// <returns>Quad-indexed dictionary containing weight values. 
        /// Indexing format is 0-base, and uses (int node1Layer, int node1PositionInLayer, int node2Layer, int node2PositionInLayer)</returns>
        public abstract Dictionary<Tuple<int, int, int, int>, double> ExtractWeights();
        /// <summary>
        /// Extracts all weights from the network, contained in a list format
        /// </summary>
        /// <returns>list of doubles representing weights, biases are stored after their repective layers
        public abstract List<Double> ExtractWeightList();

        #region File IO
        /// <summary>
        /// Saves a neural network to the specified location on disk.
        /// </summary>
        /// <param name="filePath">The file path and name to save to.  Both relative and absolute paths are supported.</param>
        /// <param name="humanReadable">If set to <c>true</c>, file will be saved in human-readable format.</param>
        public virtual void SaveNN(string filePath, bool humanReadable = false)
        {
            if (humanReadable)
            {
                throw new NotImplementedException();
            }
            else
            {
                Stream stream = File.Open(filePath, FileMode.Create);
                BinaryFormatter formatter = new BinaryFormatter();

                formatter.Serialize(stream, this);
                stream.Close();
            }
        }


        /// <summary>
        /// Reads in a neural network from the specified location on disk.
        /// </summary>
        /// <param name="filePath">The file path to the network to be loaded.  Both relative and absolute paths are supported.</param>
        /// <param name="humanReadable">If set to <c>true</c>, file will be parsed as human-readable format.</param>
        /// <returns>Reference to instance of loaded network.</returns>
        public static BaseNeuralNetwork LoadNN(string filePath, bool humanReadable = false)
        {
            BaseNeuralNetwork toLoad = null;

            if (humanReadable) //TODO:: auto-detect this.
            {
                throw new NotImplementedException();
            }
            else
            {
                Stream stream = File.Open(filePath, FileMode.Open);
                BinaryFormatter formatter = new BinaryFormatter();

                toLoad = (BaseNeuralNetwork)formatter.Deserialize(stream);
                stream.Close();
            }

            return toLoad;
        }
        #endregion

        #endregion


        #region Protected methods

        /// <summary>
        /// Propagate weight changes through network using previously set error values.
        /// </summary>
        protected abstract void BackPropagate();


        /// <summary>
        /// Propagates execution of perceptrons forward through network.
        /// </summary>
        /// <returns>The output of the last layer in the network</returns>
        protected abstract double[] ForwardPropagate();
        #endregion

        #region Data members
        protected double m_learningRate,
            m_incMomentum;
        protected double[] m_outputErrorValues;
        protected int m_numInputs,
            m_numHiddenLayers,
            m_numOutputs;
        #endregion
    }
}
