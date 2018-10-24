using Neural_Net_Cultivator.Inheritance;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Neural_Net_Cultivator.Neural_Nets
{
    [Serializable]
    class ThreeLayerPerceptron_John : BaseNeuralNetwork
    {

        #region Variables
       // double m_learningRate;
        double m_momentum;
        List<double> m_outputDirection;
        List<double> m_hiddenDirection;
        List<double> m_outputSumInputs;
        List<double> m_hiddenSumInputs;
        //number of nodes in each layer of the network
        //int m_numInput;
        //int m_numHidden;
        //int m_numOutput;
        //values at each node for the current passthrough
        List<double> m_inputValues;
        List<double> m_hiddenValues;
        List<double> m_outputValues;
        //error for hidden and output nodes
        List<double> m_outputError;
        //target values for output
        List<double> m_target;
        //weights
        List<List<double>> m_inputToHiddenWeights;
        List<List<double>> m_hiddenToOutputWeights;
        //previous cycle weight changes
        List<List<double>> m_inputToHiddenWeightChanges;
        List<List<double>> m_hiddenToOutputWeightChanges;
        //threshold values for each layer
        List<double> m_hiddenThreshold;
        List<double> m_outputThreshold;
        //seed for random number generation
        long m_seed;
        #endregion

        public ThreeLayerPerceptron_John(long seed)
        {
            m_seed = seed;
        }

        #region Getters
        public int getNumberInput()
        {
            return m_numInputs;
        }

        public int getNumberHidden()
        {
            return m_numHiddenLayers;
        }

        public int getNumberOutput()
        {
            return m_numOutputs;
        }

        public double getLearningRate()
        {
            return m_learningRate;
        }

        public List<double> getOutputs()
        {
            return m_outputValues;
        }
        #endregion

        public override double[] Execute(params double[] inputArray)
        {
            m_inputValues = inputArray.ToList<double>();

            return ForwardPropagate(); 
        }

        public override void CreateNetwork(double rate, int input, int hidden, int output, double incMomentum)
        {
            m_numInputs = input;
            m_numHiddenLayers = hidden;
            m_numOutputs = output;
            m_momentum = incMomentum;
            m_learningRate = (float)rate;

            //stuff maybe 


            //initialize variables
            m_inputValues = new List<double>();
            m_inputToHiddenWeights = new List<List<double>>();
            m_inputToHiddenWeightChanges = new List<List<double>>();
            for (int i = 0; i < m_numInputs; i++)
            {
                m_inputValues.Add(0);
                m_inputToHiddenWeights.Add(new List<double>());
                m_inputToHiddenWeightChanges.Add(new List<double>());

                for (int j = 0; j < m_numHiddenLayers; j++)
                {
                    m_inputToHiddenWeights[i].Add(0);
                    m_inputToHiddenWeightChanges[i].Add(0);
                }
            }

            m_hiddenValues = new List<double>();
            m_hiddenThreshold = new List<double>();
            m_hiddenDirection = new List<double>();
            m_hiddenSumInputs = new List<double>();
            m_hiddenToOutputWeights = new List<List<double>>();
            m_hiddenToOutputWeightChanges = new List<List<double>>();
            for (int i = 0; i < m_numHiddenLayers; i++)
            {
                m_hiddenValues.Add(0);
                m_hiddenThreshold.Add(0);
                m_hiddenDirection.Add(0);
                m_hiddenSumInputs.Add(0);
                m_hiddenToOutputWeights.Add(new List<double>());
                m_hiddenToOutputWeightChanges.Add(new List<double>());

                for (int j = 0; j < m_numOutputs; j++)
                {
                    m_hiddenToOutputWeights[i].Add(0);
                    m_hiddenToOutputWeightChanges[i].Add(0);
                }
            }

            m_outputValues = new List<double>();
            m_outputError = new List<double>();
            m_outputThreshold = new List<double>();
            m_outputDirection = new List<double>();
            m_outputSumInputs = new List<double>();
            m_target = new List<double>();
            for (int i = 0; i < m_numOutputs; i++)
            {
                m_outputValues.Add(0);
                m_outputError.Add(0);
                m_outputThreshold.Add(0);
                m_outputDirection.Add(0);
                m_outputSumInputs.Add(0);
                m_target.Add(0);
            }

            RandomizeWeights();
        }

        public void InputDataSet(List<double> set)
        {
            m_inputValues = set;
        }

        //Needs to be adjusted to interface
        public override void SetError(bool triggerBackProp, params double[] errors)
        {
            if (errors.Count<double>() == m_numOutputs)
                m_outputError = errors.ToList<double>();
            if (triggerBackProp)
            {
                BackPropagate();
            }
        }

        protected override void BackPropagate()
        {
            double temp1 = 0;

            for (int i = 0; i < m_numOutputs; i++)
            {
                m_outputDirection[i] = DerivSquashingFunction(m_outputSumInputs[i]) * m_outputError[i];
            }

            for (int i = 0; i < m_numHiddenLayers; i++)
            {
                temp1 = 0;
                m_hiddenDirection[i] = DerivSquashingFunction(m_hiddenSumInputs[i]);
                for (int j = 0; j < m_numOutputs; j++)
                    temp1 += m_outputDirection[j] * m_hiddenToOutputWeights[i][j];//c++ code has j as 0, can change back if that is correct, but seemed wrong from my understanding
                m_hiddenDirection[i] = m_hiddenDirection[i] * temp1;
            }
            
            for (int i = 0; i < m_numHiddenLayers; i++)
            {
                for (int j = 0; j < m_numOutputs; j++)
                {
                    m_hiddenToOutputWeightChanges[i][j] = m_learningRate * m_outputDirection[j] * m_hiddenValues[i] + (m_hiddenToOutputWeightChanges[i][j] * m_momentum);
                    m_hiddenToOutputWeights[i][j] += m_hiddenToOutputWeightChanges[i][j];
                }
            }
            
            for (int i = 0; i < m_numOutputs; i++)
            {
                m_outputThreshold[i] += m_learningRate * m_outputDirection[i];//had an additional *1.0, may add back in if we need to adjust value
            }

            for (int i = 0; i < m_numInputs; i++)
            {
                for (int j = 0; j < m_numHiddenLayers; j++)
                {
                    m_inputToHiddenWeightChanges[i][j] = m_learningRate * m_hiddenDirection[j] * m_inputValues[i] + (m_inputToHiddenWeightChanges[i][j] * m_momentum);
                    m_inputToHiddenWeights[i][j] += m_inputToHiddenWeightChanges[i][j];
                }
            }

            for (int i = 0; i < m_numHiddenLayers; i++)
            {
                m_hiddenThreshold[i] += m_learningRate * m_hiddenDirection[i];//had an additional *1.0, may add back in if we need to adjust value
            }
        }

       /* public override void SaveNN(string name, bool humanReadable = false)
        {
            throw new NotImplementedException();
        } */


        private double SquashingFunction(double input)
        {
            double exp_value = 0;
            double return_value = 0;

            exp_value = Math.Exp(-input);

            return_value = 1 / (1 + exp_value);

            return return_value;
        }

        private double DerivSquashingFunction(double input)
        {
            return SquashingFunction(input) * (1 - SquashingFunction(input));
        }

        //need to get Random working to use better rand rather than default random numbers
        public void RandomizeWeights()
        {
            Random tempRand = new Random();

            for (int i = 0; i < m_numInputs; i++)
            {
                for (int j = 0; j < m_numHiddenLayers; j++)
                {
                    m_inputToHiddenWeights[i][j] = tempRand.NextDouble();//will adjust to use random from c++ code, using this to expedite functionality
                    if (tempRand.NextDouble() > 0.5)
                        m_inputToHiddenWeights[i][j] = m_inputToHiddenWeights[i][j] * -1;
                }
            }

            for (int i = 0; i < m_numHiddenLayers; i++)
            {
                for (int j = 0; j < m_numOutputs; j++)
                {
                    m_hiddenToOutputWeights[i][j] = tempRand.NextDouble();
                    if (tempRand.NextDouble() > 0.5)
                        m_hiddenToOutputWeights[i][j] = m_hiddenToOutputWeights[i][j] * -1;
                }
            }
        }

        public void ZeroWeights()
        {
            for (int i = 0; i < m_numInputs; i++)
            {
                for (int j = 0; j < m_numHiddenLayers; j++)
                {
                    m_inputToHiddenWeights[i][j] = 0;
                }
            }

            for (int i = 0; i < m_numHiddenLayers; i++)
            {
                for (int j = 0; j < m_numOutputs; j++)
                {
                    m_hiddenToOutputWeights[i][j] = 0;
                }
            }
        }

        private void LayerForward(bool layer)
        {
            if (layer)
            {
                for (int i = 0; i < m_numHiddenLayers; i++)
                {
                    m_hiddenValues[i] = 0;
                    m_hiddenSumInputs[i] = 0;

                    for (int j = 0; j < m_numInputs; j++)
                    {
                        m_hiddenSumInputs[i] += m_inputValues[j] * m_inputToHiddenWeights[j][i];
                    }
                    m_hiddenSumInputs[i] += m_hiddenThreshold[i] * 1.0;
                    m_hiddenValues[i] = SquashingFunction(m_hiddenSumInputs[i]);
                }
            }
            else
            {
                for (int i = 0; i < m_numOutputs; i++)
                {
                    m_outputValues[i] = 0;
                    m_outputSumInputs[i] = 0;
                    for (int j = 0; j < m_numHiddenLayers; j++)
                    {
                        m_outputSumInputs[i] += m_hiddenValues[j] * m_hiddenToOutputWeights[j][i];
                    }
                    m_outputSumInputs[i] += m_outputThreshold[i] * 1.0;
                    m_outputValues[i] = SquashingFunction(m_outputSumInputs[i]);
                }
            }
        }

        protected override double[] ForwardPropagate()
        {
            LayerForward(true);
            LayerForward(false);

            return m_outputValues.ToArray();
        }

        
        /// <summary>
        /// Extracts all weights from the network, contained in a quad-indexed dictionary.  Bias is stored where indexes 1/3 and 2/4 match, addressing a single node.
        /// </summary>
        /// <returns>Quad-indexed dictionary containing weight values. 
        /// 4-Tuple key uses 0-base integral indexers [node1Layer, node1PositionInLayer, node2Layer, node2PositionInLayer]</returns>
        public override Dictionary<Tuple<int, int, int, int>, double> ExtractWeights()
        {
            Dictionary<Tuple<int, int, int, int>, double> weights = new Dictionary<Tuple<int, int, int, int>, double>();
            
            //TODO:: check with John that the indices are correct throughout this function
            //input to hidden weights
            for (int inputIndex = 0; inputIndex < m_inputToHiddenWeights.Count; inputIndex++)
            {
                for (int hiddenIndex = 0; hiddenIndex < m_inputToHiddenWeights[inputIndex].Count; hiddenIndex++)
                {
                    weights.Add(Tuple.Create<int,int,int,int>(0,inputIndex,1,hiddenIndex),m_inputToHiddenWeights[inputIndex][hiddenIndex]);
                }
            }

            //hidden to output weights
            for (int hiddenIndex = 0; hiddenIndex < m_hiddenToOutputWeights.Count; hiddenIndex++)
            {
                for (int outputIndex = 0; outputIndex < m_hiddenToOutputWeights[hiddenIndex].Count; outputIndex++)
                {
                    weights.Add(Tuple.Create<int, int, int, int>(1, hiddenIndex, 2, outputIndex), m_hiddenToOutputWeights[hiddenIndex][outputIndex]);
                }
                
            }

            for (int hiddenIndex = 0; hiddenIndex < m_hiddenThreshold.Count; hiddenIndex++)
            {
                //add the hidden layer thresholds
                weights.Add(Tuple.Create<int, int, int, int>(1, hiddenIndex, 1, hiddenIndex), m_hiddenThreshold[hiddenIndex]);
            }
       

            for (int outputIndex = 0; outputIndex < m_outputThreshold.Count; outputIndex++)
            {
                //add the output layer thresholds
                weights.Add(Tuple.Create<int, int, int, int>(2, outputIndex, 2, outputIndex), m_outputThreshold[outputIndex]);
            }

            return weights;
        }
        public override List<Double> ExtractWeightList()
        {
            throw new NotImplementedException();
        }
    }
}
