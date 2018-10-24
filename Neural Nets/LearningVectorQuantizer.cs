using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_Net_Cultivator.Neural_Nets
{
    //an LVQ is identical to a MLP except that it lacks the hidden layer, it still has fully connected input and output layers
    class LearningVectorQuantizer : Inheritance.BaseNeuralNetwork
    {
        //input refrence then output refrence
        double[,] weights;
        int m_inputCount;
        int m_outputCount;
        double[] m_inputArray;
        double m_MaxValueLocation;

        // Learning rate        rate at which weights are adjusted, mosly inconsequntial, must be between 0 and 1, preferably closer to 1
        // Input count          equal to the size of the vectors you are classifying
        // Hidden Layer Nodes   inconsiquential, ignore
        // Output Count         best guess as to the number of classes the inputs will fall into, overestimating this number is fine, underestimating will provide incorrect results
        // Inc momentum         momentum value for the network, inconsequencial in this case
        public override void CreateNetwork(double learningRate, int inputCount, int hiddenLayerNodes, int outputCount, double incMomentum)
        {

            m_inputCount = inputCount;
            m_outputCount = outputCount;
            m_learningRate = learningRate;


            weights = new double[inputCount,outputCount];
            Random tempRand = new Random();
            for(int i = 0;i < outputCount;i++)
            {
                for(int j = 0;j < inputCount;j++)
                {
                    weights[j,i] = tempRand.NextDouble();
                    if (tempRand.NextDouble() > 0.5)
                    {
                        weights[j,i] *= -1;
                    }
                }
            }
        }

        //returns a catagory for the input rather then an array of values, so int rather than double[]
        public override double[] Execute(params double[] inputArray)
        {
            double[] MaxValueLocation = new double[1];

            m_inputArray = inputArray;
            ForwardPropagate();
            BackPropagate();

            MaxValueLocation[0] = m_MaxValueLocation;
            return MaxValueLocation;
        }

        //will stay unimplemented, all error is calculated internally
        public override void SetError(bool triggerBackProp, params double[] outputError)
        {
            throw new NotImplementedException();
        }

        //will stay unimplemented, unessicary for the completion of the assigned task for the moment
        public override Dictionary<Tuple<int, int, int, int>, double> ExtractWeights()
        {
            throw new NotImplementedException();
        }

        protected override void BackPropagate()
        {
            for(int i = 0; i < m_numInputs;i++)
            {
                //new weight = old weight + (learning rate * (input value for that weight - old weight))
                weights[i,(int)m_MaxValueLocation] += (m_learningRate * (m_inputArray[i] - weights[i,(int)m_MaxValueLocation]));
            }
        }

        protected override double[] ForwardPropagate()
        {
            //array to comply with inherited return type
            double[] MaxValueLocation = new double[1];

            double MaxValue = -9999;

            double CurrentSum = 0;


            //for each output to the lvq
            for (int i = 0; i < m_outputCount; i++)
            {
                //reset
                CurrentSum = 0;

                //for each connection between the current output and each input
                for (int j = 0; j < m_inputCount; j++)
                {
                    //calculate the sum of connections for the current output
                    CurrentSum += m_inputArray[j] * weights[j, i];
                }
                if (CurrentSum > MaxValue)
                {
                    MaxValue = CurrentSum;
                    MaxValueLocation[0] = i;
                }
            }

            m_MaxValueLocation = MaxValueLocation[0];
            return MaxValueLocation;
        }
        public override List<Double> ExtractWeightList()
        {
            throw new NotImplementedException();
        }
    }
}
