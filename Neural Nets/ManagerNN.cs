using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Neural_Net_Cultivator;
using Neural_Net_Cultivator.Inheritance;
using Neural_Net_Cultivator.Tools_and_Reference;


namespace Neural_Net_Cultivator.Neural_Nets
{
    [Serializable]
    class ManagerNN : Inheritance.BaseNeuralNetwork
    {
        public ManagerNN(List<List<double>> TrainedWeights, List<List<double>> Input, List<List<double>> Output)
        {
            foreach(List<double> weights in TrainedWeights)
            {
                MultiLayerPerceptron temp = new MultiLayerPerceptron();
                temp.CreateNetwork(Constants.learningRate, Constants.InputCount, Constants.HiddenLayerNodes, Constants.OutputCount, Constants.Momentum);
                temp.SetWeightsList(weights);
                TrainedNetworks.Add(temp);
            }
            MInput = Input;
            MOutput = Output;
            PairIndex = 0;
        }

        public override void CreateNetwork(double learningRate, int inputCount, int hiddenLayerNodes, int outputCount, double incMomentum)
        {
            Manager.CreateNetwork(learningRate,inputCount,hiddenLayerNodes,TrainedNetworks.Count(),incMomentum);
        }

        public override double[] Execute(params double[] inputArray)
        {
            double[] ManagerChoice;
            
            // execute manager on input
            ManagerChoice = Manager.Execute(inputArray);
            // find the chosen network
            ChosenNetwork = Array.IndexOf(ManagerChoice,ManagerChoice.Max());
            // execute chosen network on input
            return TrainedNetworks[ChosenNetwork].Execute(inputArray);

        }
        public override void SetError(bool triggerBackProp, params double[] outputError)
        {
            // increment the count of which input output pair is being tested, should hopefully work
            if (PairIndex == MInput.Count())
            {
                PairIndex = 1;
            }
            else
            {
                PairIndex++;
            }

            List<double> ErrorList = new List<double>();
            int IdealNetworkIndex = 0;

            double[] Error = new double[TrainedNetworks.Count()];

            // try all the networks on the input and output pairs, recording the result, also, 0 out the values in the error array to be sent to the manager as feedback
            for(int i = 0;i < TrainedNetworks.Count();i++)
            {
                Error[i] = 0;
                ErrorList.Add(Extensions.CompareNetworks(TrainedNetworks[i],MInput[PairIndex - 1],MOutput[PairIndex - 1]));
            }
            // compare the results to find the ideal network
            IdealNetworkIndex = Array.IndexOf(ErrorList.ToArray(),ErrorList.Min());
            // if the chosen network and the ideal network are the same
            if(IdealNetworkIndex == ChosenNetwork)
            {
                // slight positive feedback to the network chosen
                Error[ChosenNetwork] = Constants.PosReinforcement;
            }
            else // if their not the same
            {
                // positive feedback to the ideal and negative to the chosen, no changes to anything else
                Error[IdealNetworkIndex] = Constants.PosReinforcement;
                Error[ChosenNetwork] = Constants.NegReinforcemnt;
            }
            // trigger internal backprop
            Manager.SetError(triggerBackProp, Error);

        }

        public override Dictionary<Tuple<int, int, int, int>, double> ExtractWeights()
        {
            return Manager.ExtractWeights();
        }

        public override List<double> ExtractWeightList()
        {
            return Manager.ExtractWeightList();
        }

        public void SetWeights(Dictionary<Tuple<int, int, int, int>, double> weights)
        {
            Manager.SetWeights(weights);
        }

        public void SetWeightsList(List<double> weights)
        {
            Manager.SetWeightsList(weights);
        }
        #region Properties
        List<List<double>> MInput;
        List<List<double>> MOutput;
        List<MultiLayerPerceptron> TrainedNetworks = new List<MultiLayerPerceptron>();
        MultiLayerPerceptron Manager = new MultiLayerPerceptron();
        int ChosenNetwork;
        int PairIndex;
        #endregion

        protected override void BackPropagate()
        {
            throw new NotImplementedException();
        }

        protected override double[] ForwardPropagate()
        {
            throw new NotImplementedException();
        }
    }
}
