using Neural_Net_Cultivator.Inheritance;
using Neural_Net_Cultivator.Neural_Nets;
using Neural_Net_Cultivator.Tools_and_Reference;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System;

namespace Neural_Net_Cultivator
{
	class NNTrainer
	{
        private List<List<double>> TrainingInputs;
        private List<List<double>> TrainingOutputs;
        private List<IBaseNeuralNetwork> NetworkArray;

		public NNTrainer(List<IBaseNeuralNetwork> IncNetworkArray)
		{
            NetworkArray = IncNetworkArray;
		}

		private void LoadData(List<List<double>> Input, List<List<double>> Output)
		{
            TrainingInputs = Input;
            TrainingOutputs = Output;
	    }

        public List<IBaseNeuralNetwork> GetTrainedNetworks(List<List<double>> Input, List<List<double>> Output)
        {
            LoadData(Input,Output);
            Train();
            return NetworkArray;
        }

        private void Train()
        {
            List<double> TrainingResults;
            List<double> Error = new List<double>();


            for (int k = 0; k < Constants.NumTrainingCycles; k++)
            {
                for (int i = 0; i < TrainingInputs.Count(); i++)
                {
                    foreach (BaseNeuralNetwork x in NetworkArray)
                    {
                        Error.Clear();
                        //execute each network with starting inputs
                        TrainingResults = x.Execute(TrainingInputs[i].ToArray()).ToList();

                        Error = Extensions.CalculateError(TrainingOutputs[i], TrainingResults);

                        x.SetError(true, Error.ToArray());
                    }
                }
            }
        }

	}
}
