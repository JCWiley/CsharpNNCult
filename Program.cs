using Neural_Net_Cultivator.Neural_Nets;
using Neural_Net_Cultivator.Inheritance;
using Neural_Net_Cultivator;
using System.Collections.Generic;
using System;
using Neural_Net_Cultivator.Tools_and_Reference;

namespace Neural_Net_Cultivator
{
    class Program
    {
        static void Main(string[] args)
        {
            // Storage
            List<double> TempLinearWeights = new List<double>();
            List<List<double>> LinearWeights = new List<List<double>>();
            List<List<int>> SortedIndices = new List<List<int>>();
            List<List<double>> NewWeights = new List<List<double>>();
            List<IBaseNeuralNetwork> ToEvaluate = new List<IBaseNeuralNetwork>();
            //List<MultiLayerPerceptron> ChosenNetworks = new List<MultiLayerPerceptron>();

            // Data set storage variables
            List<List<double>> TrainingInput = new List<List<double>>();
            List<List<double>> TrainingOutput = new List<List<double>>();
            List<List<double>> TestingInput = new List<List<double>>();
            List<List<double>> TestingOutput = new List<List<double>>();

            // Classification and sorting variables
            List<double> Classifications = new List<double>();
            double Classification;
            int NumNetworkClassifications = 0;

            // Create storage for completed networks
            List<IBaseNeuralNetwork> ListOfNetworks = new List<IBaseNeuralNetwork>();

            // Create Training class
            NNTrainer Trainer = new NNTrainer(ListOfNetworks);

            // Create LVQ
            LearningVectorQuantizer LVQ = new LearningVectorQuantizer();

            // Create Generic network storage for later testing
            List<IBaseNeuralNetwork> M = new List<IBaseNeuralNetwork>();

            // Create more generic network storage
            IBaseNeuralNetwork M2;

            //create another NN trainer for later use
            NNTrainer SecondaryTrainer = new NNTrainer(M);

            //create tester for ManagerNN
            TestingAndReporting ManagerTester;

            //create tester for generic NN
            TestingAndReporting NNTester;

            //------------
            //decrementable num training
            int Num_Training = 120;

            for (; Num_Training > 30; Num_Training--)
            {
                // Reset all storage variables for the next run through
                TempLinearWeights.Clear();
                LinearWeights.Clear();
                SortedIndices.Clear();
                NewWeights.Clear();

                TrainingInput.Clear();
                TrainingOutput.Clear();
                TestingInput.Clear();
                TestingOutput.Clear();

                Classifications.Clear();
                Classification = 0;
                NumNetworkClassifications = 0;

                ListOfNetworks.Clear();

                M.Clear();

                //Initialize LVQ // hidden layer count is here to comply with the base neural network format, it is not actually used
                LVQ.CreateNetwork(Constants.LVQLearningRate, ((Constants.InputCount + 1) * Constants.HiddenLayerNodes) + ((Constants.HiddenLayerNodes + 1) * Constants.OutputCount), Constants.LVQHiddenLayer, Constants.LVQOutputCount, Constants.LVQMomentum);

                // Load data into storage variables
                Extensions.LoadData(ref TrainingInput, ref TrainingOutput, ref TestingInput, ref TestingOutput);

                // Create networks to train
                for (int i = 0; i < Constants.NumNetworks; i++)
                {
                    ListOfNetworks.Add(new MultiLayerPerceptron());
                    ListOfNetworks[i].CreateNetwork(Constants.learningRate, Constants.InputCount, Constants.HiddenLayerNodes, Constants.OutputCount, Constants.Momentum);
                }

                // train and return all created networks
                ListOfNetworks = Trainer.GetTrainedNetworks(TrainingInput, TrainingOutput);

                // Extract network weights from networks
                for (int i = 0; i < Constants.NumNetworks; i++)
                {
                    LinearWeights.Add(ListOfNetworks[i].ExtractWeightList());
                }

                // Classify neural networks according to solution type
                for (int i = 0; i < Constants.NumNetworks; i++)
                {
                    TempLinearWeights = LinearWeights[i];
                    Classifications.Add(LVQ.Execute(TempLinearWeights.ToArray())[0]);
                }

                //sort neural networks by classification
                for (int i = 0; i < Constants.NumNetworks; i++)
                {
                    Classification = Classifications[i];
                    if (Classification == -1)
                    {
                        //do nothing
                    }
                    else
                    {
                        SortedIndices.Add(new List<int>());
                        NumNetworkClassifications++;
                        for (int j = 0; j < Constants.NumNetworks; j++)
                        {
                            if (Classifications[j] == Classification)
                            {
                                SortedIndices[NumNetworkClassifications - 1].Add(j);
                                Classifications[j] = -1;
                            }
                        }
                    }
                }

                //select best netowrk from each catagory
                //for each catagory
                for(int i = 0; i< SortedIndices.Count;i++)
                {
                    //build and evaluate each newtork in that catagory
                    for(int j=0;j < SortedIndices[i].Count;j++)
                    {
                        ToEvaluate.Add(new MultiLayerPerceptron());
                        ToEvaluate[j].CreateNetwork(Constants.learningRate, Constants.InputCount, Constants.HiddenLayerNodes, Constants.OutputCount, Constants.Momentum);
                        ((MultiLayerPerceptron)ToEvaluate[j]).SetWeightsList(LinearWeights[SortedIndices[i][j]]);
                    }
                    //ChosenNetworks.Add(Extensions.BestNetworkFromSet(ToEvaluate, TrainingInput, TrainingOutput));
                    NewWeights.Add((Extensions.BestNetworkFromSet(ToEvaluate, TrainingInput, TrainingOutput,Num_Training).ExtractWeightList()));
                    ToEvaluate.Clear();
                }


                ////merge networks according to classification
                //for (int i = 0; i < SortedIndices.Count; i++)
                //{
                //    NewWeights.Add(new List<double>());
                //    for (int j = 0; j < ((Constants.InputCount + 1) * Constants.HiddenLayerNodes) + ((Constants.HiddenLayerNodes + 1) * Constants.OutputCount); j++)
                //    {
                //        NewWeights[i].Add(0);
                //    }
                //}
                //for (int i = 0; i < SortedIndices.Count; i++)
                //{
                //    for (int j = 0; j < ((Constants.InputCount + 1) * Constants.HiddenLayerNodes) + ((Constants.HiddenLayerNodes + 1) * Constants.OutputCount); j++)
                //    {
                //        //currently just averageing the weights of source networks, easy to change if necessary
                //        for (int k = 0; k < SortedIndices[i].Count; k++)
                //        {
                //            NewWeights[i][j] += LinearWeights[(int)SortedIndices[i][k]][j];
                //        }
                //        NewWeights[i][j] = NewWeights[i][j] / SortedIndices[i].Count;

                //    }
                //}

                // Add new managerNN to the testing array
                M.Add(new ManagerNN(NewWeights, TrainingInput, TrainingOutput));

                //output count is included here to comply with the inherited function format, it is not actually used internally
                M[0].CreateNetwork(Constants.MalearningRate, Constants.MaInputCount, Constants.MaHiddenLayerNodes, 1, Constants.MaMomentum);

                SecondaryTrainer = new NNTrainer(M);

                // Train guidance network and store resultant network
                M2 = SecondaryTrainer.GetTrainedNetworks(TrainingInput, TrainingOutput)[0];

                // evaluate the trained network and print the results
                ManagerTester = new TestingAndReporting(M2, TestingInput, TestingOutput, Constants.ManagerOutputFile);
                ManagerTester.RunTest();

                M.Clear();
                // Same thing as above, just with a normal network rather then a manager network
                // Add new managerNN to the testing array
                M.Add(new MultiLayerPerceptron());

                M[0].CreateNetwork(Constants.learningRate, Constants.InputCount, Constants.HiddenLayerNodes, Constants.OutputCount, Constants.Momentum);

                SecondaryTrainer = new NNTrainer(M);

                // Train guidance network and store resultant network
                M2 = SecondaryTrainer.GetTrainedNetworks(TrainingInput, TrainingOutput)[0];

                // evaluate the trained network and print the results
                NNTester = new TestingAndReporting(M2, TestingInput, TestingOutput, Constants.NNOutputFile);
                NNTester.RunTest();

            }
        }
    }
}
