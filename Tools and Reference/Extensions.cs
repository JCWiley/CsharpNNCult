using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using System.Linq;

using Neural_Net_Cultivator;
using Neural_Net_Cultivator.Neural_Nets;
using Neural_Net_Cultivator.Inheritance;

namespace Neural_Net_Cultivator.Tools_and_Reference
{
    static class Extensions
    {
        /// <summary>
        /// Get the array slice between the two indices.
        /// </summary>
        /// <param name="source">The source array.</param>
        /// <param name="start">The start index to use (inclusive).</param>
        /// <param name="end">The end index to use (exclusive).</param>
        /// <returns>Array slice</returns>
        public static T[] Slice<T>(this T[] source, int start, int end)
        {
            // Handles negative ends.
            if (end < 0)
            {
                end = source.Length + end;
            }
            int len = end - start;

            // Return new array.
            T[] res = new T[len];
            for (int i = 0; i < len; i++)
            {
                res[i] = source[i + start];
            }
            return res;
        }

        public static void LoadData(ref List<List<double>> TrainingInput, ref List<List<double>> TrainingOutput, ref List<List<double>> TestingInput, ref List<List<double>> TestingOutput)
        {
            //store the data in the text document as an array of strings
            string[] TrainingData = System.IO.File.ReadAllLines(Constants.DataFileName);
            string temp;

            List<double> Temp = new List<double>();
            List<int> RandomIndex = new List<int>();

            List<List<double>> Inputs = new List<List<double>>();
            List<List<double>> Outputs = new List<List<double>>();

            List<double> TempInputs = new List<double>();
            List<double> TempOutputs = new List<double>();

            //for each string in training data, ignoring the last one because it is null
            for (int i = 0; i < TrainingData.GetLength(0); i++)
            {
                //grab characters from training data in groups of 4, that being the number of characters to represent a single decimal double - 1.5 plus space
                for (int j = 0; j < Constants.InputCount * Constants.InputNumDecimal + 1; j += Constants.InputNumDecimal + 1)
                {
                    //grab the chars starting at location j and continuing for the length of the current number
                    temp = TrainingData[i].Substring(j, Constants.InputNumDecimal);
                    //parse the grabbed string into the coresponding double value
                    TempInputs.Add(double.Parse(temp, System.Globalization.NumberStyles.Any));
                }
                // grab characters starting from the end of the input section and continueing till then end of the output section
                for (int j = (Constants.InputCount * Constants.InputNumDecimal) + Constants.InputCount; j < TrainingData[i].Count(); j += Constants.OutputNumDecimal+1)
                {
                    //grab data starting from j and continuing the length of the number
                    temp = TrainingData[i].Substring(j, Constants.OutputNumDecimal);
                    // parse the grabbed data into doubles
                    TempOutputs.Add(double.Parse(temp, System.Globalization.NumberStyles.Any));
                }
                //append the new inputs and outputs to the appropriate storage variables
                Inputs.Add(new List<double>(TempInputs));
                Outputs.Add(new List<double>(TempOutputs));

                //clear inputs and outputs so they can be used to store the next values on the next loop
                TempInputs.Clear();
                TempOutputs.Clear();
            }
            //create an array storing the index of each input output pair
            for (int i = 0; i < TrainingData.GetLength(0); i++)
            {
                RandomIndex.Add(i);
            }
            //shuffle the random index array, this is to ease splitting the inputs and output pairs into training and testing sets
            Shuffle<int>(RandomIndex);

            //copy input output pairs into the training input and output arrays depending on the order determined in the random index
            for (int i = 0; i < Constants.NumTraining; i++)
            {
                TrainingInput.Add(Inputs[RandomIndex[i]]);
                TrainingOutput.Add(Outputs[RandomIndex[i]]);
            }
            //copy input output pairs into the testing input and output arrays depending on the order determined in the random index
            for (int i = Constants.NumTraining; i < Constants.NumVectors; i++)
            {
                TestingInput.Add(Inputs[RandomIndex[i]]);
                TestingOutput.Add(Outputs[RandomIndex[i]]);
            }
        }
        //shuffle a templated list of information, there is probably a prewritten function to do this, but i didnt manage to find it
        public static void Shuffle<T>(this IList<T> list)
        {
            Random rng = new Random();
            int n = list.Count;
            while (n > 1)
            {
                n--;
                int k = rng.Next(n + 1);
                T value = list[k];
                list[k] = list[n];
                list[n] = value;
            }
        }
        //check the capabilities of a single network by calculating the cumulative error for a single input output set
        public static double CompareNetworks(IBaseNeuralNetwork Network, List<double> Input, List<double> DesiredOutput)
        {
            double[] ActualOutput;
            double Error = 0;
            ActualOutput = Network.Execute(Input.ToArray());
            for (int i = 0; i < ActualOutput.Count(); i++)
            {
                Error += Math.Abs(ActualOutput[i] - DesiredOutput[i]);
            }
            return Error;
        }

        public static MultiLayerPerceptron BestNetworkFromSet(List<IBaseNeuralNetwork> Networks,List<List<double>> Input,List<List<double>> DesiredOutput,int NumTraining)
        {
            List<double> Scores = new List<double>();
            double SumError = 0;
            int IdealNetworkIndex = -1;
            for (int i = 0; i< Networks.Count;i++)
            {
                for(int j = 0;j< NumTraining;j++)
                {
                    SumError += CompareNetworks(Networks[i], Input[j], DesiredOutput[j]);
                }
                Scores.Add(SumError / NumTraining);
                SumError = 0;
            }
            IdealNetworkIndex = Array.IndexOf(Scores.ToArray(), Scores.Min());

            return (MultiLayerPerceptron)Networks[IdealNetworkIndex];
        }

        public static List<double> CalculateError(List<double> Desired,List<Double> Actual)
        {
            List<double> Error = new List<double>();
            List<double> TempDesired = new List<double>();
            for(int i = 0; i< Desired.Count();i++)
            {
                if (Constants.ErrorVersion == 0)
                {
                    //Error.Add(Actual[i] - Desired[i]);
                    Error.Add(Desired[i] - Actual[i]);
                }
                if (Constants.ErrorVersion == 1) // iris specific error calculation
                {
                    foreach (double x in Desired)
                    {
                        if (x == 0)
                        {
                            TempDesired.Add(-1);
                        }
                        else
                        {
                            TempDesired.Add(x);
                        }
                    }
                    //Error.Add(TrainingOutputs[i][j] * Math.Log(TrainingResults[j]) + (1 - TrainingOutputs[i][j]) * Math.Log(1 - TrainingResults[j]));
                    //Error.Add((Desired[i] * Math.Log(Actual[i]) + (1 - Desired[i]) * Math.Log(1 - Actual[i])));
                    Error.Add(-(TempDesired[i] * Math.Log(Actual[i]) + (1 - TempDesired[i]) * Math.Log(1 - Actual[i])));
                }
            }
            return Error;
        }
    }

}