using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_Net_Cultivator.Tools_and_Reference
{
    static class DebugBox
    {
        /// <summary>
        /// Compares two identical networks
        /// </summary>
        /// <param name="skip">display output every n'th iteration</param>
        public static void CompareNetworks(int skip)
        {

            double output;

            Inheritance.IBaseNeuralNetwork JohnNet = Inheritance.BaseNeuralNetwork.LoadNN("..\\..\\Matched_ThreeLayerJohn.nn");
            Inheritance.IBaseNeuralNetwork MLPNet = Inheritance.BaseNeuralNetwork.LoadNN("..\\..\\Matched_MultiLayerPerceptron.nn");

            Dictionary<Tuple<int,int,int,int>,double> johnDict = JohnNet.ExtractWeights();
            Dictionary<Tuple<int,int,int,int>,double> mlpDict = MLPNet.ExtractWeights();

            

            #region Create matching Nets
            /*Inheritance.IBaseNeuralNetwork MLPNet = new Neural_Nets.MultiLayerPerceptron();
            Inheritance.IBaseNeuralNetwork JohnNet = new Neural_Nets.ThreeLayerPerceptron_John(System.Guid.NewGuid().GetHashCode());

            JohnNet.CreateNetwork(Constants.learningRate, 2, Constants.HiddenLayerNodes, 1, Constants.incMomentum);

            MLPNet.CreateNetwork(Constants.learningRate, 2, Constants.HiddenLayerNodes, 1, Constants.incMomentum);

            ((Neural_Nets.MultiLayerPerceptron)MLPNet).SetWeights(JohnNet.ExtractWeights());

            MLPNet.SaveNN("Matched_MultiLayerPerceptron.nn");
            JohnNet.SaveNN("Matched_ThreeLayerJohn.nn");*/
            #endregion

            //let's see if we can train an AND gate
            try
            {

                for (int i = 0; i < 10000; i++)
                {
                    /*
                    for (int j = 0; j < 10; j++)
                    {
                        output = net.Execute(0, 0)[0];
                        net.SetError(true, 0-output);
                    }
                    output = net.Execute(1, 1)[0];
                    net.SetError(true, 1 - output);
                    */
                    if (skip <= 1 || i % skip == 0)
                        Console.WriteLine("\n\n----------Iteration " + i.ToString() + "--------------------\n\n");

                    if (skip <= 1 || i % skip == 0)
                        Console.WriteLine("\n~(0,0) output~");
                    output = JohnNet.Execute(0, 0)[0];
                    JohnNet.SetError(true, 1 - output);
                    if (skip <= 1 || i % skip == 0)
                        Console.WriteLine("JohnNet: " + output.ToString());
                    output = MLPNet.Execute(0, 0)[0];
                    MLPNet.SetError(true, 1 - output);
                    if (skip <= 1 || i % skip == 0)
                        Console.WriteLine("MLPNet: " + output.ToString());

                    if (skip <= 1 || i % skip == 0)
                        Console.WriteLine("\n~(0,1) output~");
                    output = JohnNet.Execute(0, 1)[0];
                    JohnNet.SetError(true, 0 - output);
                    if (skip <= 1 || i % skip == 0)
                        Console.WriteLine("JohnNet: " + output.ToString());
                    output = MLPNet.Execute(0, 1)[0];
                    MLPNet.SetError(true, 0 - output);
                    if (skip <= 1 || i % skip == 0)
                        Console.WriteLine("MLPNet: " + output.ToString());

                    if (skip <= 1 || i % skip == 0)
                        Console.WriteLine("\n~(1,0) output~");
                    output = JohnNet.Execute(1, 0)[0];
                    JohnNet.SetError(true, 0 - output);
                    if (skip <= 1 || i % skip == 0)
                        Console.WriteLine("JohnNet: " + output.ToString());
                    output = MLPNet.Execute(1, 0)[0];
                    MLPNet.SetError(true, 0 - output);
                    if (skip <= 1 || i % skip == 0)
                        Console.WriteLine("MLPNet: " + output.ToString());

                    if (skip <= 1 || i % skip == 0)
                        Console.WriteLine("\n~(1,1) output~");
                    output = JohnNet.Execute(1, 1)[0];
                    JohnNet.SetError(true, 1 - output);
                    if (skip <= 1 || i % skip == 0)
                        Console.WriteLine("JohnNet: " + output.ToString());
                    output = MLPNet.Execute(1, 1)[0];
                    MLPNet.SetError(true, 1 - output);
                    if (skip <= 1 || i % skip == 0)
                        Console.WriteLine("MLPNet: " + output.ToString());

                    if (skip <= 1 || i % skip == 0)
                        Console.ReadKey(true);
                }

            }
            catch (Exception e)
            {

                Console.WriteLine(e.Message);
            }
        }

        public static void CreateIdenticalNets(string filenamePrefix = "Matched_")
        {

            Inheritance.IBaseNeuralNetwork JohnNet = new Neural_Nets.ThreeLayerPerceptron_John(Guid.NewGuid().GetHashCode());
            Inheritance.IBaseNeuralNetwork MLPNet = new Neural_Nets.MultiLayerPerceptron();

            JohnNet.CreateNetwork(Constants.learningRate, 2, Constants.HiddenLayerNodes, 1, Constants.Momentum);
            MLPNet.CreateNetwork(Constants.learningRate, 2, Constants.HiddenLayerNodes, 1, Constants.Momentum);

            ((Neural_Nets.MultiLayerPerceptron)MLPNet).SetWeights(JohnNet.ExtractWeights());


            JohnNet.SaveNN("..\\..\\" + filenamePrefix + "ThreeLayerJohn.nn");
            MLPNet.SaveNN("..\\..\\" + filenamePrefix + "MultiLayerPerceptron.nn");



        }
    }
}
