using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Neural_Net_Cultivator.Inheritance;

namespace Neural_Net_Cultivator.Tools_and_Reference
{
    class TestingAndReporting
    {
        public TestingAndReporting(IBaseNeuralNetwork NetworkToEvaluate,List<List<double>> TestingInput, List<List<double>> TestingOutput, string IncFilename)
        {
            Input = TestingInput;
            Output = TestingOutput;
            Network = NetworkToEvaluate;
            filename = IncFilename;
        }
        public void RunTest()
        {
            List<List<double>> ActualOutput = new List<List<double>>();
            List<List<double>> ActualOutputHigh = new List<List<double>>();
            List<List<double>> error = new List<List<double>>();

            for (int i = 0; i < Input.Count();i++)
            {
                ActualOutput.Add(new List<double>(Network.Execute(Input[i].ToArray())));
                ActualOutputHigh.Add(new List<double>());
                for (int j = 0; j < ActualOutput[i].Count();j++)
                {
                    ActualOutputHigh[i].Add(0);
                }

                ActualOutputHigh[i][ActualOutput[i].IndexOf(ActualOutput[i].Max())] = 1;

                error.Add(Extensions.CalculateError(Output[i], ActualOutput[i]));
            }
            //append error to file
            using (System.IO.StreamWriter file = System.IO.File.AppendText(filename))
            //using (System.IO.StreamWriter file = System.IO.File.CreateText(filename))
            {
                for (int i = 0; i < Output.Count(); i++)
                {
                    for (int j = 0; j < Output[i].Count(); j++)
                    {
                        file.Write(Output[i][j]);
                        file.Write(' ');
                    }
                    for (int j = 0; j < Output[i].Count(); j++)
                    {
                        file.Write(ActualOutput[i][j].ToString("F10"));
                        file.Write(' ');
                    }
                    for (int j = 0; j < Output[i].Count(); j++)
                    {
                        file.Write(ActualOutputHigh[i][j]);
                        file.Write(' ');
                    }
                    //for (int j = 0; j < Output[i].Count(); j++)
                    //{
                    //    file.Write(error[i][j].ToString("F10"));
                    //    file.Write(' ');
                    //}
                    file.Write("\r\n");
                }
                file.Write("\r\n");
                file.Close();
            }
        }
        List<List<double>> Input;
        List<List<double>> Output;
        IBaseNeuralNetwork Network;
        string filename;
    }
}
