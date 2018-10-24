
namespace Neural_Net_Cultivator.Tools_and_Reference
{
    class Constants
    {
        #region Iris Settings
        public const string DataFileName = @"C:\Users\John\Documents\Visual Studio 2015\Projects\AI\OIT-ML-2014-NNC\Neural Net Cultivator\Data and Results\iris.txt";
        public const string ManagerOutputFile = @"C:\Users\John\Documents\Visual Studio 2015\Projects\AI\OIT-ML-2014-NNC\Neural Net Cultivator\Data and Results\irisManagerResult.txt";
        public const string NNOutputFile = @"C:\Users\John\Documents\Visual Studio 2015\Projects\AI\OIT-ML-2014-NNC\Neural Net Cultivator\Data and Results\irisNNResult.txt";

        public const int NumTraining = 120; // the sum of the NumTraining and NumTesting values must equal NumVectors
        public const int NumTesting = 30;
        public const int NumVectors = 150;
        public const int NumTrials = 2;
        public const int ErrorVersion = 0; //0 for standard, 1 for iris specific
        public const int InputNumDecimal = 3; // length of an individual input, for example, 1.5 is length 3 as there are 3 characters
        public const int OutputNumDecimal = 1; // same as above, just with outputs

        //Conventional network consants
        public const double learningRate = .7;
        public const double Momentum = .1;

        public const int InputCount = 4;
        public const int HiddenLayers = 1;
        public const int HiddenLayerNodes = 4;
        public const int OutputCount = 3;
        public const int NumTrainingCycles = 100;
        public const int NumNetworks = 20;


        // Manager network constants
        public const double PosReinforcement = .3;
        public const double NegReinforcemnt = -.3;

        public const double MalearningRate = .7;
        public const double MaMomentum = .1;

        public const int MaInputCount = Constants.InputCount; // needs to equal conventional input count as they operate on the same problems
        public const int MaHiddenLayers = 1;
        public const int MaHiddenLayerNodes = 4; // does not need to equal that of the normal networks
        #endregion

        #region car acceptability settings
        ////dataset https://archive.ics.uci.edu/ml/datasets/Car+Evaluation

        //public const string DataFileName = @"C:\Users\John\Documents\Visual Studio 2015\Projects\AI\OIT-ML-2014-NNC\Neural Net Cultivator\Data and Results\Cars.txt";
        //public const string ManagerOutputFile = @"C:\Users\John\Documents\Visual Studio 2015\Projects\AI\OIT-ML-2014-NNC\Neural Net Cultivator\Data and Results\carManagerResult.txt";
        //public const string NNOutputFile = @"C:\Users\John\Documents\Visual Studio 2015\Projects\AI\OIT-ML-2014-NNC\Neural Net Cultivator\Data and Results\carNNResult.txt";

        //public const int NumTraining = 1378; // the sum of the NumTraining and NumTesting values must equal NumVectors
        //public const int NumTesting = 350;
        //public const int NumVectors = 1728;
        //public const int NumTrials = 1;
        //public const int ErrorVersion = 0; //0 for standard, 1 for iris specific
        //public const int InputNumDecimal = 1; // length of an individual input, for example, 1.5 is length 3 as there are 3 characters
        //public const int OutputNumDecimal = 1; // same as above, just with outputs

        //// Conventional network consants
        //public const double learningRate = .3;
        //public const double Momentum = .1;

        //public const int InputCount = 21;
        //public const int HiddenLayers = 1;
        //public const int HiddenLayerNodes = 10;
        //public const int OutputCount = 4;
        //public const int NumTrainingCycles = 200;
        //public const int NumNetworks = 20;

        //// Manager network constants
        //public const double PosReinforcement = .3;
        //public const double NegReinforcemnt = -.3;

        //public const double MalearningRate = .3;
        //public const double MaMomentum = .1;

        //public const int MaInputCount = Constants.InputCount; // needs to equal conventional input count as they operate on the same problems
        //public const int MaHiddenLayers = 1;
        //public const int MaHiddenLayerNodes = 4; // does not need to equal that of the normal networks
        #endregion



        //public const int MaOutputCount           //output count is calculated dynamicly based on the number of distinct solutions found by the system

        //LVQ constants
        public const double LVQLearningRate   = .8;
        public const int LVQHiddenLayer       = 0; //always 0 because the LVQ does not have a hidden layer
        public const int LVQOutputCount       = 5;
        public const double LVQMomentum       = .2;
    }
}
