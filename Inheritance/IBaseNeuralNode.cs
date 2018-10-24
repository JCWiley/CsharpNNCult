using Neural_Net_Cultivator.Activation_Functions;

namespace Neural_Net_Cultivator.Inheritance
{
    interface IBaseNeuralNode
    {
        /// <summary>
        /// Executes perceptron logic, using specified input values
        /// </summary>
        /// <param name="inputs">Inputs to parse.</param>
        double Execute(params double[] inputs);

        /// <summary>
        /// Executes perceptron logic, using currently stored input values
        /// </summary>
        double Execute();

        /// <summary>
        /// Assigns input values to the perceptron.
        /// </summary>
        /// <param name="inputs">Inputs to parse.</param>
        void SetInputs(params double[] inputs);

        /// <summary>
        /// Adjusts weights based on the specified error, in relation to the current output value
        /// </summary>
        /// <param name="error">Calculated error for the current output</param>
        /// <param name="momentum">Momentum to use relative to previous weight change.  Defaults to 0 (no momentum)</param>
        void AdjustWeights(double error, double momentum = 0);




        #region Properties
        //TODO:: ensure all of these belong in interface
        double[] Inputs { get;  }
        double[] Weights { get;  }
        int NumInputs { get; }  //TODO:: Determine if this is actually needed.  Inputs.Length works just fine.
        double Sum { get; } 
        double Output { get;  }
        double Bias { get; set; }//TODO:: re-private setter once testing has concluded
        IActivationFunction Activation { get; set; }
        #endregion

    }
}
