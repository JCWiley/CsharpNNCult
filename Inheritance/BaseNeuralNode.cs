using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_Net_Cultivator.Inheritance
{
    [Serializable]
    abstract class BaseNeuralNode : IBaseNeuralNode
    {
        #region IBaseNeuralNode Members

        #region Abstract members
        /// <summary>
        /// Executes perceptron logic, using specified input values
        /// </summary>
        /// <param name="inputs">Inputs to parse.</param>
        /// <returns></returns>
        public abstract double Execute(params double[] inputs);

        /// <summary>
        /// Executes perceptron logic, using currently stored input values
        /// </summary>
        /// <returns></returns>
        public abstract double Execute();
 
        /// <summary>
        /// Adjusts weights based on the specified error, in relation to the current output value
        /// </summary>
        /// <param name="error">Calculated error for the current output</param>
        /// <param name="momentum">Momentum to use relative to previous weight change.  Defaults to 0 (no momentum)</param>
        public abstract void AdjustWeights(double error, double momentum = 0);
        #endregion

        /// <summary>
        /// Assigns input values to the neural node.
        /// </summary>
        /// <param name="in_inputs">Inputs to parse.  If less inputs are passed in than are posessed by the perceptron, subsequent input slots will default to 0</param>
        /// <exception cref="System.ArgumentOutOfRangeException">Thrown if the number of inputs passed in exceeds the input count of the perceptron</exception>
        public virtual void SetInputs(params double[] in_inputs)
        {
            if (in_inputs.Length > Inputs.Length)
            {
                throw new ArgumentOutOfRangeException("Too many inputs passed to SetInputs");
            }

            Inputs.Initialize();
            for (int i = 0; i < in_inputs.Length; i++)
            {
                Inputs[i] = in_inputs[i];
            }
        }




        #region Interface Properties
        public virtual double[] Inputs { get; protected set; }
        public virtual double[] Weights { get; protected set; }
        public virtual int NumInputs //TODO:: Determine if this is actually needed.  Inputs.Length works just fine.
        {
            get { return Inputs.Length; }
        }
        public virtual double Sum { get; protected set; }
        public virtual double Output { get; protected set; }
        public virtual double Bias { get; set; } //TODO:: re-private setter once testing has concluded
        #endregion

        public virtual Activation_Functions.IActivationFunction Activation { get; set; }

        #endregion
    }
}
