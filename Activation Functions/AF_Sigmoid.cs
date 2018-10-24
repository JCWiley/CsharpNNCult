using System;

namespace Neural_Net_Cultivator.Activation_Functions
{
    [Serializable]
    class AF_Sigmoid : IActivationFunction
    {
        //http://mathworld.wolfram.com/SigmoidFunction.html
        #region IActivationFunction Members

        /// <summary>
        /// Executes the log-sigmoid function on the given input.
        /// </summary>
        /// <param name="input">Value to be input to the sigmoid function</param>
        /// <returns>
        /// Result of sigmoid execution.
        /// </returns>
        public double Execute(double input)
        {
            return (1 / (1 + Math.Exp(-input)));
        }

        /// <summary>
        /// Executes the derivative of the log-sigmoid function on the given input
        /// </summary>
        /// </summary>
        /// <param name="input">Value to be input to the log-sigmoid derivative function</param>
        /// <returns>
        /// Result of derivative execution.
        /// </returns>
        public double Derivative(double input)
        {
            double exp = Math.Exp(input);

            return (exp / Math.Pow((1 + exp), 2));
        }

        #endregion
    }
}
