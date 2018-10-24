using System;

namespace Neural_Net_Cultivator.Activation_Functions
{
    [Serializable]
    class AF_HyperbolicTangent : IActivationFunction
    {
        #region IActivationFunction Members

        /// <summary>
        /// Executes the hyperbolic tangent function on the given input.
        /// </summary>
        /// <param name="input">Value to be input to the hyperbolic tangent function</param>
        /// <returns>
        /// Result of hyperbolic tangent execution.
        /// </returns>
        public double Execute(double input)
        {
            return Math.Tanh(input);
        }

        /// <summary>
        /// Executes the derivative of the hyperbolic tangent function on the given input
        /// </summary>
        /// </summary>
        /// <param name="input">Value to be input to the hyperbolic tangent derivative function</param>
        /// <returns>
        /// Result of derivative execution.
        /// </returns>
        public double Derivative(double input)
        {
            double exp = Math.Tanh(input);

            return 1 - Math.Pow(exp, 2);
        }

        #endregion
    }
}
