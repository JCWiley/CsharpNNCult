using System;

namespace Neural_Net_Cultivator.Activation_Functions
{
    [Serializable]
    class AF_Step : IActivationFunction
    {

        #region IActivationFunction Members

        /// <summary>
        /// Executes a step function on the given input.
        /// Greater than 0 results in activation, 0 or less results in inhibition.
        /// </summary>
        /// <param name="input">Value to be input to the step function</param>
        /// <returns>
        /// Binary output, 1 or 0
        /// </returns>
        public double Execute(double input)
        {
            return (input > 0 ) ? 1 : 0;
        }

        /// <summary>
        /// Derivative not supported for Step equation -- non-continuous function.
        /// </summary>
        /// <exception cref="System.NotSupportedException">Derivative not supported for Step function (multi-layer nets not allowed)</exception>
        public double Derivative(double input)
        {
            throw new NotSupportedException("Derivative not supported for Step function (multi-layer nets not allowed)");
        }

        #endregion
    }
}
