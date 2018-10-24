
namespace Neural_Net_Cultivator.Activation_Functions
{

    interface IActivationFunction
    {
        /// <summary>
        /// The internal equation used to determine the perceptron's output.
        /// </summary>
        /// <param name="input">Value to be input to the equation</param>
        /// <param name="threshold">The threshold value for the function.</param>
        /// <returns>
        /// Result of equation execution
        /// </returns>
        double Execute(double input);

        /// <summary>
        /// The derivative function corresponding to this activation equation
        /// </summary>
        /// <param name="input">Value to be input to the equation</param>
        /// <returns>Result of equation execution</returns>
        double Derivative(double input);
    }
}
