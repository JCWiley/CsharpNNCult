using Neural_Net_Cultivator.Activation_Functions;
using System;

namespace Neural_Net_Cultivator.Neural_Nets
{
    [Serializable]
    class Perceptron : Inheritance.BaseNeuralNode
    {
        #region Constructors
        /// <summary>
        /// Constructs a perceptron with the specified number of inputs, learning algorithm, step size, threshold, and starting weight values
        /// </summary>
        /// <param name="in_numInputs">Number of inputs posessed by new perceptron</param>
        /// <param name="in_activation">The in_activation.</param>
        /// <param name="in_learningRate">Learning rate step</param>
        /// <param name="in_bias">The bias used to adjust the activation function.</param>
        /// <param name="input_weights">Starting input weights.  The number of weights passed in must equal the number of inputs specified.</param>
        /// <exception cref="System.ArgumentOutOfRangeException">Thrown if the number of weights given does not equal the number of inputs specified</exception>
        public Perceptron(int in_numInputs, IActivationFunction in_activation, float in_learningRate, double in_bias, params double[] input_weights) 
        {
            if(input_weights.Length != in_numInputs)
            {
                throw new ArgumentOutOfRangeException("Number of weights passed to Perceptron constructor does not match input count");
            }
            //assign internal algorithm
            Activation = in_activation;

            #region Variable initializations
            Inputs = new double[in_numInputs];
            Weights = new double[in_numInputs];
            prevWeightsDelta = new double[in_numInputs];
            Inputs.Initialize();
            prevWeightsDelta.Initialize();
            Sum = 0;
            Output = 0;
            #endregion

            //copy in all given starting weights
            for (int i = 0; i < input_weights.Length; i++)
            {
                Weights[i] = input_weights[i];
            }

            Bias = in_bias;
            learningRate = in_learningRate;
        }

        /// <summary>
        /// Constructs a perceptron with the specified number of inputs, learning algorithm, step size, and threshold
        ///  Starting weight values are randomized.
        /// </summary>
        /// <param name="in_numInputs">Number of inputs posessed by new perceptron</param>
        /// <param name="in_learningRate">Learning rate step.  Default value is 0.7.</param>
        public Perceptron(int in_numInputs, IActivationFunction in_activation, float in_learningRate = 0.7F)
        {
            //assign internal algorithm
            Activation = in_activation;

            //Create a random generator variable, seeded with a new globally unique hash.
            Random rand = new Random(Guid.NewGuid().GetHashCode()); ;

            #region Variable initialization
            Inputs = new double[in_numInputs];
            Weights = new double[in_numInputs];
            prevWeightsDelta = new double[in_numInputs];
            Inputs.Initialize();
            prevWeightsDelta.Initialize();
            Sum = 0;
            Output = 0;
            #endregion

            // Assign random weights
            for (int i = 0; i < in_numInputs; i++)
            {
                // set each weight to a random number between -1 and 1
                Weights[i] = (rand.NextDouble() * 2) - 1;
            }

            //assign the rest of the variables
            Bias = (rand.NextDouble() * 2) - 1;
            learningRate = in_learningRate;
        }

        /// <summary>
        /// Passthrough constructor to allow default learning algorithm (Step function).
        /// Constructs a perceptron with the specified number of inputs, step size, and threshold
        ///  Starting weight values are randomized.
        /// </summary>
        /// <param name="in_numInputs">Number of inputs posessed by new perceptron</param>
        /// <param name="in_learningRate">Learning rate step.  Default value is 0.7.</param>
        public Perceptron(int in_numInputs, float in_learningRate = 0.7F)
            : this(in_numInputs, new AF_Step(), in_learningRate){}
        #endregion

        #region Interface members
        /// <summary>
        /// Executes perceptron logic, using specified input values
        /// </summary>
        /// <param name="in_inputs">Inputs to parse.  If less inputs are passed in than are posessed by the perceptron, subsequent input slots will default to 0</param>
        /// <exception cref="System.ArgumentOutOfRangeException">Thrown if the number of inputs passed in exceeds the input count of the perceptron</exception>
        public override double Execute(params double[] in_inputs)
        {
            if (in_inputs.Length > Inputs.Length)
            {
                throw new ArgumentOutOfRangeException("Too many inputs passed to ExecutePerceptron");
            }
            // Reset sum
            Sum = 0;

            
            Inputs.Initialize();
            for (int i = 0; i < in_inputs.Length; i++)
            {
                //save the inputs to class properties for later use
                Inputs[i] = in_inputs[i];

                //sum together all input-weight pairs.
                //any inputs not covered by the input array default to 0, and do not contribute to the sum.
                Sum += Inputs[i] * Weights[i];
            }

            Sum += Bias;

            //run the perceptron algorithm.
            Output = Activation.Execute(Sum);

            return Output;
        }

        /// <summary>
        /// Executes perceptron logic, using last saved input values (default array of 0s from c'tor)
        /// </summary>
        public override double Execute()
        {
            // passthrough to array-arg overload
            return Execute(Inputs);
        }



        /// <summary>
        /// Adjusts weights based on the expected result, in relation to the current output value
        /// </summary>
        /// <param name="error">Calculated error for the current output</param>
        /// <param name="momentum">Momentum to use relative to previous weight change.  Defaults to 0 (no momentum)</param>
        public override void AdjustWeights(double error, double momentum = 0)
        {
            //adjust each weight
            for (int i = 0; i < Inputs.Length; i++)    
            {
                //save the weight change for use in the next calculation
                prevWeightsDelta[i] = learningRate * error * Inputs[i] + (prevWeightsDelta[i] * momentum);
                Weights[i] += prevWeightsDelta[i];
            }

            Bias += learningRate * error;
        }
        #endregion


        #region Members
        private readonly float learningRate;
        private double[] prevWeightsDelta;
        #endregion




    }
}
