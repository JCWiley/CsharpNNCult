using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuronGate
{
    class Perceptron
    {

        /// <summary>
        /// Constructs a perceptron with the specified number of inputs, step size, threshold, and starting weight values
        /// </summary>
        /// <param name="in_num_inputs">Number of inputs posessed by new perceptron</param>
        /// <param name="in_step">Learning rate step</param>
        /// <param name="in_threshold">Bias threshold</param>
        /// <param name="input_weights">Starting input weights.  The number of weights passed in must equal the number of inputs specified.</param>
        /// <exception cref="System.ArgumentOutOfRangeException">Thrown if the number of weights given does not equal the number of inputs specified</exception> 
        public Perceptron(int in_num_inputs, float in_step, float in_threshold, params float[] input_weights) 
        {
            if(input_weights.Length != in_num_inputs)
            {
                throw new ArgumentOutOfRangeException("Number of weights passed to Perceptron constructor does not match input count");
            }

            //initialize input and weight arrays
            Inputs = new float[in_num_inputs];
            Weights = new float[in_num_inputs];

            //copy in all given starting weights
            for (int i = 0; i < input_weights.Length; i++)
            {
                Weights[i] = input_weights[i];
            }

            threshold = in_threshold;
            step = in_step;
            numInputs = in_num_inputs;
        }

        /// <summary>
        /// Executes perceptron logic, using specified input values
        /// </summary>
        /// <param name="inputs">Inputs to parse.  If less inputs are passed in than are posessed by the perceptron, subsequent input slots will default to 0</param>
        /// <exception cref="System.ArgumentOutOfRangeException">Thrown if the number of inputs passed in exceeds the input count of the perceptron</exception>
        public void ExecutePerceptron(params float[] inputs)
        {
            if (inputs.Length > numInputs)
            {
                throw new ArgumentOutOfRangeException("Too many inputs passed to ExecutePerceptron");
            }

            float sum = 0;

            
            Inputs.Initialize();
            for (int i = 0; i < inputs.Length; i++)
            {
                //save the inputs to class properties for later use
                Inputs[i] = inputs[i];

                //sum together all input-weight pairs.
                //any inputs not covered by the input array default to 0, and do not contribute to the sum.
                sum += Inputs[i] * Weights[i];
            }

            //run the perceptron algorithm.  Greater than the threshold value returns 1, less than or equal returns 0
            output = (sum > threshold) ? 1 : 0;
        }

        /// <summary>
        /// Adjusts weights based on the expected result, in relation to the current output value
        /// </summary>
        /// <param name="expected">Expected result for current output</param>
        public void AdjustWeights(int expected)
        {
            //adjust each weight
            for (int i = 0; i < numInputs; i++)    
            {
                Weights[i] += step * (expected - output) * Inputs[i];
            }

            threshold += step * (expected - output) * -1;
        }

        #region Properties
        public float[] Inputs { get; private set; }
        public float[] Weights { get; private set; }
        public int output { get; private set; }


        
        public float threshold { get; private set; }
        private readonly float step;
        #endregion

        private int numInputs;



    }
}
