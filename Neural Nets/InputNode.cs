using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_Net_Cultivator.Neural_Nets
{
    [Serializable]
    class InputNode : Inheritance.BaseNeuralNode
    {
        #region Constructors
        /// <summary>
        /// Initializes a new instance of the <see cref="InputNode"/> class.  Input defaults to 0.
        /// </summary>
        public InputNode() : this(0){ }

        /// <summary>
        /// Initializes a new instance of the <see cref="InputNode"/> class with the specified input value.
        /// </summary>
        /// <param name="input">The input value for the node.</param>
        public InputNode(double input)
        {
            // Input node supports single input
            Inputs = new double[1];
            Inputs[0] = input;
            // Output equals input for Input Node
            Output = input;

            //not used in Input Nodes
            Weights = null;
            Bias = 1;
        }

        #endregion

        #region Interface methods

        /// <summary>
        /// Direct passthrough returning input value.
        /// </summary>
        /// <param name="inputs">Input to parse.</param>
        /// <returns></returns>
        public override double Execute(params double[] inputs)
        {
            if(inputs.Length != 1)
            {
                throw new ArgumentOutOfRangeException
                    (inputs.Length.ToString() + " inputs passed to InputNode.Execute. Input nodes support only single-input.");
            }

            // Input node execution is a passthrough returning the input
            SetInputs(inputs[0]);

            return Output;
        }

        /// <summary>
        /// Direct passthrough returning currently stored input value.
        /// </summary>
        public override double Execute()
        {
            // passthrough to array-arg overload
            return Execute(Inputs);
        }

        /// <summary>
        /// Not a supported feature of Input Nodes.  An exception will be thrown.
        /// </summary>
        /// <exception cref="System.NotSupportedException"> Input nodes do not support weight operations.</exception>
        public override void AdjustWeights(double error, double momentum = 0)
        {
            throw new NotSupportedException("AdjustWeights called on InputNode.  Input nodes do not support weight operations.");
        }


        /// <summary>
        /// Assigns input value to the neural node.
        /// </summary>
        /// <param name="in_inputs">Input to parse (Input Node requires a singular input)</param>
        /// <exception cref="System.ArgumentOutOfRangeException">Thrown if input count is not singular</exception>
        public override void SetInputs(params double[] in_inputs)
        {
            if (in_inputs.Length != 1)
            {
                throw new ArgumentOutOfRangeException
                    (in_inputs.Length.ToString() + " inputs passed to InputNode.SetInputs. Input nodes support only single-input.");
            }

            Inputs[0] = in_inputs[0];
        }

        #endregion

        #region Override properties
        public override double Output
        {
            get
            {
                //Input node's output is always equal to the input
                return Inputs[0];
            }
        }
        public override double Sum
        {
            get { throw new NotSupportedException("Sum property not supported by InputNode."); }
            protected set { throw new NotSupportedException("Sum property not supported by InputNode."); }
        }
        public override Activation_Functions.IActivationFunction Activation
        {
            get { throw new NotSupportedException("Activation operations not supported by InputNode."); }
            set { throw new NotSupportedException("Activation operations not supported by InputNode."); }
        }
        #endregion
    }
}
