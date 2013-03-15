/* Specific module used to build a FunctionSequential network.  The methods
 forward, backward, and updateParameters only call FunctionSequentially the
 corresponding methods of each module. The add method aims to add a
 new module, and the last added module is the output (the first
 added module is the input of the network).
 */

class FunctionSequential
{
public:
  floatTensor gradInput; // use for backward pass: contains the gradient to
  // update the previous connected layers.
  floatTensor output; // output values for the layers: may be computed with the weights
  int blockSize; // for the bunch mode

  int size; // the number of modules without lookuptable
  Module** modules; // the arrays of modules except lookuptable
  floatTensor currentOutput;
  floatTensor currentGradOutput;

  FunctionSequential(int maxSize);
  ~FunctionSequential();
  void
  changeBlockSize(int blockSize);
  //floatTensor gradInput;
  //floatTensor output;
  void
  add(Module* module);
  floatTensor&
  forward(floatTensor& input);
  floatTensor&
  backward(floatTensor& gradOutput);
  void
  updateParameters(float learningRate);
  void
  read(ioFile *iof);
  void
  write(ioFile * iof);
};

