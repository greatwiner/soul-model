/* Class for tanh activation function and must be used with an other
 Module. In this class, we only require the code for forward and
 backward. This class only stores gradInput and output but does not
 store the weight and biases.
 */

class Tanh : public Module
{
public:
  Tanh(int size0, int size1);
  ~Tanh();
  void
  changeBlockSize(int blockSize);
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
