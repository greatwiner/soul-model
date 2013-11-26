/******************************************************************
 Structure OUtput Layer (SOUL) Language Model Toolkit
 (C) Copyright 2009 - 2012 Hai-Son LE LIMSI-CNRS

 Abstract class for a layer. For linear layers: The weights, biases to
 estimate  the layers' activities
 To summarize the training procedure:
 - forward()
 - backward()
 - updateParameters()
 *******************************************************************/

class Module
{
public:
  // Name or type of modules
  string name;

  // Weight matrix to estimate the activities
  floatTensor weight;

  // Bias vector
  floatTensor bias;

  // Weight decay: Hyper parameter for update function
  float weightDecay;

  // Use for backward pass: contains the gradient for the previous
  // layer
  floatTensor gradInput;

  // Pointer used for  backward and update parameters of this layer
  // It points to the gradInput of the next layer
  floatTensor gradOutput;

  // Output values for this layer
  floatTensor output;

  // for the bunch mode
  int blockSize;
  // Only for recurrent
  int firstTime;
  intTensor iContext;

  // For RLinear, RRLinear, WRLinear: modules for recurrent like models
  floatTensor vectorInput;

  Module();
  virtual
  ~Module();

  // Change block size of bunch mode training, need to resize some tensors
  virtual void
  changeBlockSize(int blockSize) = 0;

  // Forward propagation of the input floatTensor
  // Example for the linear case:
  // the floatTensor output is the input is multiplied by weight, added
  // by bias
  virtual floatTensor&
  forward(floatTensor& input) = 0;

  // Backward propagation: collect sufficient statistics (update the
  // parameters of gradient: gradOutput) for the further update
  // (to be used in updateParameters)
  // Input must be the same forward, and gradOutput is the gradient
  // coming from the next layer (for the last layer, it is the gradient
  // of the cost function).
  virtual floatTensor&
  backward(floatTensor& gradOutput) = 0;

  // After the forward/backward pass, this method update the
  // parameters of the layer according to the statistics gathered
  // during the backward pass.

  virtual void
  updateParameters(float learningRate) = 0;

  // Read and write with fiels
  virtual void
  read(ioFile *iof) = 0;
  virtual void
  write(ioFile * iof) = 0;

  // Tie weight with other tensor (only for log bilinear
  void
  shareWeight(floatTensor& weight);
  // If we need to output something
  ioFile infoIof;

  virtual float
  distance2(Module& anotherModule) = 0;
};

