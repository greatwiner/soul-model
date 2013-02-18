#include "mainModule.H"
LinearSoftmax_Bayes::LinearSoftmax_Bayes(int inputSize, int outputSize, int blockSize,
    outils* otl) : LinearSoftmax(inputSize, outputSize, blockSize, otl)
{
  prevWeight.resize(weight);
  gradWeight.resize(weight);
  prevGradWeight.resize(weight);
  prevBias.resize(bias);
  gradBias.resize(bias);
  prevGradBias.resize(bias);
  pWeight.resize(weight);
  pBias.resize(bias);

  prevWeight.copy(weight);
  gradWeight = 0;
  prevGradWeight = 0;
  gradBias = 0;
  prevGradBias = 0;
}

void
LinearSoftmax_Bayes::changeBlockSize(int blockSize)
{
	this->blockSize = blockSize;
	V1col.resize(blockSize, 1);
	V1col = 1;
	int inputSize = gradInput.size[0];
	int outputSize = output.size[0];
	softmaxVCol.resize(blockSize, 1);
	gradInput.resize(inputSize, blockSize);
	output.resize(outputSize, blockSize);
	gradOutput.resize(output);
	preOutput.resize(outputSize, blockSize);
	prevWeight.resize(weight);
	gradWeight.resize(weight);
	prevGradWeight.resize(weight);
	bias.resize(outputSize, 1);
	prevBias.resize(bias);
	gradBias.resize(bias);
	prevGradBias.resize(bias);
	pWeight.resize(weight);
	pBias.resize(bias);

}

void
LinearSoftmax_Bayes::reset()
{
  weight.uniform(LINEAR_INIT0, LINEAR_INIT1, otl);
  bias.uniform(LINEAR_INIT0, LINEAR_INIT1, otl);
  prevWeight.copy(weight);
  gradWeight = 0;
  prevGradWeight = 0;
  gradBias = 0;
  prevGradBias = 0;
}

floatTensor&
LinearSoftmax_Bayes::backward(floatTensor& word)
{
  cerr << "ERROR: backward of LinearSoftmax for realTensor" << endl;
  exit(1);
}

floatTensor&
LinearSoftmax_Bayes::backward(intTensor& word)
{
  /*
   //gradOutput.copy(output);
   int i;
   this->input = input;
   for (i = 0; i < blockSize; i++)
   {
   if (word(i) != SIGN_NOT_WORD)
   {
   selectGradOutput.select(gradOutput, 1, i);
   selectOutput.select(output, 1, i);
   selectGradOutput.copy(selectOutput);
   selectGradOutput.scal(-1);
   gradOutput(word(i), i) += 1;

   //gradOutput(word(i), i) -= 1;
   }
   else
   {
   selectGradOutput.select(gradOutput, 1, i);
   selectGradOutput = 0;
   }
   }
   gradInput.gemm(weight, 'N', gradOutput, 'N', 1, 0);
   return gradInput;
   */
  // id < -1 => negative example with word
  // = SIGN_NOT_WORD - 1 - id (= -2 - id if SIGN_NOT_WORD = -1)
  // -2 => 0
  // -3 => 1
  // -4 => 2
  // SIGN_NOT_WORD must be <= -1

	// counts until blocksize
  int i;

  // mirror = -2
  int mirror = SIGN_NOT_WORD - 1;
  this->input = input;
  for (i = 0; i < blockSize; i++)
    {
      if (word(i) != SIGN_NOT_WORD)
        {
          if (word(i) >= 0)
            {
              selectGradOutput.select(gradOutput, 1, i);
              selectOutput.select(output, 1, i);
              selectGradOutput.copy(selectOutput);
              gradOutput(word(i), i) -= 1;
        	  //selectGradOutput.scal(output(word(i), i)*(1 - output(word(i), i)));
        	  //for test
        	  //cout << "he so: " << output(word(i), i)*(1 - output(word(i), i)) << endl;
            }
          else
            {
        	  // why this???
        	  // for test
        	  //cout << "bao dong:::" << endl;
              selectGradOutput.select(gradOutput, 1, i);
              selectOutput.select(output, 1, i);
              selectGradOutput.copy(selectOutput);
              selectGradOutput.scal(-1);
              gradOutput(mirror - word(i), i) += 1;
            }
        }
      else
        {
          selectGradOutput.select(gradOutput, 1, i);
          selectGradOutput = 0;
        }
    }
  //gradInput = weight*gradOutput
  gradInput.gemm(weight, 'N', gradOutput, 'N', 1, 0);
  // accumulate gradient
  gradWeight.gemm(input, 'N', gradOutput, 'T', 1, 1);
  gradWeight.axpy(weight, weightDecay);
  gradBias.gemv(gradOutput, 'N', V1col, 1, 1);
  return gradInput;
}

void
LinearSoftmax_Bayes::updateParameters(float learningRate)
{
	//because the objective function has the regularization term with constant weightDecay
  /*weight.gemm(input, 'N', gradOutput, 'T', -learningRate,
      1 - learningRate * weightDecay);*/
	// for Hamiltonian algorithm
	weight.axpy(pWeight, sqrt(2*learningRate));
	//bias.gemv(gradOutput, 'N', V1col, -learningRate, 1);
	bias.axpy(pBias, sqrt(2*learningRate));
}

void
LinearSoftmax_Bayes::updateRandomness(float learningRate) {
	pWeight.axpy(gradWeight, -sqrt(0.5*learningRate));
	pBias.axpy(gradBias, -sqrt(0.5*learningRate));
}

int
LinearSoftmax_Bayes::numberOfWeights() {
	return weight.size[0]*weight.size[1];
}

float
LinearSoftmax_Bayes::sumSquaredWeights() {
	return weight.sumSquared();
}

void
LinearSoftmax_Bayes::initializeP() {
	this->pWeight.initializeNormal();
	this->pBias.initializeNormal();
}

float
LinearSoftmax_Bayes::calculeH() {
	return 0.5*(this->pWeight.sumSquared() + this->pBias.sumSquared()) + 0.5*this->weightDecay*weight.sumSquared();
}

void
LinearSoftmax_Bayes::reUpdateParameters(int accept) {
	if (accept == 1) {
		prevWeight.copy(weight);
		prevBias.copy(bias);
		prevGradWeight.copy(gradWeight);
		prevGradBias.copy(gradBias);
	}
	else {
		weight.copy(prevWeight);
		// for test
		bias.copy(prevBias);
		gradWeight.copy(prevGradWeight);
		gradBias.copy(prevGradBias);
	}
}
