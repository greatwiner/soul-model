/******************************************************************
 Structure OUtput Layer (SOUL) Language Model Toolkit
 (C) Copyright 2009 - 2012 Hai-Son LE LIMSI-CNRS

 General class for module (layer)
 *******************************************************************/

#include "mainModule.H"

Module::Module()
{
}
Module::~Module()
{

}

void
Module::shareWeight(floatTensor& weight)
{
	this->weight.tieData(weight);
}
