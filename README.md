# IntelligentTicketManagement
A Deep Ensemble Learning Model for Classification and Explanation of Tickets for Customer Relationship Management (CRM)

Intelligent Ticket Management System is Deep Ensemble Neural Network model used to classify tickets. The ensemble uses four deep base classifier architectures: LSTM (Long Short-Term Memory), CNN (Convolutional Neural Network), GRU (Gated Recurrent Unit) and Transformer.
Two ensemble combiners are provided: Stacking and MOE.

# Authors
The code is developed and maintained by Paolo Zicari, Gianluigi Folino, Massimo Guarascio and Luigi Pontieri (p.zicari@dimes.unical.it , gianluigi.folino@icar.cnr.it, massimo.guarascio@icar.cnr.it, luigi.pontieri@icar.cnr.it)

# Usage
First, download this repo:

You need to have 'python3' installed.
You also need to install 'numpy', 'pandas==1.0.3', and 'sklearn <=0.21', 'imbalanced-learn==0.5.0', 'Keras==2.2.4' and 'tensorflow==1.14.0'.

Then, you can run TrainingTest_EnsembleClassifier.py


For training and testing the ensemble model, Endava Dataset is used. To download Endava Dataset go to https://github.com/gabrielpreda/Support-Tickets-Classification
