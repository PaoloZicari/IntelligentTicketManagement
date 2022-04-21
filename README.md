# IntelligentTicketManagement
A Deep Ensemble-based Framework for Classification of Customer Relationship Management (CRM) tickets.

Intelligent Ticket Management Systems (ITMS) is a machine learning based classification model that allows processing ticket text to automate their categorization. Basically, our approach takes the form of an ensemble  of deep base classifiers featuring different DNN architectures. Im more details, ITMS-Deep introduces two novel combination strategies (respectively based on stacking and mixture-of-expert architectures), both leveraging an ad hoc sub-net, named HLFE, for extracting dense (latent space) ticket representations. 

# Authors
The code is developed and maintained by Paolo Zicari, Gianluigi Folino, Massimo Guarascio and Luigi Pontieri (p.zicari@dimes.unical.it , gianluigi.folino@icar.cnr.it, massimo.guarascio@icar.cnr.it, luigi.pontieri@icar.cnr.it)

# Usage
First, download this repo.

You need to have 'python3' installed.
You also need to install 'numpy', 'pandas==1.0.3', and 'sklearn <=0.21', 'imbalanced-learn==0.5.0', 'Keras==2.2.4' and 'tensorflow==1.14.0'.

Then, you can run:
python TrainTest_EnsembleClassifier.py

For training and testing the ensemble model, Endava Dataset is used. To download Endava Dataset go to https://github.com/gabrielpreda/Support-Tickets-Classification and follow the instructions to download the "all_tickets.csv" file. Then, create the ./dataset/endava folder and put the dataset file inside. 
In order to change the dataset file name, the path, and any other parameter for the model training and test, go to the Confs.py file that contains all the settings.
