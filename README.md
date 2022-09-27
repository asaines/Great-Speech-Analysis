# Great Speech Analysis
This project focus on determining a decision rule to identify should a speech be a great one. Metrics such as emotions, polarity, lexical richness, named entities proportion, complexity, imagery and stop words, and mean sentence length are thought to were used by 137 leading scholars of the American public to classify the most significant American political speeches of the 20th century. 

Furthermore, 77 great speeches were selected to be compared with 77 typical ones selected randomly from https://www.americanrhetoric.com/top100speechesall.html. Likewise, four classifier algorithms were trained to classify important and typical speeches. Finally, a random forest classifier with tuned hyper-parameters was selected since it achieves the highest ROC and accuracy (stable) test.

# Dataset

The dataset of measures related to typical and important speeches is found here. The speeches are scraped from the previous website of American Rhetoric. This file should be placed in the dataset folder for the code to run properly. To create this dataset, three hours or so are required depending on the processing capacity. As a component of this dataset, Imagery words are used from MRC Psycholinguistic Database (https://websites.psychology.uwa.edu.au/school/mrcdatabase/uwa_mrc.htm). The imagery words dataset should be placed in the resources folder, which is also found in this repository.


# Project Structure

The project consists of the following scripts:  BasicDataset, DataLoader, Speech, SpeechDataset, preprocessors, polarity_graph, sentiments_per_position, radar, and Speech Analysis notebook that portray the analysis making use of all these scripts and their functionality. Dataloader is used along the SpeechDataset script (iterable). SpeechDataset is intended to read pdf speeches and return a Speech. The Speech scripts contain all the methods to compute all the features defined above. Proporcessors scripts contain methods such as the BasicDataset to preprocess the data directly without proper treatment or AdvancedPreprocess that treats speeches to address stop words,  lemmatization, and remove punctuations. 
