# Great Speech Analysis

This project aims to determine a decision rule to identify whether a speech is likely to be considered great. Metrics such as emotions, polarity, lexical richness, named entities proportion, complexity, imagery, stop words, and mean sentence length were analyzed, as used by 137 leading scholars to classify the most significant American political speeches of the 20th century.

Seventy-seven great speeches were compared with 77 typical ones, randomly selected from American Rhetoric (https://www.americanrhetoric.com/). Four classifier algorithms were trained to distinguish between important and typical speeches, with a random forest classifier chosen for its high ROC and accuracy.

# Dataset

The dataset includes measures related to both typical and important speeches. These speeches were scraped from the American Rhetoric website and should be placed in the dataset folder to run the code. Creating this dataset may take around three hours, depending on processing capacity. The imagery words used in this analysis are sourced from the MRC Psycholinguistic Database (https://websites.psychology.uwa.edu.au/school/mrcdatabase/uwa_mrc.htm) and should be placed in the resources folder in this repository.

# Project Structure

The project includes scripts such as BasicDataset, DataLoader, Speech, SpeechDataset, preprocessors, polarity_graph, sentiments_per_position, radar, and the Speech Analysis notebook. The analysis is conducted using these scripts. DataLoader and SpeechDataset work together to process PDF speeches, with Speech-containing methods for feature computation. The preprocessors include basic and advanced techniques like stop word removal, lemmatization, and punctuation handling.

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/asaines/Great-Speech-Analysis.git
   cd Great-Speech-Analysis

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt

3. **Run the Analysis**:
   Follow the instructions in the Speech Analysis notebook to perform the speech analysis.


# License
This project is licensed under the MIT License.

# Contact
For collaboration or inquiries, please reach out to me at asaines.kul@gmail.com.
