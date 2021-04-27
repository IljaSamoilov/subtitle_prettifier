# Try more or less model with cosine similarity
Left original model and right try more or less model  
[Foorum diff link](https://www.diffchecker.com/PYFcYg5w) 

#Try more or less model with levenshtein distance
Left original model and right try more or less model  
[Forum diff link](https://www.diffchecker.com/rywl3zbO)

Materials to read:  
https://www.semanticscholar.org/paper/COMPARISON-OF-EVALUATION-METRICS-FOR-SENTENCE-Liu-Shriberg/aca91868c0e9e5d7cc0dc533d0ab94f1360b6d76?p2df  
https://arxiv.org/pdf/1802.05667.pdf  
https://medium.com/@adriensieg/text-similarities-da019229c894  
https://medium.com/mlreview/implementing-malstm-on-kaggles-quora-question-pairs-competition-8b31b0b16a07

# Moving window implementation
Training and result data pairing is based on moving window implementation what checks differences between current, one before and next one lines.

## How to use

1. Place .vtt subtitle files to the ./data folder
2. Place .json transcript files to the ./data folder with the same names as subtitle files
3. Run moving_window.py file py3

Example files and folders are added.