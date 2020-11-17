from estnltk.wordnet import Wordnet
from estnltk.wordnet.synset import Synset

wn = Wordnet()


synset: Synset = wn['isik'][0]
target_synset: Synset = wn['persoon'][0]

