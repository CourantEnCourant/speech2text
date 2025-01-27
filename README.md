# Speech2text

## Plan du travail
* Fine-tuner un Whisper (model-W) dans deux langues différentes : français (model-F) et chinois (model-C)
* Fine-tuner encore model-B et model-C sur le japonais, pour comparer une différence de performance

## Hypothèse
* model-C va avoir une meilleure performance sur le japonais, comme nous imaginons que le chinois et plus proche du japonais (ce qui peut tout à fait être un stéréotype)
* Pour model-C, l'entraînement sera plus rapide, et nécessite moins de temps pour arriver au même résultat que model-F.

## Préparation des corpora
* Lancer `python prepare_dataset.py [code de langue comme fr/zh-CN/ja]` pour générer un corpus, qui est une sous-section de [common_voice_17_0](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0)
* Nous avons réussi à gérer 3 corpus

## Choix du model
* Voir `test_model.ipybn`, où nous avons réussi de faire tourner le model sur une phrase en chinois

## Fine-tuner
* Comme montré dans `training.ipynb`, nous rencontrons des difficultés sérieuses. Il semble que l'erreur vient plutôt de notre serveur, comme il y a une message de WARNING qui dit que la version de Linux Kernel est inférieur à la recommandation.
* Nous n'avons pas eu le temps de tester le code sur d'autres serveurs

## À améliorer pour plus tard
* Le but de ce projet est d'examiner l'idée de "transfer-learning" sur des models. Cependant, nous avons choisi un model multilingue dès le début. Il est préférable d'utiliser un server mono-lingue, comme [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base)
* Cependant, cela nécessite d'entraîner un tokenizer pour chaque langue que nous voulons tester. Avec [Sentencepiece](https://github.com/google/sentencepiece) par exemple. Ce qui prendra beaucoup de temps.

