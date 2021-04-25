# SE_Deep

### Structure
- notebooks contients differents notebooks utilisés pour diverses application (notamment sur google colab)
- test et train sont les dossiers contenant les audios. Chaque dossiers est composé d'un sous-dossier "ori".
- utils va contenir les fonctions utilisées pendant ce projet (génération de bruit, de masques, générateurs ect...)
- models va contenir les poids des modèles entrainés (format .h5, cf notebooks pour le chargement).
- gen_noise.py va nous permettre de generer les sons bruités avec un SNR donné.  Ces derniers seront generé dans les dossiers train et test (à partir des sons des sous dossiers 'ori et une combinaison aleatoire de partie de babble_sub.wav')

### Exemple d'utilsation de gen_noise.py pour génerer un seul dossiers 0dB dans train et dans test:
gen_noise.py 0

### Exemple d'utilsation de gen_noise.py pour génerer des dossiers 0dB -3dB -6dB et -8dB dans train et dans test:
gen_noise.py 0 -3 -6 -8

### Pour installer les paquets nécessaires
python -m pip install -r requirements.txt
