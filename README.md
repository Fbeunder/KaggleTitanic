
# Titanic Survival Predictor

Een uitgebreide machine learning applicatie voor het voorspellen van overlevenden van de Titanic ramp.

## Beschrijving

De Titanic Survival Predictor is een web applicatie waarmee gebruikers verschillende machine learning modellen kunnen trainen, evalueren en vergelijken om te voorspellen welke passagiers de Titanic ramp zouden overleven op basis van hun demografische en reisgegevens. De applicatie omvat een complete data processing pipeline, feature engineering mogelijkheden, model training en evaluatie, en een gebruiksvriendelijke web interface.

## Kenmerken

- **Data Processing**: Laden, verwerken en transformeren van de Titanic dataset
- **Feature Engineering**: Creëren en selecteren van relevante features
- **Model Training**: Trainen van verschillende ML algoritmes (Logistic Regression, Random Forest, etc.)
- **Model Evaluatie**: Vergelijken van modellen op verschillende metrics
- **Hyperparameter Tuning**: Optimaliseren van model parameters
- **Web Interface**: Interactieve interface voor het gebruiken van de applicatie
- **Kaggle Submissions**: Genereren van submission files voor de Kaggle competitie

## Installatie

### Vereisten

- Python 3.8 of hoger (getest t/m Python 3.12)
- Voor Python 3.13 gebruikers: installeer eerst setuptools

### Stappen

1. Clone de repository:
```
git clone https://github.com/Fbeunder/KaggleTitanic.git
cd KaggleTitanic
```

2. Maak een virtuele omgeving aan:
```
python -m venv venv
source venv/bin/activate  # Op Windows: venv\Scripts\activate
```

3. Installeer de vereiste packages:

- Voor Python 3.8-3.12:
```
pip install -r requirements.txt
```

- Voor Python 3.13:
```
pip install setuptools
pip install -r requirements.txt
```

## Gebruik

### Web Interface

Start de web interface:

```
python src/web_interface/app.py
```

Open vervolgens je browser en ga naar `http://localhost:5000`.

### Jupyter Notebooks

Er zijn ook Jupyter notebooks beschikbaar in de `notebooks` directory voor interactieve data exploratie en model ontwikkeling.

## Project Structuur

- **data/**: Dataset bestanden en verwerkte data
- **notebooks/**: Jupyter notebooks voor data analyse
- **src/**: Broncode van de applicatie
  - **data_processing/**: Modules voor data verwerking
  - **feature_engineering/**: Modules voor feature engineering
  - **modelling/**: Modules voor model training en evaluatie
  - **utilities/**: Hulpmodules
  - **web_interface/**: Web applicatie
- **tests/**: Unit tests en integratie tests

## Tests

Het project bevat uitgebreide unit tests, integratie tests en end-to-end tests om de kwaliteit van de code te waarborgen.

### Tests uitvoeren

Alle tests uitvoeren:

```
python run_tests.py --all
```

Alleen unit tests uitvoeren:

```
python run_tests.py --unit
```

Alleen integratie tests uitvoeren:

```
python run_tests.py --integration
```

Web interface tests uitvoeren:

```
python run_tests.py --webtest
```

Coverage rapport genereren:

```
python run_tests.py --all --html
```

Dit genereert een HTML coverage rapport in de `htmlcov` directory.

## Bijdragen

Bijdragen aan dit project zijn welkom! Volg deze stappen:

1. Fork de repository
2. Maak een nieuwe branch: `git checkout -b feature/mijn-nieuwe-feature`
3. Commit je wijzigingen: `git commit -am 'Voeg een nieuwe feature toe'`
4. Push naar de branch: `git push origin feature/mijn-nieuwe-feature`
5. Dien een Pull Request in

Zorg ervoor dat je tests toevoegt voor nieuwe functionaliteit en dat alle tests slagen voor je een Pull Request indient.

## Licentie

Dit project is beschikbaar onder de MIT licentie. Zie het `LICENSE` bestand voor meer informatie.

## Contact

Voor vragen of suggesties, neem contact op met de repository eigenaar of open een issue.

## Dankwoord

- [Kaggle](https://www.kaggle.com/c/titanic) voor het beschikbaar stellen van de Titanic dataset
- Alle bijdragers aan dit project
