import typing
from typing import Any, Optional, Text, Dict, List, Type
from rasa.nlu.config import RasaNLUModelConfig
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.nlu.components import Component
from rasa.nlu import utils
from rasa.nlu.model import Metadata


from emotion_classification.sentiment_predict import sentiment

