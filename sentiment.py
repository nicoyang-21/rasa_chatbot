import typing
from typing import Any, Optional, Text, Dict, List, Type
from rasa.nlu.config import RasaNLUModelConfig
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.nlu.components import Component
from rasa.nlu import utils
from rasa.nlu.model import Metadata

from emotion_classification.sentiment_predict import sentiment


class SentimentAnalyzer(Component):
    """一个预训练的情感识别组建"""

    name = "sentiment"
    provides = ["entities"]
    requires = []
    defaults = {}
    language_list = ["en"]

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None) -> None:
        super().__init__(component_config)

    def train(
            self,
            training_data: TrainingData,
            config: Optional[RasaNLUModelConfig] = None,
            **kwargs: Any,
    ) -> None:
        """训练阶段代码"""
        print('训练已经完成，基于bert的预训练语言模型进行fine_tune')

    @staticmethod
    def convert_to_rasa(value, confidence):
        """把模型的输出转化为 rasa 能够识别的输出."""

        entity = {"value": value,
                  "confidence": confidence,
                  "entity": "sentiment",
                  "extractor": "sentiment_extractor"}

        return entity

    def process(self, message: Message, **kwargs: Any) -> None:
        """使用分类器来处理文本，并且转化为 rasa 能够接受的格式"""
        output = sentiment(message.as_dict_nlu()['text'])
        print('##################### 情感识别 ###########################')
        value, confidence = output['intent'], output['confidence']
        entity = self.convert_to_rasa(value, confidence)
        message.set("entities", [entity], add_to_output=True)

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        """组建持久化的运行代码"""
        print('持久化代码')
