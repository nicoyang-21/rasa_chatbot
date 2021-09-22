# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.events import AllSlotsReset
from rasa_sdk.executor import CollectingDispatcher
import re

from rasa_sdk.forms import FormAction


class Actionhotelform(FormAction):

    def name(self) -> Text:
        return "validate_hotel_form"

    @staticmethod
    def required_slots(tracker: Tracker) -> List[Text]:
        """A list of required slots that the form has to fill"""

        return ["date_time", "phone_number", "person_number",
                "room_type"]

    @staticmethod
    def room_type_db() -> List[Text]:
        """Database of supported cuisines."""

        return [
            "标准间",
            "大床房",
            "标准双人间",
            "麻将房",
            "商务套间",
            "豪华商务套件",
            "总统套件",
        ]

    @staticmethod
    def is_int(string) -> bool:
        """Check if a string is an integer."""

        try:
            int(string)
            return True
        except ValueError:
            return False

    @staticmethod
    def is_phone_number(value) -> bool:
        """check phone_number is right"""
        pattern = re.compile(r'^1[3578]\d{9}$')
        res = pattern.match(value)
        if res:
            return True
        else:
            return False

    def validate_room_type(
            self,
            value: Text,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Validate room_type."""

        if value in self.room_type_db():
            # validation succeeded, set the value of the "cuisine" slot to value
            return {"room_type": value}
        else:
            dispatcher.utter_message(response="utter_wrong_room_type")
            # validation failed, set this slot to None, meaning the
            # user will be asked for the slot again
            return {"room_type": None}

    def validate_person_number(
            self,
            value: Text,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Validate num_people value."""

        if self.is_int(value) and int(value) > 0:
            return {"person_number": value}
        else:
            dispatcher.utter_message(response="utter_wrong_person_number")
            # validation failed, set slot to None
            return {"person_number": None}

    def validate_phone_number(self,
                              value: Text,
                              dispatcher: CollectingDispatcher,
                              tracker: Tracker,
                              domain: Dict[Text, Any],
                              ) -> Dict[Text, Any]:
        if self.is_phone_number(str(value)):
            return {"phone_number": value}
        else:
            dispatcher.utter_message(response="utter_wrong_phone_number")
            return {"phone_number": None}

    def submit(self, tracker: Tracker, dispatcher: CollectingDispatcher):
        """Define what the form has to do after all required slots are filled"""
        dispatcher.utter_template('utter_submit', tracker)
        return []


class ActionResetSlot(Action):
    def name(self) -> Text:
        return "action_resetSlot"

    def run(self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(text="执行了重置slot.*reply: action_resetSlot*")

        return [AllSlotsReset()]

