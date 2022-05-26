from tiyaro.sdk.base_handler import TiyaroBase

from flair.models import TextClassifier
from flair.data import Sentence

class TiyaroHandler(TiyaroBase):
    # For text-classification, simply adhere to the input and output format specified in __pre_process() and __post_process(). This will automatically do the following.
    #
    # 1. Tiyaro will automatically generate an OpenAPI spec for your Model's API
    # 2. Tiyaro will automatically generate sample code snippets
    # 3. Tiyaro will automatically provide Demo Screen in the Model Card of your model, to show case live demo instantly to the world
    # 4. Tiyaro will enable you, and your model users, to create experiments and compare with wide range of models in image-classification class in Tiyaro
    # 5. With Tiyaro experiments you and your model users will be able to get comprehensive comparision with other models on various metrics, graphs, etc.,

    def setup_model(self, pretrained_file_path):
        self.model = TextClassifier.load('en-sentiment')
        
    def __pre_process(self, input_string):
        sentence = Sentence(input_string)        
        return sentence

    def infer(self, input_string):
        sentence = self.__pre_process(input_string)

        self.model.predict(sentence)

        return self.__post_process(sentence)
        
    def __post_process(self, model_output):
        return model_output.labels[0].to_dict()
