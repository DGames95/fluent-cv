from typing import Tuple, Dict, Any, Callable

import matplotlib.pyplot as plt
import cv2

from .image_processor import ImageProcessor
from .frequency_space import image_to_frequency, frequency_to_image

class FrequencyProcessor(ImageProcessor):
    """
    extend processor to use frequency space
    """
    def __init__(self, save_directory="processing_results"):
        super().__init__(save_directory)

        self._current_image_frequency_space = None

    @property
    def result_frequency(self):
        return image_to_frequency(self.result)

    def apply_frequency_space_filter(self, filter: Callable, *args, **kwargs) -> 'ImageProcessor':
        self._current_image_frequency_space = filter(self.result_frequency)

        self._current_image = frequency_to_image(self._current_image_frequency_space)

        # Record the processing step

        filter_name = filter.__name__

        parameters = {
            'args': [repr(arg) for arg in args],
            'kwargs': {k: repr(v) for k, v in kwargs.items() if kwargs}
        }

        step_info = {
            'filter_name': filter_name,
            'type': "Frequncy Space",
            'parameters': parameters
        }

        self._process_steps.append(step_info)
        return self
    
    def preview_frequency_filter(self, filter_func: Callable, *args, **kwargs) -> 'ImageProcessor':
        """
        Show a preview of the filter without applying it to the processor
        """

        # Get the function name for logging
        filter_name = filter_func.__name__

        # Apply the filter
        result = filter_func(self.result_frequency, *args, **kwargs)

        plt.figure(figsize=(10, 8))
        plt.title(filter_name)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

        return self