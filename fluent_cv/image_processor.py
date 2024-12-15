from typing import Tuple, Dict, Any, Callable
import json
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt


class ImageProcessor:
    """
    Main processor that handles the image processing pipeline.
    Maintains original image and current state after each processing step.
    Allows preview and save operations between steps.
    Records processing history in JSON format.

    EXAMPLE USAGE
    processor = (ImageProcessor()
            .load_image("your_image.jpg")
            .apply_filter(cv2.medianBlur, 5)
            .preview()
            .save("output.jpg"))
    """
    
    def __init__(self, save_directory="processing_results"):
        self.save_dir = save_directory
        self._original_filepath = None
        self._original_image = None
        self._current_image = None
        self._process_steps = []  # List to store processing history

    @property
    def result(self) -> np.ndarray:
        return self._current_image

    @property
    def _image_loaded(self):
        return not (self._original_image is None)

    def load_image(self, image_path: str, flag = None) -> 'ImageProcessor':
        """
        Load image from path with error checking

        flag: e.g. cv2.IMREAD_GRAYSCALE
        """
        if self._image_loaded:
            raise ValueError(f"Image: {self._original_filepath} already loaded.")

        image = cv2.imread(image_path, flag)
        if image is None:
            raise ValueError(f"Failed to load image at path: {image_path}")
        
        self._process_steps.append({"filter_name": f"loaded image: {image_path}"})
        self._original_filepath = image_path
        self._original_image = image
        self._current_image = image

        return self
    
    def preview(self) -> 'ImageProcessor':
        """
        Display current state of image using matplotlib
        Returns self for method chaining
        """
        if not self._image_loaded:
            raise ValueError("Image not loaded yet")

        plt.figure(figsize=(10, 8))
        plt.title(str("last_applied: " + str(self._process_steps[-1]["filter_name"])))
        plt.imshow(cv2.cvtColor(self._current_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        # Make it non-blocking
        plt.ion()
        plt.draw()
        return self
    
    def save(self, filepath: str) -> 'ImageProcessor':
        timestamp = datetime.now().isoformat()

        if not self._image_loaded:
            raise ValueError("Image not loaded yet")

        # Create processing_results directory if it doesn't exist
        save_dir = Path(self.save_dir)
        save_dir.mkdir(exist_ok=True)
        
        # Get the original path components
        path_obj = Path(filepath)
        filename = path_obj.stem  # Get filename without extension
        extension = path_obj.suffix  # Get extension
        
        # Create new filename with _edit suffix
        new_filename = f"{filename}_revision_1{extension}"
        image_path = save_dir / new_filename
        
        # If file already exists with _edit suffix, increment number
        counter = 2
        while image_path.exists():
            new_filename = f"{filename}_revision_{counter}{extension}"
            image_path = save_dir / new_filename
            counter += 1
        
        # Save image and metadata
        cv2.imwrite(str(image_path), self._current_image)
        
        metadata = {
            'original_filepath_loaded': self._original_filepath,
            'timestamp': timestamp,
            'processing_steps': self._process_steps,
        }
        
        with open(image_path.with_suffix('.json'), 'w') as f:
            json.dump(metadata, f, indent=4)
            
        return self
    
    def optional_save(self) -> None:
        """
        Prompts user if they want to save the image and handles custom naming
        """
        if not self._image_loaded:
            raise ValueError("Image not loaded yet")
            
        save_response = input("Would you like to save the image? (any/n): ").lower().strip()
        
        if save_response == 'n':
            return
            
        original_name = Path(self._original_filepath).stem.strip()
        suffix = "edited"
        custom_suffix = input(f"Enter custom tags (press Enter to use '{original_name}_{suffix}'): ").strip()
        
        if custom_suffix:
            suffix = custom_suffix

        filename = str(original_name + "_" + suffix)

        print(filename)

        filepath = Path(filename).with_suffix('.png')  # Using png as default format lossless

        print(filepath)

        self.save(str(filepath))


    def apply_filter_function(self, filter_func: Callable, *args, **kwargs) -> 'ImageProcessor':
        """
        Apply any function to the image

        Parameters:
        -----------
        filter_func : Callable
            A function that takes an np.ndarray + arguments and returns np.ndarray
            Most opencv filters work in this way
        *args
            Positional arguments for the filter function
        **kwargs
            Keyword arguments for the filter function

        Returns:
        --------
        ImageProcessor
            The ImageProcessor instance for method chaining

        Example:
        --------
        # Using with positional arguments
        processor.apply_filter(cv2.medianBlur, 5)
        
        # Using with keyword arguments
        processor.apply_filter(cv2.GaussianBlur, (5,5), sigmaX=0)
        """

        # Get the function name for logging
        filter_name = filter_func.__name__
        print("applying ", filter_name)

        start = datetime.now()
        # Apply the filter
        self._current_image = filter_func(self._current_image, *args, **kwargs)

        end = datetime.now()

        # Record the processing step
        parameters = {
            'args': [repr(arg) for arg in args],
            'kwargs': {k: repr(v) for k, v in kwargs.items() if kwargs}
        }

        step_info = {
            'filter_time' : (end-start).total_seconds(),
            'filter_name': filter_name,
            'parameters': parameters,
        }

        self._process_steps.append(step_info)

        return self
    
    def preview_filter(self, filter_func: Callable, *args, **kwargs) -> 'ImageProcessor':
        """
        Show a preview of the filter without applying it to the processor
        """

        # Get the function name for logging
        filter_name = filter_func.__name__

        # Apply the filter
        result = filter_func(self._current_image, *args, **kwargs)

        plt.figure(figsize=(10, 8))
        plt.title(filter_name)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

        return self

    def show(self):
        """
        Display all previewed figures and wait for user input before closing
        """
        plt.ioff()  # Turn off interactive mode
        plt.show(block=True)  # Show all figures and block

        return self

    ### shorthands
    def s(self, filepath):
        return self.save(filepath)
    
    def p(self):
        return self.preview()
    
    def a(self, filter, *args, **kwargs):
        return self.apply_filter_function(filter, *args, **kwargs)
    
    def os(self):
        return self.optional_save()