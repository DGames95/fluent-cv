import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift

def image_to_frequency(image: np.ndarray) -> np.ndarray:
    """
    Convert an image from spatial domain to frequency domain using FFT.
    
    Parameters:
    -----------
    image : np.ndarray
        Input image as a 2D numpy array (grayscale) or 3D numpy array (color)
        For color images, the FFT is applied to each channel separately
    
    Returns:
    --------
    np.ndarray
        Frequency spectrum of the image with the same shape as input
        The output is shifted so that the zero-frequency component is centered
    """
    image = image.astype(np.float32)

    
    # Check if the image is grayscale or color
    if len(image.shape) == 2:
        # Grayscale image
        # Apply 2D FFT and shift zero frequency to center
        freq_spectrum = fftshift(fft2(image))
        
        # Convert to magnitude spectrum (absolute values)
        # Using log scale to better visualize the frequency components
        magnitude_spectrum = np.log1p(np.abs(freq_spectrum))
        
        # Normalize to 0-1 range for better visualization
        magnitude_spectrum = (magnitude_spectrum - magnitude_spectrum.min()) / \
                           (magnitude_spectrum.max() - magnitude_spectrum.min())
        
        return magnitude_spectrum
    
    elif len(image.shape) == 3:
        # Color image
        freq_spectrum = np.zeros_like(image, dtype=complex)
        magnitude_spectrum = np.zeros_like(image)
        
        # Process each channel separately
        for channel in range(image.shape[2]):
            freq_spectrum[:,:,channel] = fftshift(fft2(image[:,:,channel]))
            magnitude_spectrum[:,:,channel] = np.log1p(np.abs(freq_spectrum[:,:,channel]))
            
            # Normalize each channel
            magnitude_spectrum[:,:,channel] = \
                (magnitude_spectrum[:,:,channel] - magnitude_spectrum[:,:,channel].min()) / \
                (magnitude_spectrum[:,:,channel].max() - magnitude_spectrum[:,:,channel].min())
        
        return magnitude_spectrum
    
    else:
        raise ValueError("Input image must be 2D (grayscale) or 3D (color) array")
    

def frequency_to_image(freq_spectrum: np.ndarray) -> np.ndarray:
    """
    Convert frequency spectrum back to spatial domain using inverse FFT.
    
    Parameters:
    -----------
    freq_spectrum : np.ndarray
        Input frequency spectrum as complex-valued 2D numpy array (grayscale) 
        or 3D numpy array (color)
    
    Returns:
    --------
    np.ndarray
        Reconstructed image in spatial domain with the same shape as input
    """
    
    if len(freq_spectrum.shape) == 2:
        # Grayscale image
        # Apply inverse shift and inverse FFT
        reconstructed = np.real(ifft2(ifftshift(freq_spectrum)))
        
        # Ensure output is in valid image range
        reconstructed = np.clip(reconstructed, 0, 255)
        return reconstructed.astype(np.uint8)
        
    elif len(freq_spectrum.shape) == 3:
        # Color image
        reconstructed = np.zeros_like(freq_spectrum)
        
        # Process each channel separately
        for channel in range(freq_spectrum.shape[2]):
            reconstructed[:,:,channel] = np.real(ifft2(ifftshift(freq_spectrum[:,:,channel])))
        
        # Ensure output is in valid image range
        reconstructed = np.clip(reconstructed, 0, 255)
        return reconstructed.astype(np.uint8)
        
    else:
        raise ValueError("Input frequency spectrum must be 2D (grayscale) or 3D (color) array")