import numpy as np
import scipy.misc


def ppdrc(im, wavelength, clip=None, n=None):
    '''
     PPDRC Phase Preserving Dynamic Range Compression
    
     Generates a series of dynamic range compressed images at different scales.
     This function is designed to reveal subtle features within high dynamic range
     images such as aeromagnetic and other potential field grids. Often this kind
     of data is presented using histogram equalisation in conjunction with a
     rainbow colourmap. A problem with histogram equalisation is that the contrast
     amplification of a feature depends on how commonly its data value occurs,
     rather than on the amplitude of the feature itself. In addition, the use of a
     rainbow colourmap can introduce undesirable perceptual distortions.
    
     Phase Preserving Dynamic Range Compression allows subtle features to be
     revealed without these distortions. Perceptually important phase information
     is preserved and the contrast amplification of anomalies in the signal is
     purely a function of their amplitude. It operates as follows: first a highpass
     filter is applied to the data, this controls the desired scale of analysis.
     The 2D analytic signal of the data is then computed to obtain local phase and
     amplitude at each point in the image. The amplitude is attenuated by adding 1
     and then taking its logarithm, the signal is then reconstructed using the
     original phase values.
    
     Usage: dim = ppdrc(im, wavelength, clip, savename, n)
    
     Arguments:      im - Image to be processed (can contain NaNs)
             wavelength - Array of wavelengths, in pixels, of the  cut-in
                          frequencies to be used when forming the highpass
                          versions of the image.  Try a range of values starting
                          with, say, a wavelength corresponding to half the size
                          of the image and working down to something like 50
                          grid units. 
                   clip - Percentage of output image histogram to clip.  Only a
                          very small value should be used, say 0.01 or 0.02, but 
                          this can be beneficial.  Defaults to 0.01
               savename - (optional) Basename of filname to be used when saving
                          the output images.  Images are saved as
                          'basename-n.png' where n is the highpass wavelength
                          for that image .  You will be prompted to select a
                          folder to save the images in. 
                      n - Order of the Butterworth high pass filter.  Defaults
                          to 2
    
     Returns:       dim - Cell array of the dynamic range reduced images.  If
                          only one wavelength is specified the image is returned 
                          directly, and not as a one element cell array.
    
     Note when specifying the array 'wavelength' it is suggested that you use
     wavelengths that increase in a geometric series.  You can use the function
     GEOSERIES to conveniently do this
     
     Example using GEOSERIES to generate a set of wavelengths that increase
     geometrically in 10 steps from 50 to 800. Output is saved in a series of
     image files called 'result-n.png'
       dim = compressdynamicrange(im, geoseries([50 800], 10), 'result');
    
     View the output images in the form of an Interactive Image using LINIMIX
    
     See also: HIGHPASSMONOGENIC, GEOSERIES, LINIMIX, HISTRUNCATE
    
     Reference:
     Peter Kovesi, "Phase Preserving Tone Mapping of Non-Photographic High Dynamic
     Range Images".  Proceedings: Digital Image Computing: Techniques and
     Applications 2012 (DICTA 2012). Available via IEEE Xplore

     Copyright (c) 2012-2014 Peter Kovesi
     Centre for Exploration Targeting
     The University of Western Australia
     peter.kovesi at uwa edu au
     
     Permission is hereby granted, free of charge, to any person obtaining a copy
     of this software and associated documentation files (the "Software"), to deal
     in the Software without restriction, subject to the following conditions:
     
     The above copyright notice and this permission notice shall be included in 
     all copies or substantial portions of the Software.
    
     The Software is provided "as is", without warranty of any kind.
    
     April 2012  - Original version
     Feb   2014  - Incorporated histogram truncation
    '''
    if n == None:
        n = 2
    if clip == None:
        clip = 0.01
    
    # Check whether there is one wavelength or an array of wavelengths
    nscale = length(wavelength)
    
    # Identify no-data regions in the image (assummed to be marked by NaN
    # values). These values are filled by a call to fillnan() when passing image
    # to highpassmonogenic.  While fillnan() is a very poor 'inpainting' scheme
    # it does keep artifacts at the boundaries of no-data regions fairly small.
    mask = np.argwhere(np.isnan(im))
    phase, orient, E, f, h1f, h2f = highpassmonogenic(fillnan(im), wavelength, n)
    
    # Construct each dynamic range reduced image 
    # Make sure phase and E are enapsuled 
    phase = np.atleast_1d(phase)
    E = np.atleast_1d(E)

    range_ary = []
    dim = []
    for k in range(nscale):
        dim_tmp = histtruncate(sin(phase(k)) * log1p(E(k)), clip, clip) * mask
        
        dim.append( dim_tmp )
        range_ary.append( max(abs(dim_tmp)) )
    
    maxrange = max(range_ary)
    # Set the first two pixels of each image to +range and -range so that
    # when the sequence of images are displayed together, say using LINIMIX,
    # there are no unexpected overall brightness changes
    for k in range(nscale):
        dim(k)[0] =  maxrange
        dim(k)[1] = -maxrange    

    return dim