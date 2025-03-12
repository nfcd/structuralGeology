"""
The two functions below were made by Wiktor Weibull
at the University of Stavanger, Norway
Any questions or comments, please contact:
wiktor.w.weibull@uis.no
"""
import numpy as np

def Stolt2Dmod(data, vmig, dt, dx, thetamax):
    """
    This function implements the Stolt algorithm to model 
    zero-offset seismic data using the "exploding reflector model" 
    concept (https://wiki.seg.org/wiki/Exploding_reflectors). 
    Parameters:
    data : This is a 2D numpy array representing the band-limited 
        reflectivity image
    vmig : Single number representing the seismic wave propagation 
        velocity (must be constant in the Stolt alogrithm)
    dt : The time sampling interval in seconds (must be the same 
        as the reflectivity function)
    dx : The lateral spatial sampling in meters (must be the same 
        as the reflectivity function)
    thetamax : Aperture limit in degrees with respect to the vertical. 
        This angle can be between 0-90 degrees.
    """
    vzo = vmig/2.
    nt,nx = data.shape 
    ang=np.pi*thetamax/180.0

    # Zero-padding to avoid wrap-around noise in the Fourier domain
    nw = 2*nt
    nkx = 2*nx

    # Fourier transform to f,kx
    ZOFFT = np.fft.fftshift(np.fft.fft2(data,[nw,nkx],axes=(0,1)))

    kx = np.fft.fftshift(np.fft.fftfreq(nkx)/dx);
    f = np.fft.fftshift(np.fft.fftfreq(nw)/dt);

    ZOMAP = np.zeros([nw,nkx]).astype(complex)
    # Remapping from f,kx to kx,kz before double inverse transform over kx, kz
    for ikx in range(0,nkx):
        fkz = vzo*np.sign(f)*np.sqrt(kx[ikx]**2 + f**2/vzo**2)
        frange = np.abs(f)-np.abs((vmig*kx[ikx]/np.sin(ang))) 
        scaling = np.where(frange > 0, 1, np.exp(-0.1*np.abs(frange)))
        ZOMAP[:,ikx] = np.interp(f, fkz, ZOFFT[:,ikx])*scaling

    # Inverse transform
    ZOSTOLT = np.real(np.fft.ifft2(np.fft.ifftshift(ZOMAP),axes=(0,1)))

    #Windowing back to original size
    ZOSTOLT = ZOSTOLT[0:nt,0:nx]        
    return ZOSTOLT

def Stolt2Dmig(data, vmig, dt, dx, thetamax,datum=0):
    """
    This function implements the Stolt algorithm to migrate 
    zero-offset seismic data using the "exploding reflector model" 
    concept (https://wiki.seg.org/wiki/Exploding_reflectors). 
    Parameters:
    data : This is a 2D numpy array representing the stack section
    vmig : Single number representing the seismic wave propagation 
        velocity (must be constant in the Stolt alogrithm)
    dt : The time sampling interval in seconds (must be the same as 
        the reflectivity function)
    dx : The lateral spatial sampling in meters (must be the same as 
        the reflectivity function)
    thetamax : Aperture limit in degrees with respect to the vertical. 
        This angle can be between 0-90 degrees.
    """

    vzo = vmig/2.
    nt,nx = data.shape 
    ang=np.pi*thetamax/180.0

    # Zero-padding to avoid wrap-around noise in the Fourier domain
    nw = 2*nt
    nkx = 2*nx

    # Fourier transform to f,kx
    ZOFFT = np.fft.fftshift(np.fft.fft2(data,[nw,nkx],axes=(0,1)))

    kx = np.fft.fftshift(np.fft.fftfreq(nkx)/dx);
    f = np.fft.fftshift(np.fft.fftfreq(nw)/dt);

    MIGMAP = np.zeros([nw,nkx]).astype(complex)
    # Remapping from f,kx to kx,kz before double inverse transform over kx, kz
    for ikx in range(0,nkx):
        fkz = vzo*np.sign(f)*np.sqrt(kx[ikx]**2 + f**2/vzo**2)
        frange = np.abs(f)-np.abs((vmig*kx[ikx]/np.sin(ang))) 
        scaling = np.where(frange > 0, 1, np.exp(-0.1*np.abs(frange)))
        kz = np.sqrt(kx[ikx]**2 + f**2/vzo**2)
        MIGMAP[:,ikx] = np.interp(fkz, f, ZOFFT[:,ikx]*np.exp(-1j*2*np.pi*kz*datum))*scaling*np.exp(1j*2*np.pi*f*datum/vzo)

    # Inverse transform
    MIGSTOLT = np.real(np.fft.ifft2(np.fft.ifftshift(MIGMAP),axes=(0,1)))

    #Windowing back to original size
    MIGSTOLT = MIGSTOLT[0:nt,0:nx]        
    return MIGSTOLT