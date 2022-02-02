# vlf-mri

Very-low-fied MRI  (vlf-mri) implements data analysis algorithms for Fast Field Cycling Nuclear Magnetic Resonance data (FFC-NMR).


## Description
FFC-NMR data generaly consists in a collection of Free Induction Decay signals (FID) acquired using a set of NMR sequence parameters. In this librariry, we assume three dimensional data *s(t, τ, B<sub>relax</sub>)*, with:
- *t* : the experimental time at which a data point was acquired
- *τ* : the evolution time, between the prepolarisation and data acquision steps
- *B<sub>relax</sub>* : the relaxation magnetic field, or the value of the magnetic field during the evolution time

The objective of this librairy is, from the signal *s(t, τ, B<sub>relax</sub>)*, to compute the relaxation profiles *R<sub>1</sub>(B<sub>relax</sub>)*, which provide valuable information about the molecular dynamics in a sample.

This is performed in a few steps:
1) Experimental data  is imported from a text file (\*.sdf for now) and stored into a **FidData** object
2) Magnetization of the sample *M(τ, B<sub>relax</sub>)* is extracted from the fid signal and stored into a **MagData** object
3) Relaxation profiles *R<sub>1</sub>(B<sub>relax</sub>)* and computed from the magnetization and stored into a  **RelData** object

## Features
Each data class (FidData, MagData, RelData) implements various (and easy to use!) methods for:
- Data visualization
- Reporting
- Data cleaning and masking
- Data analysis algorithms to go from one step of the process to the next (use what you think works best!)

## Installation
Step one : clone this repository and save it your computer
Step two : add the directory where you saved this library to your path environment (see example below!)

## Example
vlf-mri is easy to use! Here's an example:

```python
# adding the path where vlf_mri is saved
import sys
sys.path.append('C:/Users/user/GitHub/vlf_mri')  # change according to your path!

from pathlib import Path
import vlf_mri

datafile = Path("test_data.sdf")
fid_data = vlf_mri.import_sdf_file(datafile)
fid_data.batch_plot()  # The data contains outliers!
```
![alt text](https://github.com/ReciprocalSpace/ReciprocalSpace/blob/main/images/fid_outliers.png)

```python
# Let's remove those bad datapoints! 
fid_data.apply_mask(sigma=5, display_report=True)  # This should do the trick!
```
![alt text](https://github.com/ReciprocalSpace/ReciprocalSpace/blob/main/images/mask_report.png)

```python
fid_data.batch_plot()  # This is better!
```

![alt text](https://github.com/ReciprocalSpace/ReciprocalSpace/blob/main/images/fid_after_mask.png)

```python
# We now want to analyse this data. This first step is to extract the sample magnetization
# by computing the mean of the FID signal between 5 and 50 us. (simple, but effective!)
mag_data_mean = fid_data.to_mag_mean(t_0=5, t_1=50)

# Now, we can compute the Relaxation profiles of the magnetization data. This line tests two models 
# on the data : one where there's only one profile, and one where there are two.
rel_data = mag_data_mean.to_rel()

# Let's now save the reports in a pdf !
mag_data_mean.save_to_pdf()
rel_data.save_to_pdf()

```
This report shows the fit (left side) and its residue (right side) considering the two models. Each line corresponds to a specific *B<sub>relax</sub>* value:

![alt text](https://github.com/ReciprocalSpace/ReciprocalSpace/blob/main/images/mag_pdf_report.png)

This reports shows the relaxation profiles (top) for the two models as a function of *B<sub>relax</sub>* (rigourously the frequency *f* [MHz] associated with this field). The bottom row displays displays the population (in \%) corresponding of each component/profile given in the model with two relaxation profiles.

![alt text](https://github.com/ReciprocalSpace/ReciprocalSpace/blob/main/images/rel_pdf_report.png)

