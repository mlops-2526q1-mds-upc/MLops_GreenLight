from codecarbon import EmissionsTracker
from codecarbon import track_emissions

with EmissionsTracker() as tracker:
    # Your training code here
    print('dummy')



@track_emissions
def training_function():
    print('dummy')