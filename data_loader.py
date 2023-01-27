import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

class Data():
    def __init__(self, filename):
        with open(filename, 'rb') as file:
            self.full_data = pkl.load(file)
        self.length = len(self.full_data)

    def __call__(self, event):
        self.current_event_data = self.full_data['evt_%i' % event]


    def test_plot(self, event):

        self.current_event_data = self.full_data['evt_%i' % event]

        plt.figure()
        plt.plot(self.current_event_data)
        plt.show()