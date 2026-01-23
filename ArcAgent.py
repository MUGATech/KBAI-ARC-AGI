import numpy as np

from ArcProblem import ArcProblem
from ArcData import ArcData
from ArcSet import ArcSet


class ArcAgent:
    def __init__(self):
        """
        You may add additional variables to this init method. Be aware that it gets called only once
        and then the make_predictions method will get called several times.
        """
        pass

    def make_predictions(self, arc_problem: ArcProblem) -> list[np.ndarray]:
        """
        Write the code in this method to solve the incoming ArcProblem.
        Your agent will receive 1 problem at a time.

        You can add up to THREE (3) the predictions to the
        predictions list provided below that you need to
        return at the end of this method.

        In the Autograder, the test data output in the arc problem will be set to None
        so your agent cannot peek at the answer (even on the public problems).

        Also, if you return more than 3 predictions in the list it
        is considered an ERROR and the test will be automatically
        marked as INCORRECT.
        """

        

        predictions: list[np.ndarray] = list()

        '''
        The next 2 lines are only an example of how to populate the predictions list.
        This will just be an empty answer the size of the input data;
        delete it before you start adding your own predictions.
        '''
        # output = np.zeros_like(arc_problem.test_set().get_input_data().data())
        # predictions.append(output)
        
        test_input_data = arc_problem.test_set().get_input_data().data()
        ys, xs = np.nonzero(test_input_data)

        ''' If no non-zero cells exist return np array with 0
        '''
        if len(xs) == 0:
            return predictions

        ''' Get min and max indicides of rows
        '''
        ymin, ymax = ys.min(), ys.max()

        ''' Get min and max indicides of columns
        '''
        xmin, xmax = xs.min(), xs.max()
        
        ''' Crop using numpy slicing
        '''
        cropped = test_input_data[ymin:ymax + 1, xmin:xmax + 1]

        predictions.append(cropped);

        return predictions

