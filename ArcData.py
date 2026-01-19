from copy import deepcopy
import numpy as np


class ArcData:
    """
    A Numpy multidimensional array representing an Arc Grid layout with data.

    From an Arc problem perspective this represents a single grid of data
    either input data or output data.
    """
    def __init__(self, data: np.ndarray):
        self._arc_array: np.ndarray = np.array(data)

    def data(self) -> np.ndarray:
        """
        Returns a copy of the numpy ndarray for this ArcData.
        """
        return deepcopy(self._arc_array)

    def shape(self) -> tuple[int, ...]:
        """
        Returns the size of this numpy ndarray for this ArcData.
        """
        return self._arc_array.shape

    def __eq__(self, other) -> bool:
        """
        Returns True if and only if the shape of
        the other ndarray is the same as this
        ArcData and all elements are equal;
        False otherwise.
        """
        return np.array_equal(self.data(), other.data())
