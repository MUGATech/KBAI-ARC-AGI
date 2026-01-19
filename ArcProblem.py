from copy import deepcopy

from ArcSet import ArcSet


class ArcProblem:
    """
    A basic Arc problem containing
    a list of training data (ArcSet(s))
    and a test set (ArcSet).
    """
    def __init__(self, problem_name: str, train: list[ArcSet], test: ArcSet):
        self._id = problem_name
        self._training_data: list[ArcSet] = train
        self._test: ArcSet = test

    def problem_name(self) -> str:
        """
        Returns the name of this ArcProblem.
        """
        return self._id

    def number_of_training_data_sets(self) -> int:
        """
        Returns the number of training input/output
        pairs for this test problem.
        """
        return len(self._training_data)

    def training_set(self) -> list[ArcSet]:
        """
        Returns all the training data as a list of ArcSets.
        """
        return deepcopy(self._training_data)

    def test_set(self) -> ArcSet:
        """
        Returns the test data as a dictionary
        with the keys of 'input' and 'output'
        """
        return deepcopy(self._test)
