from ArcData import ArcData


class ArcSet:
    """
    An ArcProblem that has both an
    input problem (ArcData) and a
    corresponding output (ArcData)
    for that particular input problem.

    The output represents either a
    transformation from the input
    or the answer to a test problem.
    """
    def __init__(self, arc_input: ArcData, arc_output: ArcData = None):
        self._input = arc_input
        self._output = arc_output

    def get_input_data(self) -> ArcData:
        """
        Returns the arc input
        data for this set.
        """
        return self._input

    def get_output_data(self) -> ArcData:
        """
        Returns the arc output
        data for this set if
        this set represents
        training type data otherwise
        this will return None.
        """
        return self._output

    def __eq__(self, other) -> bool:
        """
        Returns True if and only if
        both the input ArcData and
        the output ArcData have
        the same size and data;
        False otherwise.
        """
        return (self.get_input_data().__eq__(other.get_input_data())
                and self.get_output_data().__eq__(other.get_output_data()))

